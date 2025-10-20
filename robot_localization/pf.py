#!/usr/bin/env python3

""" This is the starter code for the robot localization project """

import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav2_msgs.msg import ParticleCloud, Particle
from nav2_msgs.msg import Particle as Nav2Particle
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from rclpy.duration import Duration
import math
import time
import numpy as np
from occupancy_field import OccupancyField
from helper_functions import TFHelper, draw_random_sample
from rclpy.qos import qos_profile_sensor_data
from angle_helpers import quaternion_from_euler
from scipy.interpolate import griddata
import matplotlib as plt
import scipy.stats as sp

class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """ 
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        q = quaternion_from_euler(0, 0, self.theta)
        return Pose(position=Point(x=self.x, y=self.y, z=0.0),
                    orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

    # TODO: define additional helper functions if needed

class ParticleFilter(Node):
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            base_frame: the name of the robot base coordinate frame (should be "base_footprint" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            last_scan_timestamp: this is used to keep track of the clock when using bags
            scan_to_process: the scan that our run_loop should process next
            occupancy_field: this helper class allows you to query the map for distance to closest obstacle
            transform_helper: this helps with various transform operations (abstracting away the tf2 module)
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            thread: this thread runs your main loop
    """
    def __init__(self):
        super().__init__('pf')
        self.base_frame = "base_footprint"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from 

        self.n_particles = 5000          # was 300 the number of particles to use
        self.proportion_random = 0.01   # proportion of particles to randomly generate each iteration
        self.xy_std_dev = 0.01           # was 0.5 standard deviation of random changes to linear position
        self.theta_std_dev = 0.2        # was 0.3 standard deviation of random changes to orientation

        self.d_thresh = 0.2             # was 0.2 the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # was math.pi/6 the amount of angular movement before performing an update
        self.sigma = 0.5  # sensor/model noise, tune as needed
        self.eps = 1e-12
        # TODO: define additional constants if needed

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.update_initial_pose, 10)

        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = self.create_publisher(ParticleCloud, "particle_cloud", qos_profile_sensor_data)

        # laser_subscriber listens for data from the lidar
        self.create_subscription(LaserScan, self.scan_topic, self.scan_received, 10)

        # this is used to keep track of the timestamps coming from bag files
        # knowing this information helps us set the timestamp of our map -> odom
        # transform correctly
        self.last_scan_timestamp = None
        # this is the current scan that our run_loop should process
        self.scan_to_process = None
        # your particle cloud will go here
        self.particle_cloud = []
        self.sum_weights = 1
        self.weight_means = []
        self.weight_stds = []

        self.std_x = []
        self.std_y = []
        self.std_theta = []

        self.sample_count = 20

        self.current_odom_xy_theta = []
        self.occupancy_field = OccupancyField(self)
        self.transform_helper = TFHelper(self)

        # we are using a thread to work around single threaded execution bottleneck
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.transform_update_timer = self.create_timer(0.05, self.pub_latest_transform)

    def pub_latest_transform(self):
        """ This function takes care of sending out the map to odom transform """
        #print("\n\n\n\nPUB LATEST TRANSFORM\n\n\n\n")
        if self.last_scan_timestamp is None:
            return
        postdated_timestamp = Time.from_msg(self.last_scan_timestamp) + Duration(seconds=0.1)
        self.transform_helper.send_last_map_to_odom_transform(self.map_frame, self.odom_frame, postdated_timestamp)
        #print("\n\n\n\nPUB LATEST TRANSFORM HAS RUN\n\n\n\n")

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        while True:
            self.run_loop()
            time.sleep(0.1)

    def run_loop(self):
        """ This is the main run_loop of our particle filter.  It checks to see if
            any scans are ready and to be processed and will call several helper
            functions to complete the processing.
            
            You do not need to modify this function, but it is helpful to understand it.
        """
        #self.get_logger().info("Start of run loop")
        loop_time = time.perf_counter() 
        # THIS WILL NOT RUN IF YOU DON'T RUN THE BAG FILE
        if self.scan_to_process is None:
            return

        msg = self.scan_to_process

        (new_pose, delta_t) = self.transform_helper.get_matching_odom_pose(self.odom_frame,
                                                                           self.base_frame,
                                                                           msg.header.stamp)
        if new_pose is None:
            # we were unable to get the pose of the robot corresponding to the scan timestamp
            if delta_t is not None and delta_t < Duration(seconds=0.0):
                # we will never get this transform, since it is before our oldest one
                self.scan_to_process = None
            return

        (r, theta) = self.transform_helper.convert_scan_to_polar_in_robot_frame(msg, self.base_frame)
        # print("r[0]={0}, theta[0]={1}".format(r[0], theta[0]))
        # clear the current scan so that we can process the next one
        self.scan_to_process = None

        self.odom_pose = new_pose
        new_odom_xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)
        # print("x: {0}, y: {1}, yaw: {2}".format(*new_odom_xy_theta))

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom_xy_theta
            self.get_logger().info("No current odom")

        elif not self.particle_cloud:
            self.get_logger().info("No particle cloud")

            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud(msg.header.stamp)
        elif self.moved_far_enough_to_update(new_odom_xy_theta):
            self.get_logger().info("We've moved")

            # we have moved far enough to do an update!
            t_resample_start = time.perf_counter()
            self.update_particles_with_odom()    # update based on odometry
            
            print(f"Update Particle with Odom: {time.perf_counter() - t_resample_start}")
            t_resample_start = time.perf_counter()
            self.update_particles_with_laser_projection(r, theta)   # update based on laser scan
            self.calculate_convergence()
            # print(f"Convergence:\nMean: {self.weight_means},\nStd Dev: {self.weight_stds}")
            # print(f"Update Particle with Laser: {time.perf_counter() - t_resample_start}")
            t_resample_start = time.perf_counter()
            self.update_robot_pose()                # update robot's pose based on particles

            print(f"Update Robot Pose: {time.perf_counter() - t_resample_start}")
            t_resample_start = time.perf_counter()
            self.resample_particles()               # resample particles to focus on areas of high density
            print(f"Resample Particles: {time.perf_counter() - t_resample_start}")

        # publish particles (so things like rviz can see them)
        #print("about to publish particles")
        self.publish_particles(msg.header.stamp)
        # print(f"Total loop time: {time.perf_counter() - loop_time}")
    def moved_far_enough_to_update(self, new_odom_xy_theta):
        return math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or \
               math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh


    def update_robot_pose(self):
        print(f"Beginning update robot pose; there are {len(self.particle_cloud)} particles remaining")
        if not self.particle_cloud:
            return

        # normalize weights once
        total_w = sum(p.w for p in self.particle_cloud)
        if total_w <= 0:
            n = len(self.particle_cloud)
            for p in self.particle_cloud:
                p.w = 1.0 / n
            total_w = 1.0

        sum_x = sum_y = sum_sin = sum_cos = 0.0
        for p in self.particle_cloud:
            w = p.w / total_w
            sum_x += p.x * w
            sum_y += p.y * w
            sum_sin += math.sin(p.theta) * w
            sum_cos += math.cos(p.theta) * w

        theta = math.atan2(sum_sin, sum_cos)
        translation = [sum_x, sum_y, 0.0]
        rotation = quaternion_from_euler(0, 0, theta)
        new_pose = self.transform_helper.convert_translation_rotation_to_pose(translation, rotation)
        
        # highest_w = 0
        # for p in self.particle_cloud:
        #     if p.w > highest_w:
        #         highest_w = p.w
        #         x = p.x
        #         y = p.y
        #         theta = p.theta
        # translation = [x, y, 0.0]
        # rotation = quaternion_from_euler(0,0,theta)
        # new_pose = self.transform_helper.convert_translation_rotation_to_pose(translation, rotation)
    
        # TODO: assign the latest pose into self.robot_pose as a geometry_msgs.Pose object ZARAIUS
        # if hasattr(self, 'robot_pose'):
            # self.get_logger().info(f"Robots old position estimate is x: {self.robot_pose.position.x}, y: {self.robot_pose.position.y}, orientation: {self.robot_pose.orientation.z}")
            # self.get_logger().info(f"Robots new position estimate is {new_pose.position.x}, y: {new_pose.position.y}, orientation {new_pose.orientation.z}")
        # self.get_logger().info(f"Difference in position estimates is {self.robot_pose.position.x - self.new_pose.position.x}")


        self.robot_pose = new_pose
        if hasattr(self, 'odom_pose'):
            # print("YOU ARE NOT CRAZY\n\n\n")
            self.transform_helper.fix_map_to_odom_transform(self.robot_pose,
                                                            self.odom_pose)
        else:
            self.get_logger().warn("Can't set map->odom transform since no odom data received")

    def update_particles_with_odom(self):
        """Update each particle based on odometry delta using the rot1-trans-rot2 model."""
        print(f"Beginning update particles with odom; there are {len(self.particle_cloud)} particles remaining")
        new_odom = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)

        if not self.current_odom_xy_theta:
            self.current_odom_xy_theta = new_odom
            return

        old_x, old_y, old_theta = self.current_odom_xy_theta
        new_x, new_y, new_theta = new_odom

        # compute delta in odom frame (old -> new)
        dx = new_x - old_x
        dy = new_y - old_y
        trans = math.hypot(dx, dy)
        rot1 = math.atan2(dy, dx) - old_theta
        rot2 = (new_theta - old_theta) - rot1

        # normalize
        rot1 = (rot1 + math.pi) % (2*math.pi) - math.pi
        rot2 = (rot2 + math.pi) % (2*math.pi) - math.pi

        # Now update stored odom (after computing delta)
        self.current_odom_xy_theta = (new_x, new_y, new_theta)

        # Motion noise params (tune — see suggestions below)
        # Option: scale noise with trans and absolute rotation
        trans_std = 0.02 + 0.05 * abs(trans)    # example: base + proportional
        rot_std = 0.02 + 0.05 * (abs(rot1) + abs(rot2))

        for p in self.particle_cloud:
            r1 = rot1 + np.random.normal(0, rot_std)
            t  = trans + np.random.normal(0, trans_std)
            r2 = rot2 + np.random.normal(0, rot_std)

            p.x += t * math.cos(p.theta + r1)
            p.y += t * math.sin(p.theta + r1)
            p.theta += (r1 + r2)
            # normalize theta
            p.theta = (p.theta + math.pi) % (2 * math.pi) - math.pi

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample in helper_functions.py.
        """
        # t_resample_start = time.perf_counter()
        print(f"Beginning resample particles; there are {len(self.particle_cloud)} particles remaining")
        if not self.particle_cloud:
            return

        weights = [p.w for p in self.particle_cloud]
        # protect against all-zero weights
        total_w = sum(weights)
        if total_w <= 0:
            # reset to uniform small weights
            for p in self.particle_cloud:
                p.w = 1.0 / len(self.particle_cloud)
            weight_arr = np.array([p.w for p in self.particle_cloud])
        else:
            weight_arr = np.array(weights) / total_w

        n_random = int(self.n_particles * self.proportion_random)
        n_not_random = self.n_particles - n_random

        random_particles = self.generate_uniform_particles(n_random)
        new_particles = draw_random_sample(self.particle_cloud, weight_arr, n_not_random)

        self.particle_cloud = []
        for p in new_particles:
            # add x/y and theta perturbations (use correct std devs)
            p.x += np.random.normal(0, self.xy_std_dev)
            p.y += np.random.normal(0, self.xy_std_dev)
            # discard particles that go out of bounds
            if not self.check_particle_bounds(p):
                # Throw away out of bounds particles
                continue
            if self.occupancy_field.get_closest_obstacle_distance(p.x,p.y) <= 0: # why is this 1 and not 0?
                # Throw away particles in obstacle
                continue
            #while self.occupancy_field.get_closest_obstacle_distance(p.x,p.y) < 1:
            #    p.x += np.random.normal(0,self.xy_std_dev)
            #    p.y += np.random.normal(0,self.theta_std_dev)
            p.theta += np.random.normal(0,self.theta_std_dev)
            self.particle_cloud.append(p)

        # add valid random particles
        self.particle_cloud += random_particles
        
        # print(f"Resample elapsed time: {time.perf_counter() - t_resample_start}")

        #particle_arr = np.array(particles)
        #weight_arr = np.array(weights)

        #width = self.occupancy_field.map.info.width * self.occupancy_field.map.info.resolution
        #height = self.occupancy_field.map.info.height * self.occupancy_field.map.info.resolution
        #x_grid,y_grid,theta_grid = np.mgrid[0:width,0:height,0:360]
        #interp = griddata(particle_arr,weight_arr,(x_grid,y_grid,theta_grid),method="linear")

        # idx_list = list(np.ndindex(width,height,360)) # problematic
        #weight_interp = interp.ravel()
        #probabilities = (weight_interp / np.sum(weight_interp)).tolist()

        #self.particle_cloud = self.generate_valid_particles(idx_list,probabilities)
        #self.particle_cloud = new_particles
        # print(f"Resample elapsed time: {time.perf_counter() - t_resample_start}")

    def check_particle_bounds(self,p):
        res = self.occupancy_field.map.info.resolution
        width = self.occupancy_field.map.info.width
        height = self.occupancy_field.map.info.height
        start_x = self.occupancy_field.map.info.origin.position.x
        start_y = self.occupancy_field.map.info.origin.position.y
        #print(f"checking px {(p.x - start_x)/res}")
        #print(f"checking py {(p.y - start_y)/res}")
        #print(f"total width {width} total height {height}")
        if not (0 < (p.x - start_x)/res < width):
            return False
        if not (0 < (p.y - start_y)/res < height):
            return False
        return True

    def  update_particles_with_laser(self, r, theta):
        """ Updates the particle weights in response to the scan data
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame for each corresponding reading 
        """
        print(f"Beginning update particles with laser; there are {len(self.particle_cloud)} particles remaining")
        # Use a likelihood-field style sensor model: weight = exp(-0.5 * (dist_to_obstacle / sigma)^2)


        # Filter out-of-bounds particles first
        self.particle_cloud = [p for p in self.particle_cloud if self.check_particle_bounds(p)]

        for p in self.particle_cloud:
            p_distance = self.occupancy_field.get_closest_obstacle_distance(p.x, p.y)
            if (math.isnan(p_distance)):
                print(f"RAHRAHRH x {p.x} y {p.y}\n\n\n\n") 
                p.w = self.eps
            else:
                p.w = math.exp(-0.5 * (p_distance **2) / (self.sigma ** 2)) + self.eps
            #print(f"input is {p.x} and {p.y}")
            #print(f"p distance {p_distance}, min dist = {min_distance}")
    def update_particles_with_laser_projection(self,r,theta):
        """ Updates the particle weights in response to the scan data
            r: the distance readings to obstacles
            theta: the angle relative to the robot frame for each corresponding reading 

            Uses projections of laser scan points on the map frame to score
        """
        r_arr = np.asarray(r)
        theta_arr = np.asarray(theta)

        valid_r = (r_arr > self.scan_r_min) & (r_arr < self.scan_r_max)
        valid_idx = np.where(valid_r)[0]

        eps = 0.001

        # Exit early if no valid scan points found
        if valid_idx.size < 1:
            print("No valid laser scan points were found")
            return
        
        # Define how many laser scan points to use 
        if hasattr(self, "sample_count") and self.sample_count < valid_idx.size:
            sample_count = self.sample_count
        else:
            sample_count = valid_idx.size

        print(f"{valid_idx.size} valid sampling points; choosing {sample_count} of them")
        # If we're sampling fewer than our number of valid points, select some at random
        if valid_idx.size > sample_count:
            valid_idx = np.random.choice(valid_idx, size=sample_count, replace=False)

        for p in self.particle_cloud:
            px = p.x
            py = p.y
            p_theta = p.theta
            score = 0

            # Project laser scan points from current particle pose to get sampled distances
            for theta_index in valid_idx:
                angle = p_theta + theta_arr[theta_index]
                proj_x = px + r_arr[theta_index] * np.cos(angle)
                proj_y = py + r_arr[theta_index] * np.sin(angle)

                # Should be near zero if the laser scan is accurate
                err = self.occupancy_field.get_closest_obstacle_distance(proj_x,proj_y)
                if math.isnan(err):
                    err = 100
                # We may want to scale or pass err through a function at this point
                score += 1/max(err,eps)

            p.w = score / sample_count
            #print(f"Weight {score}, ",end=" ")

    def update_initial_pose(self, msg):
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(msg.header.stamp, xy_theta)

    def generate_uniform_particles(self,count):
            """
            Return a list of valid particles drawn from a uniform distribution
            Used both in initialization as well as in small quantities every step
            """
            # Initialize particles w/ uniform distribution, map frame
            res = self.occupancy_field.map.info.resolution
            width_m = self.occupancy_field.map.info.width * res # physical units
            height_m = self.occupancy_field.map.info.height * res # physical units
            start_x = self.occupancy_field.map.info.origin.position.x
            start_y = self.occupancy_field.map.info.origin.position.y

            # number of cells (used for bounds checking)
            width_cells = self.occupancy_field.map.info.width
            height_cells = self.occupancy_field.map.info.height

            x_rand = np.random.uniform(low=start_x,high=start_x+width_m,size=(count))
            y_rand = np.random.uniform(low=start_y,high=start_y+height_m,size=(count))
            theta_rand = np.random.uniform(low=-1*math.pi,high=math.pi,size=(count))

            p_list = []
            for i in range(count):
                # retry until particle is not inside obstacle and within bounds
                tries = 0
                while (self.occupancy_field.get_closest_obstacle_distance(x_rand[i],y_rand[i]) < 1) or \
                      (not 0 < (x_rand[i]-start_x)/res < width_cells) or \
                      (not 0 < (y_rand[i]-start_y)/res < height_cells):
                    x_rand[i] = np.random.uniform(low=start_x,high=start_x+width_m)
                    y_rand[i] = np.random.uniform(low=start_y,high=start_y+height_m)
                    tries += 1
                    if tries > 200:
                        # fallback: break to avoid infinite loop on pathological maps
                        break
                p_list.append(Particle(x_rand[i],y_rand[i],theta_rand[i],1.0/count))
            return p_list

    def initialize_particle_cloud(self, timestamp, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is omitted, the odometry will be used """
        if xy_theta is None:
            xy_theta = self.transform_helper.convert_pose_to_xy_and_theta(self.odom_pose)

        self.particle_cloud = self.generate_uniform_particles(self.n_particles)
        self.update_robot_pose()

    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """
        print(f"Beginning normalize particles; there are {len(self.particle_cloud)} particles remaining")
        weights = [p.w for p in self.particle_cloud]
        total = sum(weights)
        if total <= 0:
            # avoid division by zero — assign uniform weights
            n = max(1, len(self.particle_cloud))
            self.particle_cloud = [Particle(p.x,p.y,p.theta,1.0/n) for p in self.particle_cloud]
            self.sum_weights = 1.0
            return

        self.sum_weights = total
        new_particles = []
        for p in self.particle_cloud:
            new_particles.append(Particle(p.x,p.y,p.theta,p.w/total))
        self.particle_cloud = new_particles

    def generate_valid_particles(self,choices,probabilities):
        """ 
        Sample particles until all are not within obstacles
        """
        print("starting generate valid particles")
        samples = draw_random_sample(choices,probabilities,self.n_particles)
        #print("drew random sample")
        new_particle_list = []
        
        for i,sample in enumerate(samples):
            sample_x,sample_y,_ = sample
            while self.occupancy_field.get_closest_obstacle_difference(sample_x,sample_y) < 1:
                samples[i] = draw_random_sample(choices,probabilities,1)
                sample_x,sample_y = samples[i]
            new_particle_list.append(Particle(samples[i][0],samples[i][1],samples[i][2],1.0))
        
        return new_particle_list
    
    def calculate_convergence(self):
        """
        Calculates mean and std dev of particle scores
        """
        x = [p.x for p in self.particle_cloud]
        y = [p.y for p in self.particle_cloud]
        theta = [p.theta for p in self.particle_cloud]

        self.std_x.append(np.std(x))
        self.std_x.append(np.std(y))
        self.std_x.append(np.std(theta))

        unscaled_weights = [p.w * self.sum_weights for p in self.particle_cloud]
        mean = np.mean(unscaled_weights)
        std = np.std(unscaled_weights)

        self.weight_means.append(mean)
        self.weight_stds.append(std)

    def publish_particles(self, timestamp):
        msg = ParticleCloud()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = timestamp
        for p in self.particle_cloud:
            msg.particles.append(Nav2Particle(pose=p.as_pose(), weight=p.w))
        self.particle_pub.publish(msg)

    def scan_received(self, msg):
        self.last_scan_timestamp = msg.header.stamp
        # we throw away scans until we are done processing the previous scan
        # self.scan_to_process is set to None in the run_loop 
        if self.scan_to_process is None:
            self.scan_to_process = msg
            self.scan_r_min = msg.range_min
            self.scan_r_max = msg.range_max

def main(args=None):
    rclpy.init()
    n = ParticleFilter()
    rclpy.spin(n)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
