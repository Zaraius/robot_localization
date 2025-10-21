#!/usr/bin/env python3

""" This is a simple node to visualize filter convergence and score over time using Matplotlib """
import rclpy
from threading import Thread
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Header, Float32MultiArray
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
import matplotlib.pyplot as plt

class ConvergenceVis(Node):
    """
    Class to visualize filter convergence 
    """
    def __init__(self):
        super().__init__('vis')
        self.time_sub = self.create_subscription(Float32MultiArray, "/metric/time", self.time_callback, 10)
        self.mean_sub = self.create_subscription(Float32MultiArray, "/metric/mean", self.mean_callback, 10)
        self.std_sub = self.create_subscription(Float32MultiArray, "/metric/std", self.std_callback, 10)
        self.x_std_sub = self.create_subscription(Float32MultiArray, "/metric/x_std", self.x_callback, 10)
        self.y_std_sub = self.create_subscription(Float32MultiArray, "/metric/y_std", self.y_callback, 10)
        self.theta_std_sub = self.create_subscription(Float32MultiArray, "/metric/theta_std", self.theta_callback, 10)
        self.timer = self.create_timer(2,self.update_plot)

        self.time = []
        self.mean = []
        self.std = []
        self.x_std = []
        self.y_std = []
        self.theta_std = []

        plt.ion()
        self.fig,self.axes = plt.subplots(1,2)
        plt.show(block=False)
        self.line_mean, = self.axes[0].plot(range(len(self.mean)),self.mean)
        self.line_ucb, = self.axes[0].plot(range(len(self.mean)),[self.mean[i] + 2*self.std[i] for i in range(len(self.mean))])
        self.line_lcb, = self.axes[0].plot(range(len(self.mean)),[self.mean[i] - 2*self.std[i] for i in range(len(self.mean))])

        self.line_x, = self.axes[1].plot(range(len(self.x_std)),self.x_std)
        self.line_y, = self.axes[1].plot(range(len(self.y_std)),self.y_std)
        self.line_theta, = self.axes[1].plot(range(len(self.theta_std)),self.theta_std)

        

    def update_plot(self):
        """
        Update plot by setting line data to current stored data and calling draw()
        """
        print("updating plot")
        self.axes[0].relim()
        self.axes[0].autoscale_view()
        self.axes[0].legend(["Mean","Mean + 2 * Std Dev","Mean - 2 * Std Dev"])
        self.axes[0].set_title("Mean and confidence bound for average particle score")

        self.axes[1].relim()
        self.axes[1].autoscale_view()
        self.axes[1].legend(["X std dev","Y std dev","Theta std dev"])
        self.axes[1].set_title("Standard deviation particle pose over time")

        self.line_mean.set_data(range(len(self.mean)),self.mean)
        self.line_ucb.set_data(range(len(self.mean)),[self.mean[i] + 2*self.std[i] for i in range(len(self.mean))])
        self.line_lcb.set_data(range(len(self.mean)),[self.mean[i] - 2*self.std[i] for i in range(len(self.mean))])

        self.line_x.set_data(range(len(self.x_std)),self.x_std)
        self.line_y.set_data(range(len(self.y_std)),self.y_std)
        self.line_theta.set_data(range(len(self.theta_std)),self.theta_std)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        plt.show(block=False)

    def time_callback(self,msg):
        """ Callback to record data timestamps"""
        self.time = msg.data

    def mean_callback(self,msg):
        """ Callback to record particle mean score """
        self.mean = msg.data

    def std_callback(self,msg):
        """ Callback to record particle std dev of score """
        self.std = msg.data

    def x_callback(self,msg):
        """ Callback to record particle x std deviation """
        self.x_std = msg.data

    def y_callback(self,msg):
        """ Callback to record particle y std deviation """
        self.y_std = msg.data

    def theta_callback(self,msg):
        """ Callback to record particle theta std deviation """
        self.theta_std = msg.data

def main():
    rclpy.init()
    vis = ConvergenceVis()
    rclpy.spin(vis)
    rclpy.shutdown()

if __name__ == "__main__":
    main()