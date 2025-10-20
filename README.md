Steps to run

ros2 launch robot_localization test_pf.py map_yaml:=src/robot_localization/maps/mac_1st_floor_final.yaml

ros2 bag play src/robot_localization/bags/macfirst_floor_take_2/macfirst_floor_take_2_0.db3 --clock

rviz2 -d ~/ros2_ws/src/robot_localization/rviz/turtlebot_bag_files.rviz

There are two bag files inside of the /bags folder. We ran our particle filter on the input sensor data and recorded the output. They correspond to the two takes of the input data, particle-filter-take-1 is the output of running our particle filter on macfirst_floor_take_1 and particle-filter-take-2 is the output of macfirst_floor_take_2. The following topics are being recorded: /accel /odom /cmd_vel /tf /tf_static /base_footprint /map /scan /particle_cloud
The /odom topic is our estimated robot pose calculated from our particle cloud whose topic is /particle_cloud

### Project Goal

The goal of our project was to implent the core functionality of a particle filter, an algorithm for robot localization within a known map. The particle filter works by making a large number of initial guesses for the current robot pose, where each pose estimate is referred to as a particle. After the robot moves a meaningful distance, the filter uses the sensor data from the robot (in our case, a lidar scan) in order to weigh each particle based on how correct it seems. If the scan data closely aligns with what it hypothetically would be given the particle is a correct guess, the filter gives more weight to the particle; otherwise, the filter gives less weight to that particle. After weighing the particles, the filter normalizes the weights such that they sum to one (i.e., define a valid probability distribution) and chooses another set of particles from the initial particles, drawn with each particle's weight giving the probability of selecting it. To explore the entire state of possible estimates, we add noise to each new particle, normally distributed in each dimension, allowing guesses to shift and become more refined with time. 

The actual pose estimate of the robot is calculated as a weighted sum of the particle positions, where the contribution of each particle is proportional to its weight. This becomes weaker in the case of a bimodal or multimodal distribution, where clusters of particles may converge to near-equally valued guesses, but the mean is still simple to implement and valid for situations where the algorithm converges. 

One downside of the algorithm is the tendency for particles to converge to local optima that may be far away from a global optimum, especially if the map has symmetry to it. To avoid this, we add a small number of particles uniformly sampled from the space of all possible poses every iteration, ensuring we can escape convergence towards a poor guess. 

### Implementation / Solution

We implemented our particle filter 

### Design Decisions


### Challenges

One initial confusion we had was with the method of resampling particles. We were initially concerned that simple moving each particle and choosing some more than others based on weight would not be ideal, as more particles in a less optimal area would be favored over fewer particles in a more optimal area; however, we now realize that so long as a particle is higher weighted than the average particle, it witll be sampled equally or more in the next iteration, and will eventually dominate with time. We made this change after first observing the infeasibility of our first method, which was enumerating tuples of coordinates and linearly interpolating the particle weights. Not only would the interpolation make convergence difficult, but enumerating the indices was not at all feasible for the size of our map --- after \(semi-arbitrarily\) choosing to use 360 indices for angle, we calculated that our single list of indices would take over 6GB of RAM, and the interpolation would probably take forever, if we waited enough for it to finish running. 

Our initial implementation of the particle scoring step of the algorithm struggled with convergence despite running quickly, particularly with the angle converging. The simplest way of scoring every particle's location was to use the minimum distance to an obstacle as reported by the laser scan, compare that to the distance to the nearest obstacle at each particle's position, computed by the occupancy grid, and use the difference as an error to invert for weighting. However, angle is only encoded in this metric indirectly, by way of the motion update being dependent on angle, and thus we found our position would converge quickly but our angle would not converge as much.

To fix this, we implemented a more sophisticated method for weighing the particles based on projecting the detected obstacles from the laser scan from every particle into the map frame, using the distance to an obstacle at that point as the error given that every definite laser endpoint should occur at an obstacle. Each particle can then give us much more information, as many pieces as there are valid laser results from the scan, and angle is directly included in calculating the particle weight. We can also choose to use fewer than the max number of laser scan beams if we figure a smaller number of beams gives sufficient information for each particle. 

### Next Steps / Improvements

One feature we didn't implement for our project was any consideration of an initial pose estimate in the robot localization. Currently, we initialize our particles from a completely uniform distribution, without any information we might have over our robot position. While this is useful in some scenarios with limted information, particle filters can also be used for situations where we might have some idea of our inital pose, but we want to refine it or even just track it over time despite sensor and motion noise. Incorporating an initial pose estimate would allow for us to use any initial information we might have in more quickly converging. 

If we had more time, we could spend more time tuning the scoring of particles in a way that would give us more control over what laser scan endpoint discrepancies correspond to what weights. With the naive approach, we just take the reciprocal of each distance to obstacle at every endpoint and sum them together. However, the reciprocal inherently gives much more weight to low errors, and much less to further ones. We could try implementing a reward function / scoring function with more parameters to tune in case we wanted to change the distribution of weight given error. 

We would also like to make certain fixed parameters of our algorithm variable depending on the current state of the particle filter. For instance, we could have an adaptive particle count, where more particles are generated when the particles haven't converged as much, while fewer are generated when it does converge. We could also have the standard deviation of our particle generation noise, or the proportion of our random particles, decrease once our pose estimate converges --- this would be similar to optimization techniques like simulated annealing, where we lower a temperature parameter (our random particle proportions and standard deviations) over time in order to find a good spot to converge initially but hold ourselves tighter to it afterwards. 

### Lessons Learned