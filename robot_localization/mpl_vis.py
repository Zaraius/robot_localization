#!/usr/bin/env python3

""" This is a simple node to visualize filter convergence and score over time using Matplotlib """
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib as plt
import matplotlib.pyplot as plt

class ConvergenceVis(Node):
    """
    Class to visualize filter convergence 

    Attributes:
        time: float array of timestamps of particle updates, converted to seconds
        mean: float array of mean particle scores over time
        std: float array of std deviation of particle scores over time
        x_std: float array of std deviations of particle X position over time
        y_std: float array of std deviations of particle Y position over time
        theta_std: float array of std deviations of particle theta over time

        time_sub, mean_sub, std_sub, x_std_sub, x_std_pub, theta_std_sub: subscribers to float arrays mentioned above,
            respectively
        
        timer: ROS2 timer with a callback to update the plot
        fig,axes: Matplotlib Figure object and Axes list describing the drawn figure
        line_x,line_y,line_theta: lines for plotting particle position and orientation convergence over time
    """
    def __init__(self):
        """ Initialize attributes and Matplotlib figure """
        self.time = []
        self.mean = []
        self.std = []
        self.x_std = []
        self.y_std = []
        self.theta_std = []

        super().__init__('vis')

        # initialize subscriptions
        self.time_sub = self.create_subscription(Float32MultiArray, "/metric/time", self.time_callback, 10)
        self.mean_sub = self.create_subscription(Float32MultiArray, "/metric/mean", self.mean_callback, 10)
        self.std_sub = self.create_subscription(Float32MultiArray, "/metric/std", self.std_callback, 10)
        self.x_std_sub = self.create_subscription(Float32MultiArray, "/metric/x_std", self.x_callback, 10)
        self.y_std_sub = self.create_subscription(Float32MultiArray, "/metric/y_std", self.y_callback, 10)
        self.theta_std_sub = self.create_subscription(Float32MultiArray, "/metric/theta_std", self.theta_callback, 10)
        self.timer = self.create_timer(2,self.update_plot)

        plt.ion()
        self.fig,self.axes = plt.subplots(1,2)
        plt.show(block=False)
        #self.line_mean, = self.axes[0].plot(range(len(self.mean)),self.mean)
        #self.line_ucb, = self.axes[0].plot(range(len(self.mean)),[self.mean[i] + 2*self.std[i] for i in range(len(self.mean))])
        #self.line_lcb, = self.axes[0].plot(range(len(self.mean)),[self.mean[i] - 2*self.std[i] for i in range(len(self.mean))])

        # initialize lines to plot displaying convergence of position and orientation
        self.line_x, = self.axes[0].plot(range(len(self.x_std)),self.x_std)
        self.line_y, = self.axes[0].plot(range(len(self.y_std)),self.y_std)
        self.line_theta, = self.axes[1].plot(range(len(self.theta_std)),self.theta_std)

        

    def update_plot(self):
        """
        Update plot by setting line data to current stored data and calling draw()
        """
        print("updating plot")
        self.axes[0].relim()
        self.axes[0].autoscale_view()
        #self.axes[0].legend(["Mean","Mean + 2 * Std Dev","Mean - 2 * Std Dev"])
        #self.axes[0].set_title("Mean and confidence bound for average particle score")
        self.axes[0].set_title("Standard deviation of particle X and Y over time")
        self.axes[0].legend(["X std dev (meters)","Y std dev (meters)"])
        self.axes[0].set_xlabel("Motion update iteration")
        self.axes[0].set_ylabel("Standard Deviation (meters)")

        self.axes[1].relim()
        self.axes[1].autoscale_view()
        self.axes[1].legend(["Theta std dev (rad)"])
        self.axes[1].set_title("Standard deviation of particle orientation over time")
        self.axes[1].set_xlabel("Motion update iteration")
        self.axes[1].set_ylabel("Standard Deviation (rad)")

        #self.line_mean.set_data(range(len(self.mean)),self.mean)
        #self.line_ucb.set_data(range(len(self.mean)),[self.mean[i] + 2*self.std[i] for i in range(len(self.mean))])
        #self.line_lcb.set_data(range(len(self.mean)),[self.mean[i] - 2*self.std[i] for i in range(len(self.mean))])

        self.line_x.set_data(range(len(self.x_std)),self.x_std)
        self.line_y.set_data(range(len(self.y_std)),self.y_std)
        self.line_theta.set_data(range(len(self.theta_std)),self.theta_std)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        plt.show(block=False)

    def time_callback(self,msg):
        """ Callback to record data timestamps 
        Args: msg (Float32MultiArray): array of data to log """
        self.time = msg.data

    def mean_callback(self,msg):
        """ Callback to record particle mean score
         Args: msg (Float32MultiArray): array of data to log """
        self.mean = msg.data

    def std_callback(self,msg):
        """ Callback to record particle std dev of score
         Args: msg (Float32MultiArray): array of data to log """
        self.std = msg.data

    def x_callback(self,msg):
        """ Callback to record particle x std deviation
         Args: msg (Float32MultiArray): array of data to log """
        self.x_std = msg.data

    def y_callback(self,msg):
        """ Callback to record particle y std deviation
         Args: msg (Float32MultiArray): array of data to log """
        self.y_std = msg.data

    def theta_callback(self,msg):
        """ Callback to record particle theta std deviation
         Args: msg (Float32MultiArray): array of data to log """
        self.theta_std = msg.data

def main():
    rclpy.init()
    vis = ConvergenceVis()
    rclpy.spin(vis)
    rclpy.shutdown()

if __name__ == "__main__":
    main()