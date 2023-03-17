#!/usr/bin/env python3
import sys
import rospy
from math import degrees
from geometry_msgs.msg import Twist
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class Project3:

    def __init__(self):
        self.init_node()
        self.init_subscriber()
        if "test" in sys.argv[1:]:
            self.test_prompt()

    def init_node(self):
        ### Initialize the ROS node here
        rospy.init_node("task2", anonymous= True)


    def init_subscriber(self):
        ### Initialize the subscriber 
        self.subscribe = rospy.Subscriber("/scan", LaserScan, self.callback)

    
    def store_scanner_data(self, data):
        ###Sets the range data to class object variable ranges
        self.ranges = data.ranges
        self.range_min = data.range_min
        self.inc = data.angle_increment 


    def find_state(self):
        ### Uses range data to determine states

        ranges = list(self.ranges)
        min_range = self.range_min
        increment = self.inc
        new_state = []
        minimum_distanceL = 1000
        minimum_distanceR = 1000
        minimum_distanceRF = 1000
        minimum_distanceF = 1000

        #replace invalid min with infinity
        for i in range(len(ranges)):
            if ranges[i] < min_range:
                ranges[i] = float('inf')
        
        for i in range(len(ranges)):
            angle = degrees(i* increment)
            
            #checking Left
            if angle >= 75 and angle <= 105: #Cone of 30
                if  ranges[i] < minimum_distanceL:
                    minimum_distanceL = ranges[i]
            
            #checking Right
            if angle >= 255 and angle <= 285: #Cone of 30
                if  ranges[i] < minimum_distanceR:
                    minimum_distanceR = ranges[i]
            #Checking RightFront
            if angle >= 300 and angle <= 330: #Cone of 30
                if  ranges[i] < minimum_distanceRF:
                    minimum_distanceRF = ranges[i]

            #checking Front
            if angle >= 0 and angle <= 15: #Cone of 30
                if  ranges[i] < minimum_distanceF:
                    minimum_distanceF = ranges[i]

            #checking Front
            if angle >= 345 and angle <= 360: #Cone of 30
                if  ranges[i] < minimum_distanceF:
                    minimum_distanceF = ranges[i]


        new_state.append(self.get_minimum_distance_index(minimum_distanceL))
        new_state.append(self.get_minimum_distance_index(minimum_distanceR))
        new_state.append(self.get_minimum_distance_index(minimum_distanceRF))
        new_state.append(self.get_minimum_distance_index(minimum_distanceF))
        return tuple(new_state)

    def get_minimum_distance_index(self, minimum_distance):
        ###Gets Uses minimum distance to find corosponding state index
        
        #define state ranges
        close = 0.25 # Close: x < 0.25
        far = 0.6 # Medium 0.25 <= x <= 0.6  #Far: x >0.6
        state_index = 0

        #Figure out range category
        if minimum_distance < close:
            state_index = 0
        elif minimum_distance >= close and minimum_distance <= far:
            state_index = 1
        else:
            state_index = 2

        return state_index
    

    #define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
    def get_next_action(self,state_index_tuple, epsilon):
        ###Gets the index of the next action based on greedy or random

        #Each corrosponding index of state
        left, right, right_front, front = state_index_tuple

        #Greedy
        if np.random.random() < epsilon:
            self.greedy = True
            return np.argmax(self.q_table[left, right, right_front, front])
        else: #choose a random action (Not Greedy)
            self.greedy = False
            return np.random.randint(self.num_actions)
        

    def move(self, action_index):
        ###Moves robot based on given action index
        movement = Twist()
        x = 0.1
        z = 0.0001

        if action_index == 1: # Right Turn
            z = 1
        if action_index == 2: #Left Turn
            z = -1
            
        movement.linear.x = x
        movement.angular.z = z
        
        #publishes movement to gazebo
        publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        publisher.publish(movement)

    def test_prompt(self):
        ###Gets file name for training
        args = sys.argv[1:]
        file_name = args[-1]
        self.q_table = np.load(file_name)
        print(self.q_table)
        self.test()

    def test(self):
        self.ranges = None
        self.previous_ranges = []
        
        #sets terminate to False at start of episode
        self.terminate = False

        #makes sure ranges is different than previous step and not Null
        while self.ranges is None or self.previous_ranges is self.ranges:
            pass
            
        #While robot is not terminated, test continues
        while True:
            #Makes sure ranges is uiniqe from previous step
            while self.previous_ranges is self.ranges:
                pass

            #sets previous range to current ranges
            self.previous_ranges = self.ranges

            #Get Robot's State
            state_tuple = self.find_state()

            #Each corrosponding index of state
            left, right, right_front, front = state_tuple
            #choose which action to take
            action_index = np.argmax(self.q_table[left, right, right_front, front])

            #perform the chosen action
            self.move(action_index)
        print('Testing complete!')


    def callback(self, data):
        self.store_scanner_data(data)
        
if __name__ == "__main__":
    Project3()