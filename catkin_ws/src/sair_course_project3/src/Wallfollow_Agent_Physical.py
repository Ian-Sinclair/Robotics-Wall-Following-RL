#!/usr/bin/env python3

import rospy
import math

# For reading command line inputs
from sys import argv
import sys
import os

# For creating data structures for q-learning
import numpy as np

# For Euler-quaternion transformation
# from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# For manual debug input
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
from select import select

# Used in graphing learning info
from matplotlib import pyplot as plt

# Used in list copying
from copy import deepcopy

# Bring in geometry structures necessary to give movement commands
from geometry_msgs.msg import Pose, Point, Quaternion, Twist

# Used to get info from the Laser sensor
from sensor_msgs.msg import LaserScan

# Used to set/get Gazebo Model State
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState

# Use this to reset the simulation
from std_srvs.srv import Empty


"""
Possible states:
    State consists of Front, RightFront, Right, and Left
    States can be one of:
        C - Close
        M - Medium
        F - Far
    State space (3^4 = 81):
        ["C/M/F", "C/M/F", "C/M/F", "C/M/F"]
"""

"""
Possible actions:
    Possible directions:
        S - Straight
        R - Right
        L - Left
    Possible distances:
        S - Short
        L - Long
    Action space (3^1 * 2^1 = 6):
        SS - Straight short
        SL - Straight long
        RS - Right short
        RL - Right long
        LS - Left short
        LL - Left long
"""

class WallfollowAgent:
    """
    A class that:
        Utilizes Reinforcement Q-Learning to follow a right-hand wall.

    Member Variables
    ----------------
    @var: name : type
        Description
    """

    #region Class initialization

    # Default the init parameters to the values of the /map OccupancyGrid
    def __init__(self):
        self.front_states = 3 # Front = C/M/F
        self.frontright_states = 3 # Front = C/M/F
        self.right_states = 3 # Right = C/M/F
        self.left_states = 3 # Left = C/M/F
        
        self.current_laser_data = None # The currently in use data from the laser scanner
        self.new_laser_data = None # Stores data from the Laser scanner of type LaserScan

        is_training = True # Default to training mode
        is_test = False # Allow to set test mode
        self.manual_control = False # Is the robot being manually controlled?

        # Perform object initialization
        self.init_node()
        self.init_publishers()
        self.init_subscriber()
        self.init_gazebo()

        # Interpret command line inputs to figure out if we want to run in training mode or test mode
        is_training, is_test, load_recent_qtable, load_best_qtable, self.manual_control = self.handle_cli_arguments()

        # There are 6 total actions the bot can take
        self.actions = ["SS", "SL", "RS", "RL", "LS", "LL"]

        # If we are running in training mode
        if is_training:
            rospy.loginfo("Initializing Wallfollow Agent in Training mode...")
            # Initialize learning
            self.initialize_qlearning(load_recent_qtable, load_best_qtable)

            # Begin the task of autonomous wallfollow navigation training
            self.wallfollow_navigate(True)
        elif is_test:
            rospy.loginfo("Initializing Wallfollow Agent in Test mode...")
            
            # Load saved q-table
            self.initialize_qlearning(load_recent_qtable) # Will load recent if not specified, but will load best otherwise

            # Enter test mode
            self.wallfollow_navigate(False)

    def handle_cli_arguments(self)->tuple:
        training = True # Default to training mode
        test = False # Allow to set test mode
        load_recent_qtable = False # Should we load the most recent existing q file
        load_best_qtable = False # Should we load the best qtable file
        manual = False # Is the robot being manually controlled?

        # rospy.loginfo(f"ARGV: {len(argv)}, {argv}")
        if len(argv) == 2 or len(argv) == 3: # Exactly one or two cli arguments
            try:
                # Determine mode based on arguments
                manual = (argv[1].casefold() == "manual")
                training = (argv[1].casefold() == "training") or manual
                test = (argv[1].casefold() == "test")

                # Check if we should load a file
                if len(argv) == 3:
                    load_recent_qtable = (argv[2].casefold() == "load")
                    load_best_qtable = (argv[2].casefold() == "best")

                # Output mode selection
                rospy.loginfo(f"Training mode? {training}")
                rospy.loginfo(f"Test mode? {test}")
                rospy.loginfo(f"Load recent Q Table? {load_recent_qtable}")
                rospy.loginfo(f"Load best Q Table? {load_best_qtable}")
                rospy.loginfo(f"Manual control? {manual}")
            except Exception:
                rospy.logwarn(f"Invalid command line arguments given, defaulting to training mode.")
        elif len(argv) != 1: # Wrong amount of cli arguments
            rospy.logwarn(f"Incorrect amount of command line arguments given, defaulting to training mode.")
        else: # No cli arguments
            rospy.loginfo(f"No command line arguments given, defaulting to training mode.")
        
        # Return parsing results
        return training, test, load_recent_qtable, load_best_qtable, manual

    #endregion

    #region Object Initialization

    def init_node(self)->None:
        """ Initializes a rospy node so we can publish and subscribe over ROS. """
        rospy.init_node('wallfollow_agent_py')

    def init_publishers(self)->None:
        """ Initializes any publishers we'll need in this agent. """
        # Initialize the publisher that publishes movement commands to the robot
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def init_subscriber(self)->None:
        """ Initializes any subscribers we'll need in this agent. """
        # Initialize the subscriber that consumes the Laser Sensor info
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.set_laser_data)

    #endregion

    #region Q-Learning

    def initialize_qlearning(self, load_recent_qtable:bool, load_best_qtable:bool = True)->None:
        """
        Initializes all data and data structures necessary for the q-learning process.
        """
        # Init function values
        self.epsilon = 0.9 # Percent of time to take best action
        self.discount_factor = 0.8 # (Gamma) Discount factor for future rewards
        self.learning_rate = 0.2 # (Alpha) Rate at which the AI should learn

        # Init actions
            #0 = SS - Straight short
            #1 = SL - Straight long
            #2 = RS - Right short
            #3 = RL - Right long
            #4 = LS - Left short
            #5 = LL - Left long

        # Init Q Table
        # First 4 dimensions represent the state, 5th dimension represents the action
        if load_recent_qtable:
            self.load_q_table("q-table.npy")
            rospy.loginfo("Loading most recent q-table!")
        elif load_best_qtable:
            self.load_q_table("q-table_best.npy")
            rospy.loginfo("Loading best q-table!")
        else: # Init q-table to all zeroes
            self.q_table = np.zeros((self.front_states, self.frontright_states, self.right_states, self.left_states, len(self.actions)))
            rospy.loginfo("Initializing q-table to all zeroes!")

        # # PART 1 STUFF START

        # # Set q table values manually for straight-wall following
        # self.manual_q_table_straight_wall()

        # # PART 1 STUFF END

        # Init the current state
        # State order is Front, FrontRight, Right, Left
        self.current_state = [0, 0, 0, 0] # Default current state to all closest values 
        self.prev_state = [0, 0, 0, 0] # Default last state to all closest values 

    # PART 1 STUFF START

    # def manual_q_table_straight_wall(self):
    #     """
    #     Manually define a Q-Table policy for following a straight wall.
    #     self.q_table[Front, Frontright, Right, Left]
    #     Front:      0=C,    1=M,    2=F
    #     Frontright: 0=C,    1=M,    2=F
    #     Right:      0=C,    1=M,    2=F
    #     Left:       0=C,    1=M,    2=F
    #     Actions:
    #         0 = SS - Straight short
    #         1 = SL - Straight long
    #         2 = RS - Right short
    #         3 = RL - Right long
    #         4 = LS - Left short
    #         5 = LL - Left long
    #     """
    #     # With manually defined values, we want to listen to our q-table all of the time
    #     self.epsilon = 1.0 # Percent of time to take best action
        
    #     # For all front values
    #     for front in range(3):
    #         # For all left values
    #         for left in range(2):
    #             self.q_table[front, 0, 0, left, 0] = 1.0 # Straight Short if near right wall with good angle
    #             self.q_table[front, 0, 1, left, 0] = 1.0 # Straight Short if moving towards right wall
    #             # self.q_table[front, 0, 1, left, 4] = 1.0 # Left Short if moving towards right wall
    #             self.q_table[front, 0, 2, left, 1] = 1.0 # Straight Long if near right wall but moving away
    #             # self.q_table[front, 0, 2, left, 5] = 1.0 # Left Long if near right wall but moving away
    #             self.q_table[front, 1, 0, left, 2] = 1.0 # Right Short if near right wall and moving away
    #             self.q_table[front, 1, 1, left, 1] = 1.0 # Straight long if near right wall with good angle
    #             self.q_table[front, 1, 2, left, 1] = 1.0 # Straight Long if away from wall (move to wall)

    # PART 1 STUFF END

    #endregion

    #region Wallfollow Navigation

    #region Perform Wallfollow

    def set_laser_data(self, data:LaserScan)->None:
        """
        Callback function that receives the LaserData from self.laser_sub.
        """
        # Receive the LaserData
        self.new_laser_data = data

    def wallfollow_navigate(self, is_training:bool, num_episodes:int = 250, show_learning_graphs:bool = True)->None:
        """
        Run num_episodes training episodes.
        """
        # Make sure not to do anything until initial laser data is retrieved
        try: # Make it so that we don't crash on ROSTimeMovedBackwardsException
            while self.new_laser_data is None and not rospy.is_shutdown():
                rospy.loginfo("Waiting to receive initial laser scan data.")
                rospy.Rate(2).sleep()
            if not self.new_laser_data is None:
                rospy.loginfo("Initial laser scan data retrieved!")
                # Initialize laser data
                self.current_laser_data = self.new_laser_data
        except: pass

        # Handle rospy shutdown
        if self.handle_rospy_shutdown():
            return None

        if is_training:
            # Define values for epsilon scaling
            start_epsilon = 0.1
            end_epsilon = 0.9
            step_epsilon = abs(end_epsilon - start_epsilon) / num_episodes
            self.epsilon = start_epsilon
        else:
            # Always take the action defined in the q-table
            self.epsilon = 1.0

        if is_training:
            # Initialize data for learning measurement
            self.init_learning_measurement(num_episodes)

            # Initialize data for learning plotting
            fig, plot_lines, action_labels, colorset = self.init_plotting(num_episodes)

        # For num_episodes amount of episodes
        current_episode_number = 0
        last_episode_number = -1
        while current_episode_number < num_episodes:
            # If this is a new episode
            if not last_episode_number == current_episode_number:
                rospy.loginfo("------------------------------------")
                rospy.loginfo(f"Beginning episode {current_episode_number+1}!")
                last_episode_number = current_episode_number
            else: # If this is an episode continuation
                rospy.loginfo(f"Continuing episode {current_episode_number+1}!")

            # Choose a valid starting position, and get the state
            try:
                # Commented for project 4
                # self.initialize_wallfollow()
                self.current_state = self.get_current_state()
            except: pass

            # Make sure we're in a nonterminal position
            has_collided = self.check_for_collision()
            try: # Make it so that we don't crash on ROSTimeMovedBackwardsException
                while has_collided and not rospy.is_shutdown():
                    # Wait until position resetting has occurred
                    rospy.Rate(5).sleep()
                    has_collided = self.check_for_collision()
            except: pass

            # If we are far away on all sides, then we are lost
            robot_is_lost = self.is_lost()
            try: # Make it so that we don't crash on ROSTimeMovedBackwardsException
                while robot_is_lost and not rospy.is_shutdown(): # Make sure we're nonterminal
                    # Wait until position resetting has occurred
                    rospy.Rate(5).sleep()
                    robot_is_lost = self.is_lost()
            except: pass

            # Handle rospy shutdown
            if self.handle_rospy_shutdown():
               return None

            # While we haven't collided with an obstacle or gotten lost
            is_terminal = has_collided or robot_is_lost
            while not is_terminal and not rospy.is_shutdown():
                # Get the laser data to use for this step
                self.current_laser_data = self.new_laser_data

                # Choose an action given the current state using epsilon greedy
                action_index, greedy_choice_made = self.get_next_action()

                print(f"A {action_index}")

                # Manual input for testing
                if self.manual_control:
                    manual_input = self.manual_input()
                    action_index = manual_input

                print(f"B {is_training}")

                if is_training:
                    # Unpause physics before performing action
                    self.gz_unpause_physics()

                print("C")

                # Perform the chosen action, update the current state.
                if not action_index == -1:
                    self.perform_action(self.actions[action_index]) # Action selected via epsilon greedy
                else:
                    self.perform_action("XX") # Control signal to stop moving

                print("D")

                # Commented for Project 4
                # # Delay until next action
                # try:
                #     rospy.Rate(2).sleep()
                # except: pass

                # Handle rospy shutdown
                if self.handle_rospy_shutdown():
                    return None

                print("E")

                if is_training:
                    # Pause physics to perform calculations
                    self.gz_pause_physics()

                print("F")

                # Store the old state and update the current state
                self.prev_state = deepcopy(self.current_state)
                self.current_state = self.get_current_state()

                print("G")

                # Check to see if new state is terminal (and start new episode if so)
                has_collided = self.check_for_collision()
                robot_is_lost = self.is_lost()
                is_terminal = has_collided or robot_is_lost

                print(f"H {has_collided} {robot_is_lost} {is_terminal}")

                if is_training:
                    # Receive reward for entering state, calculate temporal difference
                    reward_value = self.get_reward_value(self.prev_state, self.current_state, action_index, has_collided)
                    rospy.loginfo(f"Reward Obtained: {reward_value}")
                    old_q_value = self.q_table[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], action_index]
                    # new_max_q_val = np.max(self.q_table[self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]])
                    curr_q_state = self.q_table[self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]]
                    new_max_q_val = max([curr_q_state[a] for a in range(len(self.actions))]) 
                    temporal_difference = (reward_value + (self.discount_factor * new_max_q_val)) - old_q_value

                    # Update Q-value for previous state action pair
                    new_q_value = old_q_value + (self.learning_rate * temporal_difference)
                    self.q_table[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], action_index] = new_q_value

                if is_training:
                    # Measure learning rate
                    self.check_correct_action(self.prev_state, action_index, num_episodes, current_episode_number, greedy_choice_made)

            # Only move onto the next episode if there were at least 5 actions with known correct answers in each state
            starting_new_episode = True
            if is_training:
                known_state_min_actions = 5
                achieved_known_state_min_actions = all([self.total_actions[current_episode_number][a] >= known_state_min_actions for a in range(len(self.total_actions[current_episode_number]))])
                starting_new_episode = achieved_known_state_min_actions
                rospy.loginfo(f"At least {known_state_min_actions} actions taken in each known state?: {achieved_known_state_min_actions}: {self.total_actions[current_episode_number]}")
            if starting_new_episode:
                current_episode_number += 1

                # Graph learning rate
                if show_learning_graphs and is_training:
                    self.plot_learning(self.learning_rates_percentage, fig, plot_lines, action_labels, colorset)

                if is_training:
                    # Increment epsilon at the end of each episode
                    self.epsilon += step_epsilon
                    # print(f"New epsilon: {self.epsilon}")

                    # Save q-table to file
                    self.save_q_table("q-table.npy")
            else:
                continue
        
            # Handle rospy shutdown
            if self.handle_rospy_shutdown():
               return None

            # Tell robot to stop moving
            self.perform_action("XX") # Control signal to stop moving
            
            # Handle collision
            if has_collided:
                rospy.loginfo("Agent has collided with an obstacle!")
            elif robot_is_lost:
                rospy.loginfo("Agent strayed too far from obstacles!")

        if is_training:
            rospy.loginfo("------------------------------------")
            rospy.loginfo(f"Completed {num_episodes} episode(s) of training!")
            rospy.loginfo(f"Displaying results in test mode:")

            # Enter test mode
            self.wallfollow_navigate(False)
        else:
            rospy.loginfo("Test mode demonstration concluded.")

    #endregion

    #region Wallfollow State Setting and Checking

    def initialize_wallfollow(self)->None:
        """ Reset the simulation and set the robot's starting position. """
        # Reset the Gazebo environment
        rospy.loginfo("Resetting Gazebo simulation!")
        self.gz_reset_simulation()

        # Move robot to given starting location (pause physics when moving)
        self.gz_pause_physics()
        x_pos, y_pos, z_rot = self.get_starting_location()
        self.gz_set_model_state(x_pos, y_pos, z_rot)
        rospy.loginfo(f"Initializing robot to position ({round(x_pos, 2)}, {round(y_pos, 2)}) with z rotation {round(z_rot, 2)}!")
        self.gz_unpause_physics()

    def get_starting_location(self)->tuple:
        """
        Returns a random starting location from a list of set locations of the form (x_pos, y_pos, z_rot).
        For rotation, 0 = up, 90 = left, 180 = down, 270 = right
        """
        # List of starting locations of the form (x_pos, y_pos, z_rot)
        starting_locations = np.array([
            # (0.75, 2.0, 180.0), # Test straight wall
            # (0.85, 0.75, 135.0), # Test diagonal into straight wall
            (2.0, 1.0, 90.0), # Top Left, approaching U-turn
            (2.0, 0.0, 90.0), # Top Center, approaching U-turn
            (1.0, -2.0, 0.0), # Center Right, approaching corner
            (0.0, 2.0, 180.0), # Center Left, approaching corner
            (-2.0, -2.0, 0.0), # Bottom Right, approaching I turn
            (-2.0, 2.0, 270.0), # Bottom Left, approaching I turn
            ])
        return starting_locations[np.random.randint(0, len(starting_locations))]

    def check_for_collision(self, collision_range:float = 0.165)->bool:
        """
        Examines the LaserData and determines if we're close enough to an obstacle in any direction that we have crashed.

        Parameters
        ----------
        collision_range : float
            The minimum distance that the laser sensor must detect for there for be a collision.

        Returns
        -------
        bool : True if we have crashed, and false otherwise.
        """
        for i in range(len(self.current_laser_data.ranges)):
            # Check additional 0 case for project 4
            if self.current_laser_data.ranges[i] <= collision_range and (not self.current_laser_data.ranges[i] == 0.0):
                return True
        return False
    
    def is_lost(self, lost_range:float = 1.0):
        """
        Examines the LaserData and determines if we're far away enough from all obstacles in any direction that we are lost.

        Parameters
        ----------
        lost_range : float
            The minimum distance that the laser sensor must detect in all directions for the robot to be lost.

        Returns
        -------
        bool : True if we are lost, and false otherwise.
        """
        for i in range(len(self.current_laser_data.ranges)):
            if self.current_laser_data.ranges[i] <= lost_range and (not self.current_laser_data.ranges[i] == 0.0):
                return False
        return True

    def get_current_state(self)->list:
        """
        Takes measurements from the Laser sensor and determines the current state.
        """
        # Calculate the ranges
        range_front = float('inf')
        range_frontright = float('inf')
        range_right = float('inf')
        range_left = float('inf')

        # Project 4 stuff
        # A list in which to store the directions that correspond to the laser measurements
        num_laser_ranges = len(self.current_laser_data.ranges)
        direction_angles = []

        # Store the converted angular directions in the list
        for i in range(num_laser_ranges):
            radians_of_current_measure = i * self.current_laser_data.angle_increment
            degrees_of_current_measure = math.degrees(radians_of_current_measure)
            direction_angles.append(degrees_of_current_measure)

        # Init ranges
        range_front = float('inf')
        range_frontright = float('inf')
        range_right = float('inf')
        range_left = float('inf')

        # Take 7 measurements on each side (15 total) and take the min for each direction
        # Added 0 distance checks for project 4
        # Added checks for when less than 360 laser ranges are given
        half_cone = 15/2 # Half the angle of the directional cone
        for i in range(num_laser_ranges):
            if (not self.current_laser_data.ranges[i] == 0.0):
                # Front cone
                if direction_angles[i] <= (0 + half_cone) or direction_angles[i] >= (360 - half_cone):
                    if self.current_laser_data.ranges[i] < range_front:
                        range_front = self.current_laser_data.ranges[i]     
                # Frontright cone
                elif direction_angles[i] <= (299 + half_cone) and direction_angles[i] >= (299 - half_cone):
                    if self.current_laser_data.ranges[i] < range_frontright:
                        range_frontright = self.current_laser_data.ranges[i]
                # Right cone
                elif direction_angles[i] <= (269 + half_cone) and direction_angles[i] >= (269 - half_cone):
                    if self.current_laser_data.ranges[i] < range_right:
                        range_right = self.current_laser_data.ranges[i]
                # Left cone
                elif direction_angles[i] <= (89 + half_cone) and direction_angles[i] >= (89 - half_cone):
                    if self.current_laser_data.ranges[i] < range_left:
                        range_left = self.current_laser_data.ranges[i]



        # for i in range (-7,8):
        #     if self.current_laser_data.ranges[0 + i] < range_front and not self.current_laser_data.ranges[0 + i] == 0.0:
        #         range_front = self.current_laser_data.ranges[0 + i]
        #     if self.current_laser_data.ranges[299 + i] < range_frontright and not self.current_laser_data.ranges[299 + i] == 0.0:
        #         range_frontright = self.current_laser_data.ranges[299 + i]
        #     if self.current_laser_data.ranges[269 + i] < range_right and not self.current_laser_data.ranges[269 + i] == 0.0:
        #         range_right = self.current_laser_data.ranges[269 + i]
        #     if self.current_laser_data.ranges[89 + i] < range_left and not self.current_laser_data.ranges[89 + i] == 0.0:
        #         range_left = self.current_laser_data.ranges[89 + i]

        # Categorize the ranges
        cat_front = self.categorize_front(range_front)
        cat_frontright = self.categorize_frontright(range_frontright)
        cat_right = self.categorize_right(range_right)
        cat_left = self.categorize_left(range_left)

        # Return the categorizations as the state
        # rospy.loginfo(f"State distances: F:{cat_front}:{round(range_front,2)}, FR:{cat_frontright}:{round(range_frontright,2)}, R:{cat_right}:{round(range_right,2)}, L:{cat_left}:{round(range_left,2)}")
        return [cat_front, cat_frontright, cat_right, cat_left]

    #endregion

    #region Directional Distance Categorizations

    def categorize_front(self, dist:float)->int:
        """
        Given a distance for the front sensor, categorize it.
        0=C, 1=M, 2=F
        """
        # Handle inf case
        # Check additional 0 case for project 4
        if dist == float("inf") or dist == 0.0:
            return 2

        # Perform categorization
        if dist < 0.35:
            return 0
        elif dist <= 0.7:
            return 1
        else:
            return 2

    def categorize_frontright(self, dist:float)->str:
        """
        Given a distance for the frontright sensor, categorize it.
        0=C, 1=M, 2=F
        """
        # Handle inf case
        # Check additional 0 case for project 4
        if dist == float("inf") or dist == 0.0:
            return 2

        # Perform categorization
        if dist < 0.3:
            return 0
        elif dist <= 0.6:
            return 1
        else:
            return 2

    def categorize_right(self, dist:float)->str:
        """
        Given a distance for the right sensor, categorize it.
        0=C, 1=M, 2=F
        """
        # Handle inf case
        # Check additional 0 case for project 4
        if dist == float("inf") or dist == 0.0:
            return 2

        # Perform categorization
        if dist < 0.275:
            return 0
        elif dist <= 0.525:
            return 1
        else:
            return 2

    def categorize_left(self, dist:float)->str:
        """
        Given a distance for the left sensor, categorize it.
        0=C, 1=M, 2=F
        """
        # Handle inf case
        # Check additional 0 case for project 4
        if dist == float("inf") or dist == 0.0:
            return 2

        # Perform categorization
        if dist < 0.3:
            return 0
        elif dist <= 0.6:
            return 1
        else:
            return 2

    #endregion

    #region Manual Input

    def manual_input(self)->int:
        """
        Checks for keyboard inputs and returns action indices or -1 if no input.
        """
        # Get a key press
        key = self.get_key()
        if key == 's':  # if key 's' is pressed 
            # rospy.loginfo('You Pressed the S Key!')
            return 0
        elif key == 'w':  # if key 'w' is pressed 
            # rospy.loginfo('You Pressed the W Key!')
            return 1
        elif key == 'd':  # if key 'd' is pressed 
            # rospy.loginfo('You Pressed the D Key!')
            return 2
        elif key == 'e':  # if key 'e' is pressed 
            # rospy.loginfo('You Pressed the E Key!')
            return 3
        elif key == 'a':  # if key 'a' is pressed 
            # rospy.loginfo('You Pressed the A Key!')
            return 4
        elif key == 'q':  # if key 'q' is pressed 
            # rospy.loginfo('You Pressed the Q Key!')
            return 5
        
        # No keys were pressed
        return -1

    def get_key(self)->str:
        settings = self.saveTerminalSettings()
        timeout = rospy.get_param("~key_timeout", 0.5)

        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            rlist, _, _ = select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def saveTerminalSettings(self):
        if sys.platform == 'win32':
            return None
        return termios.tcgetattr(sys.stdin)

    #endregion

    #region Actions

    def perform_action(self, action:str)->None:
        """
        Perform the given action.
        """
        # # Get the robot's orientation
        # pos, ori = self.gz_get_model_state()
        # # Get the Euler z rotation from the quaternion ori
        # rot = euler_from_quaternion([ori.w,ori.x,ori.y,ori.z])
        # forward_vector = [math.cos(rot[2]), math.sin(rot[2])]
        # forward_vector = [1,1]

        # print(f"Perform Action: {action}")

        print("10")

        # Create a twist to hold movement info
        move_twist = Twist()
        move_twist.linear.x = 0
        move_twist.linear.y = 0
        move_twist.linear.z = 0
        move_twist.angular.x = 0
        move_twist.angular.y = 0
        move_twist.angular.z = 0

        print("11")

        # Angular velocity
        turn_mod = 3.0 if action[1] == "S" else 4.5
        if action[0] == "R": # Right Turn
            # Short straight move
            move_twist.angular.z = math.radians(-12.5 * turn_mod)
        elif action[0] == "L": # Left Turn
            # Long straight move
            move_twist.angular.z = math.radians(12.5 * turn_mod)

        print("12")

        # Linear velocity
        if action[1] == "S":
            # Short straight move
            # move_twist.linear.x = forward_vector[0] * 0.03
            # move_twist.linear.y = forward_vector[1] * 0.03
            move_twist.linear.x = 0.03
            move_twist.linear.y = 0.03
            move_twist.linear.z = 0.03
        elif action[1] == "L":
            # Long straight move
            # move_twist.linear.x = forward_vector[0] * 0.06
            # move_twist.linear.y = forward_vector[1] * 0.06
            move_twist.linear.x = 0.06
            move_twist.linear.y = 0.06
            move_twist.linear.z = 0.06

        # Publish the given twist
        # rospy.loginfo(f"Publishing move_twist: {move_twist}")
        print("BP")
        self.movement_pub.publish(move_twist)
        print("AP")

    def get_next_action(self)->tuple:
        """
        Uses the epsilon-greedy algorithm to choose the next action to take.
        Returns
        -------
        tuple(int, bool) : A tuple containing the index of the next action to take and a bool representing if a greedy action was chosen.
        """
        # If random normalized value is less than epsilon
        if np.random.random() < self.epsilon:
            # Choose the most promising value from the q-table for this state
            # rospy.loginfo(f"CS: {self.current_state[0]}, {self.current_state[1]}, {self.current_state[2]}, {self.current_state[3]}")
            return (np.argmax(self.q_table[self.current_state[0], self.current_state[1], self.current_state[2], self.current_state[3]]), True)
        else: # Otherwise, select a random action
            return (np.random.randint(len(self.actions)), False) # Random value 0 to len(self.actions)-1

    #endregion

    #region Rewards

    def get_reward_value(self, old_state:list, current_state:list, action_index:int, has_collided:bool)->float:
        """
        Function that determines how much to reward the algorithm
        based on the old state, the current state, the action that
        was taken, and whether or not the robot has collided with
        something.

        Returns
        -------
        float : The reward value
        """
        # Return a large negative reward if we've hit a wall (terminal state) (don't punish for getting lost)
        if has_collided:
            return -100.0

        # I need to identify states for which there are correct actions and reward positively when the action taken is the corresponding reward
        # Front, FrontRight, Right, Left
        if current_state[2] == 0 or current_state[2] == 2 or current_state[0] == 0 or current_state[3] == 0: # If close right, far right, close front, or close left
            return -1.0

        # Return no reward by default so as not to encourage behavioral loops
        return 0.0

    #endregion

    #region Save/Load

    def save_q_table(self, filename:str)->None:
        """
        Saves the q-table data to a file of the given filename.
        """
        with open(os.path.join(sys.path[0], filename), "wb") as q_file:
            rospy.loginfo("Saved updated q_table to q-table.npy!")
            np.save(q_file, self.q_table)
    
    def load_q_table(self, filename:str)->None:
        """
        Loads q-table data from a file of the given filename.
        """
        try:
            # Load q-table from file
            with open(os.path.join(sys.path[0], filename), "rb") as q_file:
                rospy.loginfo("Loaded q_table from q-table.npy!")
                self.q_table = np.load(q_file)
        except Exception:
            rospy.logwarn("q-table.npy file not found! Defaulting to 0 initialized q-table.")
            self.q_table = np.zeros((self.front_states, self.frontright_states, self.right_states, self.left_states, len(self.actions)))

    #endregion

    #region Learning Measurement

    def init_learning_measurement(self, num_episodes:int)->None:
        # One inner list for each episode and one value for each action
        self.total_actions = [[0,0,0] for i in range(num_episodes)]
        self.correct_actions = [[0,0,0] for i in range(num_episodes)]
        # Init all values to NaN for plotting reasons
        self.learning_rates_percentage = [[0,0,0] for i in range(num_episodes)]

    def check_correct_action(self, state, action_index:int, num_episodes:int, current_episode_number:int, greedy_action:bool)->None:
        """
        Checks several manually defined "correct actions" to see if the correct action was taken given the state.
        """
        # Actions
        # SS - Straight short
        # SL - Straight long
        # RS - Right short
        # RL - Right long
        # LS - Left short
        # LL - Left long

        # States
        # Front, FrontRight, Right, Left
        # ["C/M/F", "C/M/F", "C/M/F", "C/M/F"]

        adjusted_action_index = 0
        if action_index == 2 or action_index == 3:
            adjusted_action_index = 1
        elif action_index == 4 or action_index == 5:
            adjusted_action_index = 2

        # Define states for which correct actions are known
        straight_states = (state[0] == 1 and state[2] == 1)
        right_states = (state[0] == 2 and state[2] == 1)
        left_states = (state[0] == 0 and state[2] == 1)

        # If this was a greedy action and not a random action
        if greedy_action:
            # Only continue if the state is in the list of states who have correct actions
            if (straight_states or right_states or left_states):
                # If close front and medium right (Like for U-turns and Left turns)
                if left_states:
                    # If Left Short or Left Long, you did well
                    if action_index == 4 or action_index == 5:
                        # Increment correct actions for this action
                        self.correct_actions[current_episode_number][adjusted_action_index] += 1

                # If far front and medium right (Like on I-turns)
                elif right_states:
                    # If Right Short or Right Long, you did well
                    if action_index == 2 or action_index == 3:
                        # Increment correct actions for this action
                        self.correct_actions[current_episode_number][adjusted_action_index] += 1

                # If medium front and medium right (Along right wall)
                elif straight_states:
                    # If Straight Short or Straight Long, you did well
                    if action_index == 0 or action_index == 1:
                        # Increment correct actions for this action
                        self.correct_actions[current_episode_number][adjusted_action_index] += 1

                # Increment total actions for this action
                self.total_actions[current_episode_number][adjusted_action_index] += 1

        # Calculate learning rate percentage for this action
        lrp = self.correct_actions[current_episode_number][adjusted_action_index] / max(1.0, self.total_actions[current_episode_number][adjusted_action_index])
        self.learning_rates_percentage[current_episode_number][adjusted_action_index] = lrp # Make this value and all successive values lrp
        
        # If percentage is close to zero, try to set it to the last non-zero value for better graphing
        if self.learning_rates_percentage[current_episode_number][adjusted_action_index] <= 0.01:
            for i in range(current_episode_number, -1, -1):
                if self.learning_rates_percentage[i][adjusted_action_index] > 0.01:
                    self.learning_rates_percentage[current_episode_number][adjusted_action_index] = self.learning_rates_percentage[i][adjusted_action_index]
        # rospy.loginfo(f"Learning Rate Percentage: {current_episode_number}, {adjusted_action_index}, {self.learning_rates_percentage[current_episode_number][adjusted_action_index]}")


    def init_plotting(self, num_episodes:int):
        x_axis = list(range(num_episodes))
        # y_axis = [[((i+1)/num_episodes),0.0,0.0,0.0,0.0,0.0] for i in range(num_episodes)]
        y_axis = [[((i+1)/num_episodes),0.0,0.0] for i in range(num_episodes)]

        plt.ion()
        fig = plt.figure("Learning Rate Plot")
        ax = fig.add_subplot(111)

        plt.title('Learning Rate (All Actions)')
        plt.xlabel('Episode Number')
        plt.ylabel('Correct Actions Selection Ratio')

        colorset = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        line0,line1,line2 = ax.plot(x_axis, y_axis, 'b-') # Returns a tuple of line objects, thus the comma
        plot_lines = [line0,line1,line2]
        action_labels = ["Action Straight", "Action Right", "Action Left"]

        return fig, plot_lines, action_labels, colorset

    def plot_learning(self, plot_data:list, fig, plot_lines:list, action_labels:list, colorset):
        plot_data_zip = list(zip(*plot_data))

        for i in range(len(plot_lines)):
            plot_lines[i].set_ydata(plot_data_zip[i])
            plot_lines[i].set_color(colorset[i])
            plot_lines[i].set_label(action_labels[i])
        fig.legend([action_labels[0],action_labels[1],action_labels[2]], loc='upper right')
        fig.canvas.draw()
        fig.canvas.flush_events()
        

    #endregion

    #endregion

    #region Gazebo Services

    def init_gazebo(self)->None:
        # Create Gazebo proxy functions
        self.gz_reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.gz_get_model_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.gz_set_model_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.gz_unpause_phys_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.gz_pause_phys_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

    def gz_reset_simulation(self)->bool:
        """ Reset the Gazebo simulation. """
        rospy.loginfo("Waiting for service: /gazebo/reset_simulation...")
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.gz_reset_proxy()
            rospy.loginfo("Gazebo simulation reset complete!")
            return True
        except (rospy.ServiceException) as exc:
            rospy.logerr(f"/gazebo/reset_simulation service call failed.\nError: {str(exc)}")
            return False

    def gz_get_model_state(self)->tuple:
        """
        Get and return the state of the robot model in Gazebo.
        
        Returns
        -------
        tuple : An (x_pos, y_pos, z_rot) tuple with position information from the model state.
        """
        # rospy.loginfo("Waiting for service: /gazebo/get_model_state...")
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            response = self.gz_get_model_proxy("turtlebot3_waffle_pi", "")
            # rospy.loginfo(f"Got Gazebo model state!")
            return (response.pose.position, response.pose.orientation)
        except (rospy.ServiceException) as exc:
            rospy.logerr(f"/gazebo/get_model_state service call failed.\nError: {str(exc)}")
            return None

    def gz_set_model_state(self, x_pos:float = 0.0, y_pos:float = 0.0, z_rot: float = 0.0)->bool:
        """ Set the state of the robot model in Gazebo. """
        rospy.loginfo("Waiting for service: /gazebo/set_model_state...")
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # Convert Euler rotation to quaternion
            rot_quat = quaternion_from_euler(0.0, 0.0, math.radians(z_rot))

            new_model_state = ModelState()
            new_model_state.model_name = "turtlebot3_waffle_pi"
            new_model_state.pose.position.x = x_pos
            new_model_state.pose.position.y = y_pos
            new_model_state.pose.position.z = 0
            new_model_state.pose.orientation.x = rot_quat[0]
            new_model_state.pose.orientation.y = rot_quat[1]
            new_model_state.pose.orientation.z = rot_quat[2]
            new_model_state.pose.orientation.w = rot_quat[3]

            self.gz_set_model_proxy(new_model_state)
            # rospy.Rate(1).sleep() # Sleep for 1 second to ensure proper reset
            # rospy.loginfo("Set Gazebo model state!")
            return True
        except (rospy.ServiceException) as exc:
            rospy.logerr(f"/gazebo/set_model_state service call failed.\nError: {str(exc)}")
            return False

    def gz_unpause_physics(self)->bool:
        """ Unpause the Gazebo physics. """
        # rospy.loginfo("Waiting for service: /gazebo/unpause_physics...")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.gz_unpause_phys_proxy()
            # rospy.loginfo("Gazebo physics unpaused!")
            return True
        except (rospy.ServiceException) as exc:
            rospy.logerr(f"/gazebo/unpause_physics service call failed.\nError: {str(exc)}")
            return False

    def gz_pause_physics(self)->bool:
        """ Pause the Gazebo physics. """
        # rospy.loginfo("Waiting for service: /gazebo/pause_physics...")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.gz_pause_phys_proxy()
            # rospy.loginfo("Gazebo physics paused!")
            return True
        except (rospy.ServiceException) as exc:
            rospy.logerr(f"/gazebo/pause_physics service call failed.\nError: {str(exc)}")
            return False

    #endregion

    def handle_rospy_shutdown(self)->bool:
        # Handle rospy shutdown
        if rospy.is_shutdown():
            rospy.loginfo("rospy has shut down!")
            return True
        return False


if __name__ == '__main__':
    # Initialize our Wallfollow Agent
    wallfollow_agent = WallfollowAgent()