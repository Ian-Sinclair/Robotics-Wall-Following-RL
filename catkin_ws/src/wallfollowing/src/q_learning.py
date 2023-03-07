#!/usr/bin/env python3
import itertools
import json
import math
import sys, getopt
import rospy
import numpy as np
import os
import random
import rosgraph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler



'''
        ------------------------------------------------------------
        ------------------------------------------------------------
        --      Navigation Software for Robotic Wall Following    --
        ------------------------------------------------------------
        ------------------------------------------------------------
'''


'''
    NOTE! This file contains logic to construct and test a navigation system 
        generated through reinforcement learning.

        On a high level, this either exploits or finds a mapping between robot 
        states and actions, that result in some consistent or learned behavior.

        Indented is to learn a type of wall following behavior that prevents the 
            robot from crashing while maintain a constant distance to a wall on 
            the right of the robots frame.

        States are defined by the the tuple:

        state <-- (right status , front status , left status , right diagonal status)

        where the status of each direction is informed by the minimum lidar distance
        over some interval [a,b]deg.
        The observation for each direction is found by the key,

            right <-- [245 , 355]deg
            front <-- [-30 , 30]deg
            left <-- [55 , 125]deg
            right diagonal <-- [265 , 359]deg
        
        After which, the minimum distance reading in each direction is discretized into
        a finite set by the thresholds

            -------------------------------------------------------
            right status <-- 'close' :          (0 , 0.35), 
                             'tilted close' :   (0.35 , 0.9*d_w), 
                             'good' :           (0.9*d_w , 1.1*d_w), 
                             'tilted far' :     (1.1*d_w , 1.5*d_w), 
                             'far' :            (1.5*d_w , 20)
            -------------------------------------------------------
            front status <-- 'front' : {'close' : (0 , 0.5) , 
                            'medium' : (0.5 , 0.75) , 
                            'far' : (0.75 , 20)
            -------------------------------------------------------
            left status  <-- 'left' : {'close' : (0,0.6), 
                             'far' : (0.6,20) 
            -------------------------------------------------------
            orientation_forward Status  <-- 'close' : (0,0.85) , 
                                            'far' : (0.85,20)
            -------------------------------------------------------
        
        A single state is some combination of each of the directional status's.
        And so there are 

            |S| = 5 X 3 X 2 X 2 = 60 states

        Each state points to a single action in the action space.

            A = 'straight'
                'slight left'
                'hard left'
                'slight right'
                'hard right'
        Each of which publishes a corresponding velocity to the robot.
'''
'''
    NOTE! TABLE OF CONTENTS

        --q_learning()
            - class object controls training and testing along with file 
                generation of q table management 
                
            -- init_robot_control_params()
                - Function that initializes control parameters,
                    including the state space discitization and action
                    space parameters.
                    Along with the possible starting positions of the robot.
            
            -- training
                - runs a learning cycle to train a new q table and save it to a file.
            
            -- run_epoc
                - runs a single learning episode, handles q table updating and simulation
                    logic
            -- demo 
                - loads a pre-saved q table from file and simulates the learned behavior.
                    No q table updates are preformed in demo.
'''

'''
    NOTE! MODES     ----->   Terminal Parameters  <------

            There are two primary modes either 
                --train
                or
                --demo
            
            -- For more information run,

                $ rosrun wallfollowing q_learning.py --help

            -- To train a new model; try running,

                $ rosrun wallfollowing q_learning.py --train --out_filename test_table_name --num_epocs=100 --plot_out_file 'test_plots'
            
            You can alter the parameters.

            -- To demo a pre-saved model, the following will load the optimal Q table from Temporal Difference learning

                $ rosrun wallfollowing q_learning.py --demo
            
            Otherwise select which json q table you want to demo by running,

                $ rosrun wallfollowing q_learning.py --demo --in_filename <q table file name>

'''



class q_learning() :
    """Reinforcement learning class controlling training and testing
        navigation software for robotic wall following

    Returns:
        _type_: _description_
    """
    '''
        Modes:   Training, testing
            Training:
                informed by:
                    in_file    --> loads a saved Q table
                    out_file  --> location to save new Q_table
                    table_out_file  -->  location to save plots of learning.
                    num_epocs  --> number of epocs used in training
                    min_epsilon  --> minimum epsilon value
                    max_epsilon  --> maximum epsilon value (decreases linearly)
                Traditional --> Default
                SARSA --> special input

    '''
    def __init__( self , out_filename = None , 
                        in_filename = None , 
                        train = False , 
                        num_epocs = 100,
                        headless = False, 
                        demo = True, 
                        demo_rounds = 25 ,
                        plot_out_file = None, 
                        strategy = 'Temporal Difference',
                        driving = False,
                        robot = False ) :


        #  Handles incorrect input arguments errors.
        if num_epocs < 1 and train == True :
            rospy.logerr(f'ERROR: number of episodes is {num_epocs} must be at least 1')
        
        if train == True and demo == True :
            rospy.logwarn(f' Both training and demo is {True}, training will occur first')

        #  Constructs data structure to save training plot information
        self.plot_out_file = plot_out_file
        self.known_action_states = None
        if plot_out_file != None :
            self.record_info = {
                'Accumulated Rewards' : 
                    {
                        'epocs' : [0] ,
                        'data' : [0]
                    } ,
                'Correct Slight Left Ratio' :
                    {
                        'epocs' : [0],
                        'data' : [0]
                    } ,
                'Correct Slight Right Ratio' : 
                    {
                        'epocs' : [0],
                        'data' : [0]
                    } ,
                'Correct Hard Left Ratio' :
                    {
                        'epocs' : [0],
                        'data' : [0]
                    } ,
                'Correct Hard Right Ratio' : 
                    {
                        'epocs' : [0],
                        'data' : [0]
                    } ,
                'Correct Straight Ratio' :
                    {
                        'epocs' : [0],
                        'data' : [0]
                    }
            }
            self.known_action_states = self.load_q_table_from_JSON('known_states_tracker')


        self.out_filename = out_filename
        self.in_filename = in_filename
        #  Stores relevant data from callbacks
        self.cache = { 'scan data' : None ,
                 'state' : None ,
                 'action' : None ,
                 'velocity' : None ,
                 'incoming scan data' : None,
                 'position' : None }

        #  Initializes state and action space
        self.q_table = self.init_robot_control_params()

        self.init_node()
        self.init_subscriber()
        self.init_publisher()

        if robot == True :
            #  Loads Q table from file if enabled
            if type(in_filename) == type(' ') :
                rospy.loginfo(f'loading q table from file {in_filename}')
                self.q_table = self.load_q_table_from_JSON( in_filename )
            
            self.demo_robot( self.q_table )

        else :
            self.init_services()
            self.unpause_physics()

            rospy.on_shutdown( self.shutdown )

            #  Loads Q table from file if enabled
            if type(in_filename) == type(' ') :
                rospy.loginfo(f'loading q table from file {in_filename}')
                self.q_table = self.load_q_table_from_JSON( in_filename )

            #  Saves blank Q table to file if enabled
            if out_filename != None :
                rospy.loginfo(f'saving q table from file {in_filename}')
                self.save_q_table_to_JSON( self.q_table , out_filename )

            #  Runs training cycle if enabled
            if train == True :
                rospy.loginfo(f'---------------Training Mode: {strategy}')
                self.q_table = self.training( self.q_table , num_epocs = num_epocs, strategy= strategy )
                rospy.loginfo(f' Training Cycle Complete: DEMOing for 25 rounds ')
                self.demo( self.q_table , limit = 25 )

            #  Runs testing cycle if enabled
            if demo == True :
                if in_filename == None :
                    rospy.logwarn(f'WARNING: Cannot load Q_table because arg: in_filename is {in_filename}.')
                    rospy.logwarn(f'WARNING: Loading Q table from::: ''best_Q_table.json'' instead.')
                    rospy.logwarn(f'WARNING: to load in Q table for demo, run, \n \
                                \t $ rosrun wallfollowing q_learning.py --demo --in_filename <json file name>')
                    self.q_table = self.load_q_table_from_JSON('best_Q_table')

                self.demo( self.q_table , limit = demo_rounds )



    def demo_robot( self , q_table = None ) :
        self.ROS_MASTER_URL = None
        self.ROS_HOSTNAME = None

        #  Wait for gazebo's first callback
        self.cache['scan data'] = None
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        #  If no q table is given, assumes there is a none empty q table available
        if q_table == None :
            q_table = self.q_table

        rospy.loginfo('Demo: initialized')

        while not rospy.is_shutdown() :
            #  Discretized scan data to Q table state information
            state = self.scan_to_state(self.cache['scan data'].ranges)
            self.cache['state'] = state


            #  Gets the highest utility action for the robots current state
            action = max( q_table[state], key = q_table[state].get )

            #  Converts Q table action to linear and angular velocities
            x , nz = self.actions[action]

            #  Publishes linear and angular velocities as Twist object 
            self.publish_velocity( x = x , nz = nz )

            if min(self.cache['scan data'].ranges) < 0.14 :
                rospy.loginfo(f'----Stuck:  starting unstucking procedure ------')
                self.publish_velocity( x = 0 , nz = 0 )
                break



    def init_robot_control_params( self ) :
        """Initializes states and action space.
            Along with wall following parameters.

        Returns:
            dict: Q table dictionary
        """        

        
        #  Desired distance from walls
        d_w = 0.5

        self.d_w = d_w

        #  Sensor ranges in degrees for which lidar sensors to use for each direction
        #  on the robot.
        self.scan_key = {
            'right' : [ ( 245 , 375 ) ] ,
            'front' : [ ( 0 , 30 ) , ( 330 , 359 ) ] ,
            'orientation_forward' : [ ( 300 , 330 ) ] ,
            'left' : [ ( 55 , 125 ) ] ,
        }

        #  Possible places on the simulation map
        #  where the robot can start
        self.start_positions = [
            (-1.7,-1.7,0 ),
            (0,2,math.pi),
            #(1.8,0.8,math.pi/2),
            #(1.7,-1.7,math.pi/2),
            #(1.8,1.8,3*math.pi/2),
            #(-2,0.5,0)
        ]

        #  List of linear speeds
        self.linear_actions = { 'fast' : 0.2 }

        #  List of angular speeds
        self.rotational_actions = {'straight' : 0.001 , 
                                    'turn left' : 0.95, 
                                    'hard left' : 1.4, 
                                    'turn right' : -0.95, 
                                    'hard right' : -1.4
                                    }

        #  List of all possible actions from any state
        self.actions = [s for s in itertools.product(*[self.linear_actions.keys() , self.rotational_actions.keys()])]

        temp = {}
        for v,w in self.actions :
            temp[f'linear: ({str(v)}) , angular: ({str(w)})'] = ( self.linear_actions[v] , self.rotational_actions[w] )

        self.actions = temp.copy()

        '''
            A single state is defined by the status of 
            each element in the tuple (right , front , left)
        '''
        self.directional_states = {
            'right' : ['close' , 'tilted close', 'good' , 'tilted far' , 'far'],
            'front' : ['close' , 'medium' , 'far'],
            'orientation_forward' : ['close' , 'far'],
            'left' : ['close' , 'far']
        }
        #  Sets state thresholds to discretized scan distance data.
        self.thresholds = {
            'right' : {'close' : (0 , 0.25), 'tilted close' : (0.25 , 0.9*d_w), 'good' : (0.9*d_w , 1.1*d_w), 'tilted far' : (1.1*d_w , 1.5*d_w), 'far' : (1.5*d_w , 20)},
            'front' : {'close' : (0 , 0.45) , 'medium' : (0.45 , 0.75) , 'far' : (0.75 , 20)},
            'orientation_forward' : {'close' : (0,1), 'far' : (1,20) },
            'left' : {'close' : (0,0.75), 'far' : (0.75,20) }
        }

                #  Creates a blank Q table with all state/action pairs. And saves to file.
        if self.in_filename == None :
            self.q_table = {}

            new_states = []
            for direction , states in self.directional_states.items() :
                mod_states = []
                for s in states :
                    mod_states += [ direction + ' : ' + s ]
                new_states += [ mod_states ]

            new_states = [s for s in itertools.product(*new_states)]

            return self.construct_blank_q_table( {} , new_states , self.actions , default_value=0 )
        return None


    def init_node( self ) :
        rospy.init_node( 'wall_following' , anonymous=True )


    def init_publisher( self ) :
        '''
            Inits publisher to cmd_vel with message Twist.
            (Controls the velocity of the robot)
        '''
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)


    def init_subscriber(self) :
        '''
            Subscribes to /scan topic into self.scan_callback function
            and to /cmd_vel into self.cmd_vel_callback function.
        '''
        rospy.Subscriber('scan' , LaserScan , callback=self.scan_callback)
        rospy.Subscriber("/cmd_vel", Twist, callback=self.cmd_vel_callback)


    def init_services( self ) :
        '''
            Loads ROSPY services for interfacing with Gazebo simulation
        '''

        rospy.loginfo(f'-----Waiting For Services-----')
        rospy.wait_for_service('/gazebo/reset_world')
        try :
            self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            rospy.loginfo(f'-----Rospy Service [reset world] activated')
        except :
            rospy.logerr(f'Unable to connect to rospy service reset world')
            raise
            
        rospy.wait_for_service('/gazebo/set_model_state')
        try :
            self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo(f'-----Rospy Service [set model state] activated')
        except :
            rospy.logerr(f'Unable to connect to rospy service set model state')

        rospy.wait_for_service('/gazebo/pause_physics')
        try :
            self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        except :
            rospy.logerr(f'Unable to connect to rospy service pause physics')
        
        rospy.wait_for_service('/gazebo/unpause_physics')
        try :
            self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        except :
            rospy.logerr(f'Unable to connect to rospy service pause physics')        


    def scan_callback( self , scan ) :
        '''
            Sets scan data cache when new scan information is available.
        '''
        '''
            Possibly encode state, then check if state has been updated?
            http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
        '''
        t = scan
        t.ranges = [float(a) if 0.1<a<3.5 else 3.5 for a in scan.ranges]
        self.cache['scan data'] = t
        self.cache['incoming scan data'] = t


    def cmd_vel_callback( self , data ) :
        '''
            Maintains a record of the current velocity of the robot
        '''
        self.cache['velocity'] = data


    def num_to_threshold( self , direction , value ) :
        '''
            Converts distance value in a direction into a discrete threshold
            Thresholds are initialized in __init__() and are static.
        '''
        for key,threshold in self.thresholds[direction].items() :
            a,b = threshold
            if a <= value < b :
                return key


    def scan_to_state( self , scanArray ) :
        '''
            Discretizes scan information into Q table states.
        '''
        scanArray = list(np.clip(scanArray , a_min=0 , a_max=3.5))
        state = []
        for direction,ranges in self.scan_key.items() :
            state_range = []
            for a,b in ranges :
                state_range += scanArray[a:b]
            value = min(state_range)
            threshold = self.num_to_threshold( direction , value )
            state += [f'{direction} : {threshold}']
        return tuple(state)


    def publish_velocity( self, x = 0 , y = 0 , z = 0 , nx = 0 , ny = 0 , nz = 0 ) :
        '''
            Converts velocity information to Twist object
        '''
        twist = Twist()
        twist.linear.x = x
        twist.linear.y = y
        twist.linear.z = z
        twist.angular.x = nx
        twist.angular.y = ny
        twist.angular.z = nz
        self.velocity_publisher.publish( twist )


    def save_q_table_to_JSON( self , q_table , filename) :
        """Saves Q Table dictionary as JSON file at 'filename' location

        Args:
            q_table (dict): Q table to be saved
            filename (str): unextended filename

        Returns:
            Boolean: Returns False if failed, otherwise True.
        """        
        #  Filenames for important documents are protected so they cant be written over
        if filename in ['best_Q_table', 'Manual_Q_table', 'known_states_tracker', 'Optimal_Q_Table_SARSA', 'Optimal_Q_Table_TD'] :
            rospy.logerr(f'filename: {filename} is protect, save Q table to a different filename')
            return False
        
        #  Converts tuple type state keys to string type
        string_q_table = {}
        for key in q_table.keys() :
            str_key = ''
            for s in key :
                str_key += s + '\t'
            string_q_table[str_key] = q_table[key].copy()
            
        #  Dumps to JSON
        with open(os.path.join(sys.path[0], f'{filename}.json') , "w" ) as f:
            json.dump( string_q_table , f , indent=4 )
        return True


    def load_q_table_from_JSON( self , filename ) :
        """Loads q_table from JSON file at filename.
            EX: load_q_table_from_JSON( 'Manual_Q_table' )


        Args:
            filename (str): unextended filename

        Returns:
            dict: Q table
        """        
        q_table = {}
        with open(os.path.join(sys.path[0], f'{filename}.json'), "r") as json_file:
            data = json.load(json_file)
        for string,actions in data.items() :
            key = tuple( string.split('\t')[:-1] )
            q_table[key] = actions.copy()
        
        return q_table


    def set_model_pos(self,x=0,y=0,z=0,nx=0,ny=0,nz=0,w=1) :
        """Sets the position of the waffle bot in the world

        Args:
            x (int, optional): position in x. Defaults to 0.
            y (int, optional): position in y. Defaults to 0.
            z (int, optional): position in z. Defaults to 0.
            nx (int, optional): Rotation in x quaternion. Defaults to 0.
            ny (int, optional): Rotation in y quaternion. Defaults to 0.
            nz (int, optional): Rotation in z quaternion. Defaults to 0.
            w (int, optional): Rotation in w quaternion. Defaults to 1.
        """        
        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_waffle'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
        state_msg.pose.orientation.x = nx
        state_msg.pose.orientation.y = ny
        state_msg.pose.orientation.z = nz
        state_msg.pose.orientation.w = w

        try :
            self.set_state( state_msg )
        except :
            raise


    def get_distance( self , scanArray, direction ) :
        """ Gets the minimum scan measurement in the scan array
            over the interval informed by 'direction'

        Args:
            scanArray (list): Scan array object from sensor array
            direction (str): direction cache key string

        Returns:
            float: minimum distance in interval from direction
        """        
        scanArray = list(np.clip(scanArray , a_min=0.1 , a_max=3.5))
        state_range = []
        ranges = self.scan_key[direction]
        for a,b in ranges :
            state_range += scanArray[a:b]
        return min(state_range)


    def get_reward( self , state , scanArray, action= 0 ) :
        """Gets the reward for a particular action and new state.
            Uses both an extrinsic reward based on the robots surroundings
            and intrinsic reward based on the risk associated with the selected
            action

        Args:
            state (tuple): current state of the robot
            scanArray (list): Complete 360 scan information
            action (str, optional): action selected to reach the current state. Defaults to 0.

        Returns:
            float: reward
        """        
        #  determines the risk of the selected action.
        if action != 0 :
            action = 0.5*abs(self.actions[ action ][1])
            if action < 1.2 : action = 0
        #  checks if the robot collides with a wall
        if self.is_blocked(scanArray) : 
            return -5

        #  gets distance measurements for each direction
        #x = self.get_distance(scanArray , 'right')
        #y = self.get_distance(scanArray , 'front')
        #z = self.get_distance(scanArray , 'left')
        y = min( scanArray[0:30] + scanArray[330:359] )
        x = min( scanArray[180:359] )

        #  the robot is a good distance from the wall on the right
        if abs(x-self.d_w) < 0.1 and y>0.5 : return 5 - action
        
        #  graduated punishment encouraging the robot to return to a good position
        fx = -5*math.sin((math.pi*(x-0.5))/6) + y - action
        #if z< 0.75 :
        #    fx += - (3.5 - z)
        return fx


    def update_Q_table( self, q_table , state , reward , action , new_state , gamma = 0.8, alpha = 0.2, strategy = 'SARSA', new_action = None ) :
        """Updates the q table based on Temporal Difference or SARSA update strategy.

        Args:
            q_table (dict): Q table
            state (tuple): previous state
            reward (float): reward for new state
            action (str): action selected at previous state
            new_state (tuple): state resulting from preforming the action
            gamma (int, optional): discount rate. Defaults to 0.8.
            alpha (float, optional): learning rate. Defaults to 0.2.
            strategy (str, optional): literal. ['Temporal Difference' , 'SARSA']
            new_action (str, optional): action leaving the new state (SARSA). Defaults to None.

        Returns:
            _type_: _description_
        """        
        if strategy == 'Temporal Difference' :
            sample = reward + gamma*max(q_table[new_state].values())
        elif strategy == 'SARSA' :
            if new_action != None :
                #print('SARSA')
                sample = reward + gamma*q_table[new_state][new_action]
            else :
                rospy.logwarn(f'CANNOT Update SARSA Q table: new action is {new_action}')
        else : 
            rospy.logerr(f'ERROR updating Q table: unknown strategy, {strategy}, ---ABORTING---')
            sys.exit(2)

        q_table[state][action] = ((1-alpha) * q_table[state][action]) + (alpha * sample)
        return q_table


    def training( self , q_table, num_epocs = 100, strategy = 'Temporal Difference' ) :
        """Runs a full training cycle. updating a saving q table along the way.
            if enabled, plots are updated in real time as the agent learns.

        Args:
            q_table (dict): Q table
            num_epocs (int, optional): number of episodes to run. Defaults to 100.
            strategy (str, optional): Literal ['Temporal Difference' , 'SARSA']. Defaults to 'Temporal Difference'.

        Returns:
            dict: Q table
        """        
        rospy.loginfo(f'-------Initializing Training Cycle-------')

        #  Wait for gazebo's first callback
        self.unpause_physics
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        self.cache['incoming scan data'] = None
        
        epoc = 0
        epsilon = 0.8
        temp = epsilon
        update_plot_flag = False

        history = {
                    'slight left' : {
                        'correct' : 0,
                        'total' : 0
                    },
                    'slight right' : {
                        'correct' : 0,
                        'total' : 0
                    },
                    'hard left' : {
                        'correct' : 0,
                        'total' : 0
                    },
                    'hard right' : {
                        'correct' : 0,
                        'total' : 0
                    },
                    'straight' : {
                        'correct' : 0,
                        'total' : 0
                    },
                }

        while epoc < num_epocs and not rospy.is_shutdown() :
            rospy.sleep(0.001)
            rospy.loginfo(f'\t\t\t\t Running epoc {epoc+1}/{num_epocs}')
            epoc += 1

            #  epsilon is linearly reduces as the number of episodes increases
            #  from 0.9 as episode 0 to 0.1 at the final episode.
            epsilon = temp*(1-epoc/num_epocs)+0.1
            rospy.loginfo(f'------ Epsilon:  {round(epsilon,3)}  -------------')

            #  --------------------------------------------------------------------------------------
            #  --------------------------------------------------------------------------------------
                                    #  Runs a single learning cycle
            q_table, accum_reward, run_info = self.run_epoc( q_table , epsilon , strategy = strategy )
            #  --------------------------------------------------------------------------------------
            #  --------------------------------------------------------------------------------------

            #  Saves q table to file if enabled.
            if self.out_filename != None :
                self.save_q_table_to_JSON( q_table , self.out_filename )

            #  --------------------------------------------------------------------------------------
                                    #  Logic to update plots for each action if enabled
            #  --------------------------------------------------------------------------------------
            update_plot_flag = False
            if self.plot_out_file != None :
                if accum_reward != 0 :
                    self.record_info['Accumulated Rewards']['epocs'] += [ epoc-1 ]
                    self.record_info['Accumulated Rewards']['data'] += [ accum_reward ]
                if run_info['slight left']['total'] > 0 :
                    if run_info['slight left']['correct'] > 0 :
                        if history['slight left']['total'] + run_info['slight left']['total'] > 10 :
                            run_info['slight left']['total'] = history['slight left']['total'] + run_info['slight left']['total']
                            run_info['slight left']['correct'] = history['slight left']['correct'] + run_info['slight left']['correct']
                            history['slight left']['total'] = 0
                            history['slight left']['correct'] = 0
                            self.record_info['Correct Slight Left Ratio']['data'] += [ run_info['slight left']['correct']/run_info['slight left']['total'] ]
                            self.record_info['Correct Slight Left Ratio']['epocs'] += [epoc-1]
                            update_plot_flag = True
                        else :
                            history['slight left']['total'] += run_info['slight left']['total']
                            history['slight left']['correct'] += run_info['slight left']['correct']

                if run_info['slight right']['total'] > 0 :
                    if run_info['slight right']['correct'] > 0 :
                        if history['slight right']['total'] + run_info['slight right']['total'] > 10 :
                            run_info['slight right']['total'] = history['slight right']['total'] + run_info['slight right']['total']
                            run_info['slight right']['correct'] = history['slight right']['correct'] + run_info['slight right']['correct']
                            history['slight right']['total'] = 0
                            history['slight right']['correct'] = 0
                            self.record_info['Correct Slight Right Ratio']['data'] += [ run_info['slight right']['correct']/run_info['slight right']['total'] ] 
                            self.record_info['Correct Slight Right Ratio']['epocs'] += [epoc-1]
                            update_plot_flag = True
                        else : 
                            history['slight right']['total'] += run_info['slight right']['total']
                            history['slight right']['correct'] += run_info['slight right']['correct']                            
                if run_info['hard left']['total'] > 0 :
                    if run_info['hard left']['correct'] > 0 :
                        if history['hard left']['total'] + run_info['hard left']['total'] > 10 :
                            run_info['hard left']['total'] = history['hard left']['total'] + run_info['hard left']['total']
                            run_info['hard left']['correct'] = history['hard left']['correct'] + run_info['hard left']['correct']
                            history['hard left']['total'] = 0
                            history['hard left']['correct'] = 0
                            self.record_info['Correct Hard Left Ratio']['data'] += [ run_info['hard left']['correct']/run_info['hard left']['total'] ]
                            self.record_info['Correct Hard Left Ratio']['epocs'] += [epoc-1]
                            update_plot_flag = True
                        else : 
                            history['hard left']['total'] += run_info['hard left']['total']
                            history['hard left']['correct'] += run_info['hard left']['correct'] 
                if run_info['hard right']['total'] > 0 :
                    if run_info['hard right']['correct'] > 0 :
                        if history['hard right']['total'] + run_info['hard right']['total'] > 10 :
                            run_info['hard right']['total'] = history['hard right']['total'] + run_info['hard right']['total']
                            run_info['hard right']['correct'] = history['hard right']['correct'] + run_info['hard right']['correct']
                            history['hard right']['total'] = 0
                            history['hard right']['correct'] = 0
                            self.record_info['Correct Hard Right Ratio']['data'] += [ run_info['hard right']['correct']/run_info['hard right']['total'] ] 
                            self.record_info['Correct Hard Right Ratio']['epocs'] += [epoc-1]
                            update_plot_flag = True
                        else : 
                            history['hard right']['total'] += run_info['hard right']['total']
                            history['hard right']['correct'] += run_info['hard right']['correct']
                if run_info['straight']['total'] > 0 :
                    if run_info['straight']['correct'] > 0 :
                        if history['straight']['total'] + run_info['straight']['total'] > 10 :
                            run_info['straight']['total'] = history['straight']['total'] + run_info['straight']['total']
                            run_info['straight']['correct'] = history['straight']['correct'] + run_info['straight']['correct']
                            history['straight']['total'] = 0
                            history['straight']['correct'] = 0
                            self.record_info['Correct Straight Ratio']['data'] += [ run_info['straight']['correct']/run_info['straight']['total'] ] 
                            self.record_info['Correct Straight Ratio']['epocs'] += [epoc-1]
                            update_plot_flag = True
                        else : 
                            history['straight']['total'] += run_info['straight']['total']
                            history['straight']['correct'] += run_info['straight']['correct'] 
                if update_plot_flag == True :
                    self.plot( self.record_info , self.plot_out_file )
        rospy.loginfo(f'-------Training Complete-------')
        self.reset_world()
        return q_table


    def run_epoc( self , q_table = None , epsilon = 0 , limit = 150, strategy = None ) :
        '''
            runs a single epoc in training
        '''

        total_reward = 0
        run_info = {
            'slight left' : {
                'correct' : 0,
                'total' : 0
            },
            'slight right' : {
                'correct' : 0,
                'total' : 0
            },
            'hard left' : {
                'correct' : 0,
                'total' : 0
            },
            'hard right' : {
                'correct' : 0,
                'total' : 0
            },
            'straight' : {
                'correct' : 0,
                'total' : 0
            },
        }

        random_action_flag = False

        #  Wait for gazebo's first callback
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        #  Loads blank q table object from init_parameters
        if q_table == None :
            q_table = self.q_table

        
        count = 0

        #  Resets the position of the robot
        rospy.loginfo(f'----Resetting World-----')
        self.reset_world()

        #  Repositions the robot to one of the valid starting positions
        x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        if x == 'random' :
            x = random.random()*4 - 2
            y = random.random()*4 - 2
            theta = random.randint(0,359)
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
        self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)


        #  Waits for new lidar data after resetting the robot.
        self.cache['incoming scan data'] = None
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        rospy.loginfo(f'----Scan Data Found----')

        #  Gets the first state in the starting position
        state = self.scan_to_state(self.cache['scan data'].ranges)
        self.cache['state'] = state

        #  Gets the highest utility action for the robots current state
        action = max( q_table[state], key = q_table[state].get )
        #  Move randomly based on epsilon greedy strategy
        action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
        if action == 'random' : 
            random_action_flag = True
            action = np.random.choice(list(self.actions.keys()) , p=self.softmax(q_table , state))

        #  Converts Q table action to linear and angular velocities
        x , nz = self.actions[ action ]

        #  Saves action
        self.cache[ 'action' ] = action

        #  Saves a history of everything the robot does
        crash_queue = [(state,action,self.cache['position'])]

        #  Publishes linear and angular velocities as Twist object 
        self.publish_velocity( x = x , nz = nz )

        #  Sets terminating conditions for robot. 
        repeat_limit = 1000
        repeat_states = 0

        while not rospy.is_shutdown() and count < limit and repeat_states < repeat_limit :
            #  Main training loop
            #  Will terminate when robot hits a wall, 
            # gets lost (experiences enough states)
            # or spins (repeats to many states)

            # -----------------------------
            # -------- STEP SIZE ----------
            rospy.sleep(0.1)
            self.init_robot_control_params()
            #  Equal to the update rate of the LiDAR scanners
            
            #  Records choice information to plots.
            direction , mod = self.is_known_state(self.cache[ 'state' ]  , self.cache[ 'action' ])
            if random_action_flag == False :
                if direction != None :
                    run_info[direction]['total'] += 1
                    run_info[direction]['correct'] += mod

            #  Discretized scan data to Q table state information
            new_state = self.scan_to_state(self.cache['scan data'].ranges)
            repeat_states += 1

            #  Checks terminating condition, for spinning
            if repeat_states > 0.9*repeat_limit  :
                rospy.loginfo(f'TIMEOUT')
                q_table = self.update_Q_table( q_table , self.cache['state'] , -5 , self.cache[ 'action' ] , new_state, new_action=action, strategy=strategy)
                break

            #  Checks terminating condition, for getting lost
            if new_state != self.cache['state'] :
                repeat_states = 0
                count += 1  
            state = self.cache['state']

            #  Gets the highest utility action for the robots current state
            random_action_flag = False
            action = max( q_table[new_state], key = q_table[new_state].get )

            #  Selects action based on epsilon greedy strategy
            action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
            if action == 'random' : 
                random_action_flag = True
                if 'NaN' in self.softmax( q_table , state ) :
                    print(self.softmax( q_table , state ))
                action = np.random.choice(list(self.actions.keys()), p=self.softmax( q_table , state ) )
            
            #  Converts Q table action to linear and angular velocities
            x , nz = self.actions[ action ]

            #  Gets reward for entering a new state.
            reward = self.get_reward( new_state , self.cache['scan data'].ranges, self.cache[ 'action' ] )

            #  update q table based on strategy variable.
            q_table = self.update_Q_table( q_table , state , reward , self.cache[ 'action' ] , new_state, new_action=action, strategy=strategy)

            #  Record everything that the robot does...
            crash_queue += [ ( new_state , action , self.cache['position']) ]

            #  iterate total reward for plots
            total_reward += reward
            
            #  Update cache with new state and action
            self.cache[ 'state' ] = new_state
            self.cache[ 'action' ] = action

            #  Publishes linear and angular velocities as Twist object 
            self.publish_velocity( x = x , nz = nz )
            

            #  Checks terminating condition for colliding with walls
            #  Resets simulation if robot is blocked
            if self.is_blocked(self.cache['scan data'].ranges) :
                rospy.loginfo(f'----Resetting World-----')
                self.reset_world()
                self.publish_velocity( x = 0 , nz = 0 )
                break
            
        #  Returns q table, and plot information
        return q_table , total_reward , run_info


    def softmax( self , q_table , state , T = 10 ) :
        """Random distribution generator. Given a state, ranks the
            probability of randomly selecting each action by the expected
            utility of that action.
            Actions with a higher utility have a higher chance of being selected

        Args:
            q_table (dict): Q table
            state (tuple): current state of the robot
            T (int, optional): Temperature parameter to prevent intractable calculation. Defaults to 10.

        Returns:
            list: probability space
        """        
        z = sum( [math.exp(a/T) for a in q_table[state].values() ] )
        return [ math.exp(a/T)/z for a in q_table[state].values() ]


    def is_known_state( self , state , action ) :
        '''
            checks if state is in known states
            determine if the correct action has been preformed.
        '''
        mod = 0
        direction = None
        if state in self.known_action_states.keys() :
            if action in self.known_action_states[state] :
                mod = 1
            if "linear: (fast) , angular: (turn left)" in ''.join(str(self.known_action_states[state])) : 
                direction = 'slight left'
            if "linear: (fast) , angular: (turn right)" in ''.join(str(self.known_action_states[state])) :
                direction = 'slight right'
            if "linear: (fast) , angular: (hard left)" in ''.join(str(self.known_action_states[state])) : 
                direction = 'hard left'
            if "linear: (fast) , angular: (hard right)" in ''.join(str(self.known_action_states[state])) :
                direction = 'hard right'
            if "linear: (fast) , angular: (straight)" in ''.join(str(self.known_action_states[state])) :
                direction = 'straight'
        return direction , mod

    
    def demo( self , q_table = None , limit = 25) :
        """Demo a q_table strategy in gazebo
            After the robot hits a wall, the simulation will
            reset, and the robot will be randomly placed on the map.
            by default there are100 maximum simulation resets, 
            until the demo completes.

        Args:
            q_table (dict, optional): Q table to demo if None then demos class q_table. Defaults to None.
            limit (int , optional): maximum number of times the simulation can reset.
        """        

        #  Wait for gazebo's first callback
        self.cache['scan data'] = None
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        #  If no q table is given, assumes there is a none empty q table available
        if q_table == None :
            q_table = self.q_table

        rospy.loginfo('Demo: initialized')
        #  Resets world and positions robot
        self.reset_world()
        x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        if x == 'random' :
            x = random.random()*4 - 2
            y = random.random()*4 - 2
            theta = random.randint(0,359)
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))    
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
        self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)

        #  Waits for LiDAR scan
        self.cache['scan data'] = None
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        count = 0

        #  step limit.
        timeout = 0
        timeout_limit = 10000
        
        while not rospy.is_shutdown() and count < limit :
            rospy.sleep(0.001)
            timeout += 0.1

            #  Discretized scan data to Q table state information
            state = self.scan_to_state(self.cache['scan data'].ranges)
            self.cache['state'] = state


            #  Gets the highest utility action for the robots current state
            action = max( q_table[state], key = q_table[state].get )

            #  Converts Q table action to linear and angular velocities
            x , nz = self.actions[action]

            #  Publishes linear and angular velocities as Twist object 
            self.publish_velocity( x = x , nz = nz )

            #  Resets simulation if robot is blocked
            if self.is_blocked(self.cache['scan data'].ranges) or timeout > timeout_limit :
                self.publish_velocity( x = 0 , nz = 0 )
                timeout = 0
                rospy.loginfo(f'----Resetting World-----')
                self.reset_world()
                x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
                if x == 'random' :
                    x = random.random()*4 - 2
                    y = random.random()*4 - 2
                    theta = random.randint(0,359)
                nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
                self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)
                #  Waits for new lidar data after resetting the robot.
                self.cache['incoming scan data'] = None
                while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
                    rospy.loginfo(f'-----Waiting for scan data-------')
                    rospy.sleep(1)
                count += 1
                rospy.loginfo(f'Wallfollowing Attempt {count}/{limit}')
                continue
        rospy.loginfo(f'-------DEMO Complete--------')
        self.shutdown()


    def shutdown( self ) :
        '''
            Safe rospy shutdown
            Also resets the robots position
        '''
        self.reset_world()
        rospy.loginfo(f'Ending Simulation')
        

    def is_blocked( self, scanArray) :
        '''
            NOTE! either check every lidar scanner or just the front. (right now checking everything.)
        '''
        for range in scanArray :
            if range < 0.14 :
                return True
        return False


    def construct_blank_q_table( self , q_table : dict, states : list , actions : list, default_value = 0 ) :
        '''
            Makes a blank Q table
        '''
        action_dict = {}
        for a in actions :
            action_dict[a] = default_value
        for state in states :
            q_table[state] = action_dict.copy()
        return q_table


    def plot( self,plot_info : dict, outfile = 'default plots') :
        '''
            Plot_info = {
                plot_title = {
                    epocs: list x axis
                    data: list y axis (must be same size of epocs)
                    }
            }
        '''
        try :
            for line_name in plot_info.keys() :
                if line_name != 'Accumulated Rewards':
                    plt.plot(plot_info[line_name]['epocs'], plot_info[line_name]['data'], linewidth=1.5, label = line_name)
            plt.legend()
            plt.title('Correct Action Ratio Temporal Difference')
            plt.xlabel('Episode')
            plt.ylabel('Correct Actions / Total Greedy Actions')

            plt.savefig(os.path.join(sys.path[0], f'{outfile}.pdf'))
            plt.close()
            return True
        except :
            raise
            return False


'''
    ----------------------------------------------------------------------------
    ---------------------------------  MAIN ------------------------------------
    ----------------------------------------------------------------------------
'''

def help() :
    rospy.logwarn(f'\
        -h <Headless mode ENABLED (DISABLED by default)> \n \
        --train\t <flag to enable training> \n\
        --demo\t <flag to demo a cycle> \n\
        --strategy\t <literal either [Temporal Difference , SARSA]> \
        --in_filename\t  <File to load q table from (just the name of the file not the path or extension)> \n\
        --out_filename\t <File to save q table too (just the name of the file not the path or extension)>\n \
        --plot_out_file\t <Filename to save training plots> \n\
        --num_epocs\t <Number of epocs to run in training cycle> \n \
        --demo_rounds\t <number of trails in demo>')
    rospy.logwarn(f'\n EXAMPLE: \n \
                \t $ rosrun wallfollowing q_learning.py --train --out_filename test_table_name --num_epocs=10\n \
                \t $ rosrun wallfollowing q_learning.py --demo --in_filename best_Q_table --demo_rounds=10\n')
    sys.exit(2)



def main( argv ) :
    out_filename = None 
    in_filename = None
    plot_out_file = None
    train = False
    num_epocs = 100
    headless = False 
    demo = False
    demo_rounds = 25
    strategy = 'Temporal Difference'
    driving = False
    robot = False


    ros_running = False
    for _ in range(5) :
        if not rosgraph.is_master_online(): # Checks the master uri and results boolean (True or False)
            rospy.logerr(f'OS INCOMPATIBILITY ERROR: \t -ROS MASTER is OFFLINE-')
        else : 
            ros_running = True
            break
    if ros_running == False :
        rospy.logwarn(f'\nDont forget to run the launch file.')
        rospy.logwarn(f'\n\t$ roslaunch wallfollowing wallfollow.launch\n')
        sys.exit(2)


    arguments = ["--in_filename","--out_filename", "--plot_out_file", "--num_epocs", "--demo_rounds", "--train", "--demo","--SARSA", "--help", '--robot', '-h']

    if len(argv) == 0 :
        rospy.logwarn(f'ERROR: No arguments passed')
        rospy.logwarn(f'arguments are {arguments}')
        rospy.logwarn(f'EXAMPLE: \n \
                      \t $ rosrun wallfollowing q_learning.py --train --out_filename Test_Q_table --num_epocs=100\n\n \
                      \t $ rosrun wallfollowing q_learning.py --demo --in_filename Test_Q_table --demo_rounds=50')
        rospy.logwarn(f'\n for more information run, \n\n \
                      \t $ rosrun wallfollowing q_learning.py --help')
        return False

    protected_files = ['best_Q_table', 'Manual_Q_table', 'known_states_tracker', 'Optimal_Q_Table_SARSA', 'Optimal_Q_Table_TD']

    try :
      opts, args = getopt.getopt(argv, "h",
                                ["in_filename=","out_filename=", "plot_out_file=", "num_epocs=", "demo_rounds=", "robot", "train", "demo", "strategy=", "drive", "help"])
    except getopt.GetoptError:
        rospy.logerr(f'ERROR: when accepting command line arguments')
        rospy.logerr(f'------- Running Help Command -----------')
        help()
    for opt , arg in opts :
        if opt == '-h' :
            rospy.loginfo('Headless mode ENABLED')
            headless = True
        if opt == '--in_filename' :
            in_filename = arg
        if opt == '--out_filename' :
            if arg in protected_files :
                rospy.logerr(f'out_filename: {arg} is a PROTECTED file, cannot write be edited')
                rospy.logerr(f'-----Aborting------')
                sys.exit(2)
            out_filename = arg
        if opt == '--plot_out_file' :
            if arg in protected_files :
                rospy.logerr(f'out_filename: {arg} is a PROTECTED file, cannot write be edited')
                rospy.logerr(f'-----Aborting------')
                sys.exit(2)
            plot_out_file = arg
        if opt == '--num_epocs' :
            if not arg.isnumeric() :
                rospy.logerr(f' arg: num_epocs is {arg}, must be type integer numeric string ')
                rospy.logerr(f' EXAMPLE : $ rosrun wallfollowing q_learning.py  --num_epocs 100 ')
                sys.exit(2)
            num_epocs = int(arg)
        if opt == '--demo_rounds' :
            if not arg.isnumeric() :
                rospy.logerr(f' arg: demo_rounds is {arg}, must be type integer numeric string ')
                rospy.logerr(f' EXAMPLE : $ rosrun wallfollowing q_learning.py  --demo_rounds 25 ')
                sys.exit(2)
            demo_rounds = int(arg)
        if opt == '--strategy' :
            strategy = arg
            if strategy not in ['Temporal Difference', 'SARSA'] :
                rospy.logwarn(f'Strategy ({strategy}) not recognized. Must be either Temporal Difference or SARSA')
                rospy.logwarn(f'Defaulting: strategy = Temporal Difference')
                strategy = 'Temporal Difference'
        if opt == '--train' :
            train = True
        if opt == '--demo' :
            demo = True
        if opt == '--drive' :
            driving = True
        if opt == '--help' :
            help()
        if opt == '--robot' :
            robot = True
    q_learning(out_filename,in_filename,train,num_epocs,headless,demo,demo_rounds,plot_out_file, strategy, driving, robot)



if __name__ == '__main__':
    main(sys.argv[1:])
    