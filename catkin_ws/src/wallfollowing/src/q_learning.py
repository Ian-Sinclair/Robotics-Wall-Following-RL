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

import matplotlib.pyplot as plt
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sshkeyboard import listen_keyboard



'''
    TODO! Place description of states here before turn in
'''



class q_learning() :

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


    '''
    TODO! Add a controller model for rotational speed.
    NOTE! Possible make this a fuzzy logic controller or a new RL model to affect turning speed.
    '''


    '''
    NOTE! fix reward structure, 
        Either parabolic
        or just reward for being in the correct spot.
    '''

    def __init__( self , out_filename = None , 
                        in_filename = None , 
                        train = False , 
                        num_epocs = 100,
                        headless = False, 
                        demo = True, 
                        demo_rounds = 25 ,
                        plot_out_file = None, 
                        SARSA = False,
                         driving = False ) :


        if num_epocs < 1 and train == True :
            rospy.logerr(f'ERROR: number of episodes is {num_epocs} must be at least 1')
        
        if train == True and demo == True :
            rospy.logwarn(f' Both training and demo is {True}, training will occur first')

        self.plot_out_file = plot_out_file
        if plot_out_file != None :
            self.record_info = {
                'accumulated rewards' : [] ,
                'total epocs' : None
            }
        self.init_robot_control_params()

        self.init_node()
        self.init_subscriber()
        self.init_publisher()
        self.init_services()

        rospy.on_shutdown( self.shutdown )

        self.out_filename = out_filename
        self.in_filename = in_filename

        #  Creates a blank Q table with all state/action pairs. And saves to file.
        if in_filename == None :
            self.q_table = {}

            new_states = []
            for direction , states in self.directional_states.items() :
                mod_states = []
                for s in states :
                    mod_states += [ direction + ' : ' + s ]
                new_states += [ mod_states ]

            #new_states += [[str(c) for c in range(10)]]
            new_states = [s for s in itertools.product(*new_states)]

            self.q_table = self.construct_blank_q_table( {} , new_states , self.actions , default_value=0 )

        #  Loads Q table from file
        if type(in_filename) == type(' ') :
            rospy.loginfo(f'loading q table from file {in_filename}')
            self.q_table = self.load_q_table_from_JSON( in_filename )

        if out_filename != None :
            rospy.loginfo(f'saving q table from file {in_filename}')
            self.save_q_table_to_JSON( self.q_table , out_filename )

        if driving == True :
            self.driving()

        if train == True :
            self.q_table = self.training( self.q_table , num_epocs = num_epocs )
            rospy.loginfo(f' Training Cycle Complete: DEMOing for 25 rounds ')
            self.demo( self.q_table , limit = 25 )

        if demo == True :
            if in_filename == None :
                rospy.logwarn(f'WARNING: Cannot load Q_table because arg: in_filename is {in_filename}.')
                rospy.logwarn(f'WARNING: Loading Q table from::: ''best_Q_table.json'' instead.')
                rospy.logwarn(f'WARNING: to load in Q table for demo, run, \n \
                              \t $ rosrun wallfollowing q_learning.py --demo --in_filename <json file name>')
                self.q_table = self.load_q_table_from_JSON('best_Q_table')

            self.demo( self.q_table , limit = demo_rounds )


    def init_robot_control_params( self ) :
        self.cache = { 'scan data' : None ,
                 'state' : None ,
                 'action' : None ,
                 'velocity' : None ,
                 'incoming scan data' : None }
        
        #  Sensor ranges in degrees for which lidar sensors to use for each direction
        #  on the robot.
        self.scan_key = {
            'right' :  [ ( 210 , 330 ) ] ,
            'front' : [ ( 0 , 30 ) , ( 330 , 359 ) ] ,
            'left' : [ ( 45 , 135 ) ] ,
            'right_diagonal' : [ ( 300 , 359 ) ]
        }

        self.start_positions = [
           # (-1.7,-1.7,0 ),
           # (0,2,math.pi),
            (2,1.3,math.pi/2),
           # (1.7,-1.7,math.pi/2),
           #  (0,0,0)
        ]

        #  List of linear speeds
        self.linear_actions = { 'fast' : 0.2 }

        self.slight_turn_vel = math.pi/4
        self.hard_turn_vel = math.pi/2

        
        self.rotational_actions = {'straight' : 0 , 
                                    'slight left' : -self.slight_turn_vel , 
                                    'slight right' : self.slight_turn_vel , 
                                    'hard right' : self.hard_turn_vel , 
                                    'hard left' : -self.hard_turn_vel 
                                    }

        #  List of all possible actions from any state
        self.actions = [s for s in itertools.product(*[self.linear_actions.keys() , self.rotational_actions.keys()])]

        temp = {}
        for v,w in self.actions :
            temp[f'linear: ({str(v)}) , angular: ({str(w)})'] = ( self.linear_actions[v] , self.rotational_actions[w] )
        
        self.actions = temp.copy()
        #self.actions = { 'left' : (0.2 , -math.pi/4) , 'right' : (0.2 , math.pi/4) }

        '''
            A single state is defined by the status of 
            each element in the tuple (right , front , left)
        '''
        self.directional_states = {
            'right' : ['close' , 'tilted close' , 'good' , 'tilted far' , 'far'],
            'front' : ['close' , 'far'],
            'left' : ['close' , 'far'],
            'right_diagonal' : ['close' , 'far']
        }

        #  Desired distance from walls
        d_w = 0.75

        self.d_w = d_w

        fast_tr = 4*( self.linear_actions['fast'] / self.hard_turn_vel )  #  Fast turning radius

        #  Sets state thresholds to discretized scan distance data.
        self.thresholds = {
            'right' : {'close' : (0 , 0.8*d_w) , 'tilted close' : (0.8*d_w , 0.95*d_w) , 'good' : (0.95*d_w , 1.05*d_w), 'tilted far' : (1.05*d_w , 1.2*d_w) , 'far' : (1.2*d_w , 20)},
            'front' : {'close' : (0 , d_w) , 'far' : (d_w , 20)},
            'left' : {'close' : (0 , 1.2*fast_tr) , 'far' : ( 1.2*fast_tr , 20)},
            'right_diagonal' : { 'close' : (0 , 2*d_w) , 'far' : (2*d_w , 20) }
        }


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


    def scan_callback( self , scan ) :
        '''
            Sets scan data cache when new scan information is available.
        '''
        '''
            Possibly encode state, then check if state has been updated?
            http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
        '''
        self.cache['scan data'] = scan
        self.cache['incoming scan data'] = scan


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
        if filename in ['Manual_Q_table'] :
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
            q_table[key] = actions
        
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
        scanArray = list(np.clip(scanArray , a_min=0 , a_max=3.5))
        state_range = []
        ranges = self.scan_key[direction]
        for a,b in ranges :
            state_range += scanArray[a:b]
        return min(state_range)


    def get_reward( self , state , scanArray , previous_scanArray ) :
        if self.is_blocked(scanArray) : return -10

        '''
        if 'good' in state[0] :
            return 0

        if 'tilted' in state[0] :
            return -1
        return -2
        '''
        xp = self.get_distance(previous_scanArray , 'right')
        #yp = self.get_distance(previous_scanArray , 'front')

        x = self.get_distance(scanArray , 'right')
        y = self.get_distance(scanArray , 'front')

        '''
        f_x = 0

        if y < 0.5 : return -5

        if ((x-self.d_w)**2)**0.5 < 0.1 : return 1

        if abs(xp-self.d_w) < abs(x-self.d_w)-0.05 :
            return 0

        return -1
        '''
        ''''''
        #if y < 0.4 : return -5

        #if ((x-self.d_w)**2)**0.5 > 0.5 : return -2

        #if ((x-self.d_w)**2)**0.5 < 0.1 : return 1
        if x < 0.2 : x = 0.2
        if y < 0.2 : y = 0.2
        #f_x = -5*((x-self.d_w)**2)
        f_x = 5*math.cos(1.2*(x-self.d_w))-math.exp(-5*(x-0.7))-1
        #f_x = -5*abs(x-self.d_w)
        f_y = -(1/(y**1.3))+1.2

        #print(f'x={x}\t y={y} \t f_x={f_x} \t f_y={f_y} \tz={f_x+f_y}')
        return f_x + f_y


    def update_Q_table( self, q_table , state , reward , action , new_state , gamma = 0.8, alpha = 0.2 ) :
        sample = reward + gamma*max(q_table[new_state].values())
        q_table[state][action] = (1-alpha) * q_table[state][action] + alpha * sample
        return q_table


    def training( self , q_table, num_epocs = 100, strategy = 'Temporal Difference' ) :
        rospy.loginfo(f'-------Initializing Training Cycle-------')

        #  Wait for gazebo's first callback
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        self.cache['incoming scan data'] = None
        
        epoc = 0
        epsilon = 0.2
        temp = epsilon

        if self.plot_out_file != None :
            self.record_info['total epocs'] = num_epocs

        while epoc < num_epocs and not rospy.is_shutdown() :
            rospy.loginfo(f'\t\t\t\t Running epoc {epoc+1}/{num_epocs}')
            epoc += 1
            epsilon = temp*(1-epoc/num_epocs)
            rospy.loginfo(f'-------- Epsilon:  {epsilon}  ----------')
                
            q_table, accum_reward = self.run_epoc( q_table , epsilon )
            
            if self.out_filename != None :
                self.save_q_table_to_JSON( q_table , self.out_filename )
                q_table = self.load_q_table_from_JSON( self.out_filename )
            rospy.sleep(0)
            if self.plot_out_file != None :
                self.record_info['accumulated rewards'] += [accum_reward]
                self.plot(self.record_info ,self.plot_out_file)
        rospy.loginfo(f'-------Training Complete-------')
        self.reset_world()
        return q_table


    def run_epoc( self , q_table = None , epsilon = 0 , limit = 150 ) :
        '''
            runs a single epoc in training
        '''
        #  Wait for gazebo's first callback
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        if q_table == None :
            rospy.logwarn('Q table argument not specified in training cycle: --grabbing class Q table.')
            q_table = self.q_table
        flag_random_action = False
        self.reset_world()
        count = 0
        accum_reward = 0
        rospy.loginfo(f'----Resetting World-----')
        self.reset_world()
        x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
        self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)
        #  Waits for new lidar data after resetting the robot.
        self.cache['incoming scan data'] = None
        wait_counter = 0
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            wait_counter += 1
            if wait_counter > 5 :
                rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        rospy.loginfo(f'----- Initializing Simulation ------')

        state = self.scan_to_state(self.cache['scan data'].ranges)
        self.cache['state'] = state

        #  Gets the highest utility action for the robots current state
        action = max( q_table[state], key = q_table[state].get )
        action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
        if action == 'random' : 
            flag_random_action = True
            action = np.random.choice(list(self.actions.keys()))

        #  Converts Q table action to linear and angular velocities
        x , nz = self.actions[ action ]

        self.cache[ 'action' ] = action

        crash_queue = [(state,action, self.cache['scan data'].ranges)]

        #  Publishes linear and angular velocities as Twist object 
        self.publish_velocity( x = x , nz = nz )

        repeat_limit = 1000
        repeat_states = 0
        #rewards = 0
        while not rospy.is_shutdown() and count < limit and repeat_states < repeat_limit :
            rospy.sleep(0.1)

            #  Discretized scan data to Q table state information
            new_state = self.scan_to_state(self.cache['scan data'].ranges)
            repeat_states += 1

            if repeat_states > 0.9*repeat_limit  :
                rospy.loginfo(f'TIMEOUT')
                q_table = self.update_Q_table(q_table , self.cache['state'] , -100 , self.cache['action'] , self.cache['state'])
                break


            #  Resets simulation if robot is blocked
            if self.is_blocked(self.cache['scan data'].ranges) :
                #  update with crash reward....
                if len(crash_queue) > 1 :
                    for i in range(2,3) :
                        if len(crash_queue) > i :
                            s,a,_ = crash_queue[-i]
                            q_table = self.update_Q_table(q_table , s , -100 , a , crash_queue[-i+1][0])
                    self.publish_velocity( x = 0 , nz = 0 )
                    q_table = self.update_Q_table(q_table , crash_queue[-1][0] , -100 , self.cache['action'] , new_state)
                rospy.sleep(0.2)
                break

            if new_state != self.cache['state'] : pass

            state = self.cache['state']

        #  Gets the highest utility action for the robots current state
            flag_random_action = False
            action = max( q_table[new_state], key = q_table[new_state].get )
            action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
            if action == 'random' : 
                flag_random_action = True
                action = np.random.choice(list(self.actions.keys()))

            #  Converts Q table action to linear and angular velocities
            x , nz = self.actions[ action ]

            #  Publishes linear and angular velocities as Twist object 
            self.publish_velocity( x = x , nz = nz )

            reward = self.get_reward( new_state , self.cache['scan data'].ranges, crash_queue[-1][2] )

            if flag_random_action == False :
                accum_reward += reward
        #  update q table.
            q_table = self.update_Q_table( q_table , state , reward , self.cache[ 'action' ] , new_state )

            crash_queue += [ (new_state,action,self.cache['scan data'].ranges) ]

            self.cache[ 'state' ] = new_state
            self.cache[ 'action' ] = action

        return q_table, accum_reward


    def driving( self ) :
        rospy.loginfo(f'Starting Driving Mode')
        rospy.loginfo(f'-----------------------------')
        rospy.loginfo(f'--------  Driving Instructions -------')
        rospy.loginfo(f'c or esc to end the simulation')
        rospy.loginfo(f'w to drive forward')
        rospy.loginfo(f'a to turn left')
        rospy.loginfo(f'd to turn right')
        rospy.loginfo(f's to reverse')
        rospy.loginfo(f'e to stop')
        rospy.loginfo(f'r to reset the robot position')
        kb = KBHit()



        self.reset_world()
        x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
        self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)
        #  Waits for new lidar data after resetting the robot.
        self.cache['incoming scan data'] = None
        wait_counter = 0
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            wait_counter += 1
            if wait_counter > 5 :
                rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        rospy.loginfo(f'----- Initializing Simulation ------')
        state = self.scan_to_state(self.cache['scan data'].ranges)
        self.cache['state'] = state

        #  Gets the highest utility action for the robots current state
        action = max( self.q_table[state], key = self.q_table[state].get )

        #  Converts Q table action to linear and angular velocities
        x , nz = self.actions[ action ]

        self.cache[ 'action' ] = action

        crash_queue = [(state,action, self.cache['scan data'].ranges)]

        while True:

            if kb.kbhit():
                key = kb.getch()
                if ord(key) == 27: # ESC
                    break

                if key == 'q' or key == 'c' :
                    sys.exit(2)
                if key == 'w' :
                    self.publish_velocity(x=0.2,nz=0)
                if key == 'a' :
                    self.publish_velocity(x=0.2,nz=self.hard_turn_vel)
                if key == 'd' :
                    self.publish_velocity(x=0.2,nz=-self.hard_turn_vel)
                if key == 's' :
                    self.publish_velocity(x=-0.2 , nz=0)
                if key == 'e' :
                    self.publish_velocity(x=0 , nz=0)
                if key == 'r' :
                    self.reset_world()
                    x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
                    nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
                    self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)
                    self.publish_velocity(x=0,nz=0)
            new_state = self.scan_to_state(self.cache['scan data'].ranges)

            if new_state != self.cache['state'] : pass
            repeat_states = 0
            #count += 1  
            state = self.cache['state']

            #  Gets the highest utility action for the robots current state
            action = max( self.q_table[new_state], key = self.q_table[new_state].get )
            #action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
            #if action == 'random' : 
            #    action = np.random.choice(list(self.actions.keys()))
            #  Converts Q table action to linear and angular velocities
            x , nz = self.actions[ action ]


            reward = self.get_reward( new_state , self.cache['scan data'].ranges, crash_queue[-1][2] )
            crash_queue += [ (new_state,action,self.cache['scan data'].ranges) ]
            print(reward)
            rospy.sleep(1)
            self.cache[ 'state' ] = new_state
            self.cache[ 'action' ] = action


        kb.set_normal_term()


    
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
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        if q_table == None :
            q_table = self.q_table

        rospy.loginfo('Demo: initialized')
        self.reset_world()
        x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        nx,ny,nz,w = tuple(quaternion_from_euler(0,0,theta))
        self.set_model_pos(x,y,nx=nx,ny=ny,nz=nz,w=w)
        count = 0
        
        while not rospy.is_shutdown() and count < limit :
            rospy.sleep(0.1)

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
            if self.is_blocked(self.cache['scan data'].ranges) :
                rospy.loginfo(f'----Resetting World-----')
                self.reset_world()
                x,y,theta = self.start_positions[np.random.choice(range(len(self.start_positions)))]
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
        self.reset_world()
        rospy.loginfo(f'Ending Simulation')
        


    def is_blocked( self, scanArray) :
        '''
            NOTE! either check every lidar scanner or just the front. (right now checking everything.)
        '''
        for range in scanArray :
            if range < 0.2 :
                return True
        return False


    def construct_blank_q_table( self , q_table : dict, states : list , actions : list, default_value = 0 ) :
        action_dict = {}
        for a in actions :
            action_dict[a] = default_value
        for state in states :
            q_table[state] = action_dict.copy()
        return q_table



    def plot( self,plot_info : dict, outfile = 'default plots') :
        '''
            Plot_info = {
                total epocs. # num
                utilized epocs. # num
                percent of correct actions for each state/epoc. # list
                total accumulated reward/epoc. # list
            }
        '''
        if plot_info['total epocs'] == None : return False
        epocs = list(range(len(plot_info['accumulated rewards']))) # list  

        accum_rewards = plot_info['accumulated rewards']


        fig, ( accumulated_reward_plot , Correct_States_plot ) = plt.subplots(2,1)
        accumulated_reward_plot.set_xlim([0, plot_info['total epocs']])

        accumulated_reward_plot.plot(epocs , accum_rewards, color="#6c3376", linewidth=3)
        accumulated_reward_plot.set_title('Accumulated Rewards')


        Correct_States_plot.plot(epocs , accum_rewards, color="#6c3376", linewidth=3)
        Correct_States_plot.set_title('Correct States')

        fig.align_labels()
        plt.tight_layout()

        plt.savefig(os.path.join(sys.path[0], f'{outfile}.pdf'))
        plt.close()







import os

# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select


class KBHit:

    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.
        '''

        if os.name == 'nt':
            pass

        else:

            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)


    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.
        '''

        if os.name == 'nt':
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        '''

        s = ''

        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')

        else:
            return sys.stdin.read(1)


    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        '''

        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]

        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode('utf-8')))


    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        if os.name == 'nt':
            return msvcrt.kbhit()

        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []






def help() :
    rospy.logwarn(f'\
        -h <Headless mode ENABLED (DISABLED by default)> \n \
        --train\t <flag to enable training> \n\
        --demo\t <flag to demo a cycle> \n\
        --SARSA\t <flag to enable SARSA learning strategy> \
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
    SARSA = False
    driving = False


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


    arguments = ["--in_filename","--out_filename", "--plot_out_file", "--num_epocs", "--demo_rounds", "--train", "--demo","--SARSA", "--help", '-h']

    if len(argv) == 0 :
        rospy.logwarn(f'ERROR: No arguments passed')
        rospy.logwarn(f'arguments are {arguments}')
        rospy.logwarn(f'EXAMPLE: \n \
                      \t $ rosrun wallfollowing q_learning.py --train --out_filename Test_Q_table --num_epocs=100\n\n \
                      \t $ rosrun wallfollowing q_learning.py --demo --in_filename Test_Q_table --demo_rounds=50')
        rospy.logwarn(f'\n for more information run, \n\n \
                      \t $ rosrun wallfollowing q_learning.py --help')
        return False

    protected_files = ['best_Q_table', 'Manual_Q_table']

    try :
      opts, args = getopt.getopt(argv, "h",
                                ["in_filename=","out_filename=", "plot_out_file=", "num_epocs=", "demo_rounds=", "train", "demo", "SARSA", "drive", "help"])
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
        if opt == '--SARSA' :
            SARSA = True
        if opt == '--train' :
            train = True
        if opt == '--demo' :
            demo = True
        if opt == '--drive' :
            driving = True
        if opt == '--help' :
            help()

    q_learning(out_filename,in_filename,train,num_epocs,headless,demo,demo_rounds,plot_out_file, SARSA, driving)



if __name__ == '__main__':
    main(sys.argv[1:])
    