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

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


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
                        SARSA = False ) :


        if num_epocs < 1 and train == True :
            rospy.logerr(f'ERROR: number of episodes is {num_epocs} must be at least 1')
        
        if train == True and demo == True :
            rospy.logwarn(f' Both training and demo is {True}, training will occur first')


        self.init_robot_control_params()

        self.init_node()
        self.init_subscriber()
        self.init_publisher()
        self.init_services()

        rospy.on_shutdown(self.shutdown)

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

            new_states = [s for s in itertools.product(*new_states)]

            self.q_table = self.construct_blank_q_table( {} , new_states , self.actions , default_value=0 )

        #  Loads Q table from file
        if type(in_filename) == type(' ') :
            rospy.loginfo(f'loading q table from file {in_filename}')
            self.q_table = self.load_q_table_from_JSON( in_filename )

        if out_filename != None :
            rospy.loginfo(f'saving q table from file {in_filename}')
            self.save_q_table_to_JSON( self.q_table , out_filename )

        if train == True :
            self.training( num_epocs = num_epocs )
            rospy.loginfo(f' Training Cycle Complete: DEMOing for 25 rounds ')
            self.demo( self.q_table , limit = 25 )

        if demo == True :
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
            'front' : [ ( 0 , 60 ) , ( 300 , 359 ) ] ,
            'left' : [ ( 45 , 135 ) ] ,
            'right_diagonal' : [ ( 300 , 330 ) ]
        }

        self.start_positions = [
            (-1.7,-1.7,1),
            (0,2,50),
            (1.9,0.7,1),
            (1.7,-1.7,1)
        ]


        #  List of linear speeds
        self.linear_actions = { 'fast' : 0.2 }

        self.slight_turn_vel = math.pi/8
        self.hard_turn_vel = math.pi/2

        
        self.rotational_actions = {'straight' : 0 , 
                                    'slight left' : self.slight_turn_vel , 
                                    'slight right' : -self.slight_turn_vel , 
                                    'hard right' : -self.hard_turn_vel , 
                                    'hard left' : self.hard_turn_vel 
                                    }

        #  List of all possible actions from any state.
        #self.actions = ['straight' , 'slight left' , 'slight right' , 'hard right' , 'hard left']

        self.actions = [s for s in itertools.product(*[self.linear_actions.keys() , self.rotational_actions.keys()])]

        temp = {}
        for v,w in self.actions :
            temp[f'linear: ({str(v)}) , angular: ({str(w)})'] = ( self.linear_actions[v] , self.rotational_actions[w] )
        
        self.actions = temp.copy()
        self.linear_actions = {'slow' : 0.1 , 'fast' : 0.2}

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

        fast_tr = 2*( self.linear_actions['fast'] / self.hard_turn_vel )  #  Fast turning radius

        #  Sets state thresholds to discretized scan distance data.
        self.thresholds = {
            'right' : {'close' : (0 , 0.6*d_w) , 'tilted close' : (0.6*d_w , 0.95*d_w) , 'good' : (0.95*d_w , 1.05*d_w), 'tilted far' : (1.05*d_w , 1.4*d_w) , 'far' : (1.4*d_w , 20)},
            'front' : {'close' : (0 , 2 * fast_tr) , 'far' : (2*fast_tr , 20)},
            'left' : {'close' : (0 , 1.5*fast_tr) , 'far' : ( 1.5*fast_tr , 20)},
            'right_diagonal' : { 'close' : (0 , 1.2*d_w) , 'far' : (1.2*d_w , 20) }
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


    def get_state_from_sub_scan( self , sub_scan , states ) :
        '''
            Convert states into actual ranges
            check which state the minimum of range belongs to
            return that state.
        '''
        pass


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
            string_q_table[str_key] = q_table[key]
            
        #  Dumps to JSON
        with open(os.path.join(sys.path[0], f'{filename}.json') , "w" ) as f:
            json.dump( string_q_table , f , indent=4 )
        return True


    def load_q_table_from_JSON( self , filename ) :
        """Loads q_table from JSON file at filename.
            EX: load_q_table_from_JSON( 'Manual_Q_table' )
            NOTE: File must be in same location as python script.
                /catkin_ws/src/wallfollowing/src/...

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



    def get_distance( self , scanArray, direction = 'right') :
        scanArray = list(np.clip(scanArray , a_min=0 , a_max=3.5))
        state_range = []
        ranges = self.scan_key[direction]
        for a,b in ranges :
            state_range += scanArray[a:b]
        return min(state_range)

    def get_reward( self , state , scanArray ) :
        if self.is_blocked(scanArray) : return -10


        '''
        if 'good' in state[0] :
            return 0

        if 'tilted' in state[0] :
            return -1
        return -2
        '''
        x = self.get_distance(scanArray , 'right')

        if ((x-self.d_w)**2)**0.5 > 0.5 : return -2

        if ((x-self.d_w)**2)**0.5 < 0.1 : return 1

        return -1
        #return -5*((x-self.d_w)**2)



    def update_Q_table( self, q_table , state , reward , action , new_state , gamma = 0.8, alpha = 0.2 ) :
        sample = reward + gamma*max(q_table[new_state].values())
        q_table[state][action] = (1-alpha) * q_table[state][action] + alpha * sample
        return q_table



    def training( self , num_epocs = 100 ) :
        rospy.loginfo(f'-------Initializing Training Cycle-------')

        #  Wait for gazebo's first callback
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        self.cache['incoming scan data'] = None
        
        epoc = 0
        epsilon = 1
        temp = epsilon

        while epoc < num_epocs and not rospy.is_shutdown() :
            rospy.loginfo(f'\t\t\t\t Running epoc {epoc+1}/{num_epocs}')
            epoc += 1
            epsilon = temp*(1-epoc/num_epocs)
            rospy.loginfo(f'-------- Epsilon:  {epsilon}  ----------')
            self.run_epoc( self.q_table , epsilon )
            if self.out_filename != None :
                self.save_q_table_to_JSON( self.q_table , self.out_filename )
            rospy.sleep(0)
        rospy.loginfo(f'-------Training Complete-------')
        self.reset_world()
        return True



    def run_epoc( self , q_table = None , epsilon = 0 , limit = 150 ) :
        '''
            runs a single epoc in training
        '''
        
        #  Wait for gazebo's first callback
        while self.cache['scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)

        if q_table == None :
            q_table = self.q_table

        self.reset_world()
        count = 0

        rospy.loginfo(f'----Resetting World-----')
        self.reset_world()
        x,y,nz = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        self.set_model_pos(x,y,nz=nz)
        #  Waits for new lidar data after resetting the robot.
        self.cache['incoming scan data'] = None
        while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            rospy.loginfo(f'-----Waiting for scan data-------')
            rospy.sleep(1)
        rospy.loginfo(f'----Scan Data Found----')

        state = self.scan_to_state(self.cache['scan data'].ranges)
        self.cache['state'] = state

        #  Gets the highest utility action for the robots current state
        action = max( q_table[state], key = q_table[state].get )
        action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
        if action == 'random' : 
            action = np.random.choice(list(self.actions.keys()))

        #  Converts Q table action to linear and angular velocities
        x , nz = self.actions[ action ]

        self.cache[ 'action' ] = action

        crash_queue = [(state,action)]

        #  Publishes linear and angular velocities as Twist object 
        self.publish_velocity( x = x , nz = nz )

        repeat_limit = 1000
        repeat_states = 0

        while not rospy.is_shutdown() and count < limit and repeat_states < repeat_limit :
            rospy.sleep(0.01)

            #self.cache['incoming scan data'] = None
            #while self.cache['incoming scan data'] == None and not rospy.is_shutdown() :
            #    rospy.sleep(0)

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
                    s,a = crash_queue[-2]
                    q_table = self.update_Q_table(q_table , s , -10 , a , new_state)
                    print('---------------------------------------------')
                    print(s)
                    print(self.cache['state'])
                    print(new_state)
                    print('---------------------------------------------')
                q_table = self.update_Q_table(q_table , new_state , -10 , self.cache['action'] , new_state)
                break


            if new_state != self.cache['state'] :
                repeat_states = 0
                count += 1  
                state = self.cache['state']

                #  Gets the highest utility action for the robots current state
                action = max( q_table[new_state], key = q_table[new_state].get )
                action = np.random.choice( [action , 'random'] , p=[ 1-epsilon , epsilon ] )
                if action == 'random' : 
                    action = np.random.choice(list(self.actions.keys()))

                #  Converts Q table action to linear and angular velocities
                x , nz = self.actions[ action ]
                #  Publishes linear and angular velocities as Twist object 
                self.publish_velocity( x = x , nz = nz )

                reward = self.get_reward( new_state , self.cache['scan data'].ranges )

                #  update q table.
                q_table = self.update_Q_table( q_table , state , reward , self.cache[ 'action' ] , new_state )

                crash_queue += [ (new_state,action) ]

                self.cache[ 'state' ] = new_state
                self.cache[ 'action' ] = action


    
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
        x,y,nz = self.start_positions[np.random.choice(range(len(self.start_positions)))]
        self.set_model_pos(x,y,nz=nz)
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
                x,y,nz = self.start_positions[np.random.choice(range(len(self.start_positions)))]
                self.set_model_pos(x,y,nz=nz)
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
        return False



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
                                ["in_filename=","out_filename=", "plot_out_file=", "num_epocs=", "demo_rounds=", "train", "demo", "SARSA", "help"])
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
            demo_rounds = arg
        if opt == '--demo_rounds' :
            if not arg.isnumeric() :
                rospy.logerr(f' arg: demo_rounds is {arg}, must be type integer numeric string ')
                rospy.logerr(f' EXAMPLE : $ rosrun wallfollowing q_learning.py  --demo_rounds 25 ')
                sys.exit(2)
            demo_rounds = arg
        if opt == '--SARSA' :
            SARSA = True
        if opt == '--train' :
            train = True
        if opt == '--demo' :
            demo = True
        if opt == '--help' :
            help()

    q_learning(out_filename,in_filename,train,num_epocs,headless,demo,demo_rounds,plot_out_file, SARSA)

if __name__ == '__main__':
    main(sys.argv[1:])
    