#!/usr/bin/env python3
import itertools
import json
import rospy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class q_learning() :
    def __init__( self ) :
        self.cache = {'scan data' : None ,
                 'state' : None}
        self.actions = ['straight' , 'slight left' , 'slight right' , 'hard right' , 'hard left']

        '''
            A single state is defined by the status of 
            each element in the tuple (right , front , left)
        '''
        self.directional_states = {
            'right' : ['to close' , 'good' , 'medium', 'far'],
            'front' : ['to close' , 'good' , 'medium', 'far'],
            'left' : ['to close' , 'good' , 'far'],
        }

        self.q_table = {}


        '''
            Detect walls on the right.
                Right side --> all base states
                front --> all base states
                left --> to close , good , far
                behind --> none
        '''

        new_states = []
        for direction,states in self.directional_states.items() :
            mod_states = []
            for s in states :
                mod_states += [ direction + ' : ' + s ]
            new_states += [ mod_states ]

        new_states = [s for s in itertools.product(*new_states)]

        self.q_table = self.construct_blank_q_table( {} , new_states , self.actions , default_value=0 )

        #self.save_q_table_to_JSON( self.q_table , 'Manual_Q_table' )


        self.init_node()
        self.init_subscriber()
        self.demo( self.q_table )

    def init_node( self ) :
        rospy.init_node( 'wall_following' , anonymous=True )


    def init_subscriber(self) :
        rospy.Subscriber('scan' , LaserScan , callback=self.scan_callback)


    def scan_callback( self , scan ) :
        '''
            Possibly encode state, then check if state has been updated?
            http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html
        '''
        pass


    def get_state_from_sub_scan( self , sub_scan , states ) :
        '''
            Convert states into actual ranges
            check which state the minimum of range belongs to
            return that state.
        '''
        pass


    def save_q_table_to_JSON( self , q_table , filename) :
        '''
            Save q table to JSON format to be reloaded later
        '''
        if filename in ['Manual_Q_table'] :
            rospy.logerr(f'filename: {filename} is protect, save Q table to a different filename')
            return None
        string_q_table = {}
        for key in q_table.keys() :
            str_key = ''
            for s in key :
                str_key += s + '\t'
            string_q_table[str_key] = q_table[key]
        with open(f"{filename}.json" , "w" ) as f:
            json.dump( string_q_table , f , indent=4 )


    def load_q_table_from_JSON( self , filename ) :
        '''
            loads q table from JSON file.
        '''
        pass


    def run_epoc( self ) :
        '''
            runs a single epoc in training
        '''
        pass
    

    def demo( self , q_table = None ) :
        '''
            Demos an agent in Gazebo/RviZ
        '''
        while rospy.is_shutdown() :
            rospy.sleep(0)


    def construct_blank_q_table( self , q_table : dict, states : list , actions : list, default_value = 0 ) :
        print(actions)
        action_dict = {}
        for a in actions :
            action_dict[a] = 0
        for state in states :
            q_table[state] = action_dict
        return q_table



if __name__ == '__main__':
    q_learning()