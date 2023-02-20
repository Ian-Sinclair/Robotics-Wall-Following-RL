#!/usr/bin/env python3
import sys, getopt
import rospy
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random


def plot( plot_info : dict, outfile = 'default plots') :
    '''
        Plot_info = {
            total epocs. # num
            utilized epocs. # num
            percent of correct actions for each state/epoc. # list
            total accumulated reward/epoc. # list
        }
    '''
    epocs = list(range(plot_info['total epocs'])) # list  

    accum_rewards = plot_info['accum_reward']

    accum_rewards += [None for i in range(len(accum_rewards) , plot_info['total epocs'])]

    x = epocs


    fig, ( accumulated_reward_plot , Correct_States_plot ) = plt.subplots(2,1)
    accumulated_reward_plot.set_xlim([0, plot_info['total epocs']])

    accumulated_reward_plot.plot(epocs , accum_rewards, color="#6c3376", linewidth=3)
    accumulated_reward_plot.set_title('Accumulated Reward')


    Correct_States_plot.plot(epocs , accum_rewards, color="#6c3376", linewidth=3)
    Correct_States_plot.set_title('Correct States')

    fig.align_labels()
    plt.tight_layout()

    plt.savefig(os.path.join(sys.path[0], f'{outfile}.pdf'))
    plt.close()



for i in range(100) :
    plot_info = {
        'total epocs' : 20 ,
        'accum_reward' : [random.randint(0,10) ,random.randint(0,10),3,random.randint(0,10),3,2,1,random.randint(0,10),12,random.randint(0,10),4]
    }

    plot(plot_info)
    time.sleep(1)