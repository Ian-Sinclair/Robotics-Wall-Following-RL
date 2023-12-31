# Robotics - Reinforcement Learning: Wall Following


## Abstract

Reinforcement learning algorithms are used to teach
an autonomous mobile robot to follow walls and avoid obstacles in
a fixed simulated environment. This involves learning an optimal
mapping from states to control actions that allows the robot to
maintain a fixed distance to walls without crashing. An optimal
policy can be formally described by the statement, if at time t the
robot is at a desired distance from a wall, dw , then at time t + 1,
this distance is maintained while continuously moving forward
and without crashing. Generally, a policy is preferred if it induces
the robot to cover more distance along the wall in less time.
Two strategies are employed, being off-policy temporal difference
learning (Q-learning) and on-policy SARSA. The effectiveness of
either strategy is determined by the learning convergence rate
and accuracy (episodes vs. correct actions)

## Demo

Demo of turtlebot navigating the circumfrence of it's environment, along with training cycles for Q-learning and SARSA.

[![Watch the video](misc/images/demo_thumbnail.png)](https://youtu.be/Ce2aRc1InvM)

## Summary

Considered is a navigation problem that allows an au-
tonomous agent to target its control strategy by its local
surroundings. Wall following constitutes having the robot
move along the circumference of its environment without
crashing. In addition, a robot that covers greater portions of the
wall faster is considered better. And correspondingly, there is a
balance between moving quickly vs moving safely. If the robot
is too close to the wall it will likely crash, vs too far away it
will lose the topology of the wall and get lost.

To simplify the learning problem, a single static environ-
ment is used during training. This is in contrast to a stochastic
environment commonly used during generalizable reinforce-
ment learning models.

<img src="misc/images/envImage.png" alt="drawing" width="50%"/>


### Design Parameters

#### States/ Actions

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

<img src="misc/images/stateSpaceImage.png" alt="drawing" width="50%"/>

        The action space A is a set of control inputs that affect the
        velocity of the robot. The RL model is tasked with selecting
        a single action from the action space for each state. And
        correspondingly, the controller can only adjust the velocity
        of the robot if the state changes.

        Each state points to a single action in the action space.

            A = 'straight'
                'slight left'
                'hard left'
                'slight right'
                'hard right'
        Each of which publishes a corresponding velocity to the robot.


#### Rewards

both an intrinsic and
extrinsic reward is used to reward the robot for maintaining
a distance dw on its right from the wall, and punish it for
preforming actions that are risky or may make the robot
difficult to control in a real world setting. The extrinsic reward
uses the literal distance, d, from the robots right side to the
wall as input to the graduated reward function.

And the intrinsic reward, slightly punishes the use of risky
actions, for example, turning too quickly.

This is designed to promote the use of manageable turning
speeds in cases where they are acceptable, while still allowing
the algorithm to pick greater angular speeds if necessary to
prevent crashing.


### Evaluation of Success

Success is evaluated by two metrics, being the learning
convergence rate and accuracy. The convergence rate and accu-
racy are informed by selecting a few states with known ’best’
actions and comparing the ratio of steps where the algorithm
encounters a known state and also greedily selects the correct
action against the number of epocs. Faster convergence speed
indicates that the learning strategy is capable of learning an
’acceptable’ behavior with fewer training cycles. While, a
greater accuracy informs the quality of the policy each learning
strategy is capable of finding.

## Table Of Contents

- ros-packages/wallfollowing/launch/wallfollow.launch
  - Launch file containing model and world information

- ros-packages/wallfollowing/src/q_learning.py
  - Primary navigation software.
        - Is used to train new RL models, either using SARSA or Q learning
        - Demos Q tables in simulation

- ros-packages/wallfollowing/src/models/Optimal_Q_Table_TD.JSON
  - The best Q table for Temporal Difference Learning (try demoing)

- ros-packages/wallfollowing/src/models/Optimal_Q_TABLE_SARSA.JSON
  - The best Q table for SARSA learning (try demoing)

- ros-packages/wallfollowing/src/models/known_states_tracker.JSON
  - List of states and actions that are used to track the behavior during training.
  - Informs the learning convergence plots.

- ros-packages/wallfollowing/src/models/Test_Q_table.JSON
  - File placeholder for training throwaway Q tables (You can write over this)

## Run Setup Files

First in its own terminal start the launch file.

```console
roslaunch wallfollowing wallfollow.launch
```

If this throws an error, you may need to resource the terminal

```console
cd catkin_ws
source devel/setup.bash
roslaunch wallfollowing wallfollow.launch
```

## Navigation Software

The file

```console
/src/q_learning.py
```

Here you can train a new model, or demo a pre-saved Q table (behavior)

Run

```console
rosrun wallfollowing q_learning.py --help
```

For more information about how to start training/testing cycles.

## Training a new RL model

The simplest way to train a new model is with

```console
rosrun wallfollowing q_learning.py --train
```

This will launch a training cycle with all default parameters.  
However, it is more useful to specify some of your own parameters.  
Try running,

```console
rosrun wallfollowing q_learning.py --train --num_epocs=100 --out_filename Test_Q_table --plot_out_file 'Default Plots' --strategy 'Temporal Difference'
```

This will launch a training cycle for 100 episodes, and save the final q table to the file 'Test_Q_table'  
Note, all files are saved to the file location where the .py script is running and will write over any existing files. (Run carefully)  
Code Breakdown  

- num_epocs  <---- Number of episodes in a learning cycle
- out_filename <---- File name to save Q table
- plot_out_file  <---- file name to save convergence plots
- strategy <---- this is a mode section that can be 'Temporal Difference' or 'SARSA'

## Testing a model

Here the behavior of a Q table is tested in simulation,  
The Q table is note updated during this mode.  
The fastest way to demo a Q table is to run,

```console
rosrun wallfollowing q_learning.py --demo
```

This will automatically select the optimal Q table for temporal difference and demo it over 25 cycles.  
However, you can also select a different Q table.  
run

```console
rosrun wallfollowing q_learning.py --demo --in_filename 'Optimal_Q_Table_TD'
```

to demo the the best Q table for temporal difference  
Run

```console
rosrun wallfollowing q_learning.py --demo --in_filename 'Optimal_Q_Table_SARSA'
```

I think the SARSA Q table has better performance.  
to demo the best Q table for SARSA  
Finally, run

```console
rosrun wallfollowing q_learning.py --demo --in_filename 'Test_Q_table'
```

to demo the Q table you made in the previous section.

## Connecting to the Robot (Turtlebot 3 Waffle Pi)

Ensure the following is installed

```console
sudo apt-get install ros-kinetic-dynamixel-sdk
sudo apt-get install ros-kinetic-turtlebot3-msgs
sudo apt-get install ros-kinetic-turtlebot3
```

Export the waffle pi model,

```console
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
```

SSH into the robot in its own terminal.

```console
ssh ubuntu@192.168.9.{Robot Number}
```

Or turtlebot 1 and 2,

```console
ssh pi@192.168.9.{Robot Number}
```

Next, run the bring up software,

```console
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```

In a new terminal  
Find your wifi IP address under inet Addr, ###.###.#.###

```console
ifconfig
```

Update ./bashrc with the correct ROS master IP

```console
vim ~/.bashrc
```

The last line has the form,

```console
export ROS_MASTER_URI=http://192.168.9.{Robot Number}:11311
export ROS_HOSTNAME={Your Computers Host IP}
```

For help with vim, see the help with vim section.  
After updating and saving ~/.bashrc, source the terminal

```console
source ~/.bashrc
```

You can try to teleop the robot with,

```console
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

Or run the navigation software by following the next section.

## Running on the robot

After connecting to the ROS master on the robot, run the command,

```console
rosrun wallfollowing q_learning.py --robot --in_filename Optimal_Q_Table_SARSA
```

For best result, place robot near a wall before starting.

## Troubleshooting

It is likely you will need to resource every terminal you enter.

```console
cd catkin_ws
source devel/setup.bash
```

## Help With VIM

Start by entering insert mode by pressing 'I'.  
Make the necessary changes to the document.  
To save, press,  
'esc'  
':'  
'wq'  
Then press 'enter'  
The document is now saved, don't forget to resource the terminal after.
