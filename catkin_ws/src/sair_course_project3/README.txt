Author Name: Ryan Dunagan
Package Name: "sair_course_project3"

#########################
##### PART 1 README #####
#########################

First, run "roscore" and boot up Gazebo with the command "roslaunch sair_course_project3 wallfollow.launch".
This will only start up Gazebo, not my node (this is intentional). This is because my node takes in command
line arguments, and I was having trouble getting the cli args to play nice with launch files. (It also helps
to see the console output for my node in its own terminal window)

"Wallfollow_Agent.py" is the file that defines the wallfollow node functionality.

Next, run my node with the command "rosrun sair_course_project3 Wallfollow_Agent.py". If no additional arguments
are given, this will default into training mode. You can give it one of three arguments: training, demo, manual.

training: Runs training episodes, just as if no arguments were given.
demo: Plays back the learned behaviors from the q-table. (Not yet implemented)
manual: Allows you to manually execute actions with key presses.
    q = Left Long, w = Straight Long, e = Right Long, a = Left Short, s = Straight Short, d = Right Short

#########################
##### PART 2 README #####
#########################

First, run "roscore" and boot up Gazebo with the command "roslaunch sair_course_project3 wallfollow.launch".
This will only start up Gazebo, not my node (this is intentional). This is because my node takes in command
line arguments, and I was having trouble getting the cli args to play nice with launch files. (It also helps
to see the console output for my node in its own terminal window)

"Wallfollow_Agent.py" is the file that defines the wallfollow node functionality.

My best generated q-table can be found in sair_course_project3/src/q-table_best.npy. It is a pickled numpy
n-dimensional array, and isn't human readable. Similarly, sair_course_project3/src/q-table.npy is the most
recently created q-table (this is updated during every episode in training mode).

Next, run my node with the command "rosrun sair_course_project3 Wallfollow_Agent.py". If no additional arguments
are given, this will default into training mode and will not load an existing q-table. You can give it one of three
arguments: training, test, manual. If you are launching in training mode, you can tell it whether or not to load an
existing q-table with the "load" command, and you can load the best qtable with the "best" command.

Here are examples of commands that will run the node:

# Run the node in test mode with the best qtable with either of these commands
rosrun sair_course_project3 Wallfollow_Agent.py test
rosrun sair_course_project3 Wallfollow_Agent.py test best

# Run the node in test mode with the most recently made qtable with this command
rosrun sair_course_project3 Wallfollow_Agent.py test load

# Run the node in training mode and load the best qtable with this command
rosrun sair_course_project3 Wallfollow_Agent.py training best

# Run the node in training mode and load the most recently made qtable with this command
rosrun sair_course_project3 Wallfollow_Agent.py training load

# Run the node in training mode with an empty qtable with either of these commands
rosrun sair_course_project3 Wallfollow_Agent.py
rosrun sair_course_project3 Wallfollow_Agent.py training