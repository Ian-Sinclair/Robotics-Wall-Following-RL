# COMP-4510-Project-3

## Table Of Contents

- /launch/wallfollow.launch 
    - Launch file containing model and world information

- /src/q_learning.py
    - Primary learning model, can also be used for demoing q tables 

- /src/Manual_Q_table.JSON
    - Manually created Q table for demo in task 1


## Watch the video for task 1
[![Watch the video for part 1]](https://youtu.be/2-q_YRZvHpM)


## Running setup files for project 3 task 1

First in its own terminal start the launch file.
```console
$ roslaunch wallfollowing wallfollow.launch
```
If this throws an error, you may need to resource the terminal
```console
$ cd catkin_ws
$ source devel/setup.bash
$ roslaunch wallfollowing wallfollow.launch
```

Next, i a new terminal run the demo file
```console
$ rosrun wallfollowing q_learning.py
```

This will run a demo of the manually created Q table