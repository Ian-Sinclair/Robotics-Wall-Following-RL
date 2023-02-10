# COMP-4510-Project-3

## Running setup files for project 3 task 1

First start the launch file.
```console
$ roslaunch wallfollowing wallfollow.launch
```
If this throws an error, you may need to resource the terminal
```console
$ cd catkin_ws
$ source devel/setup.bash
$ roslaunch wallfollowing wallfollow.launch
```

Next, run the demo file
```console
rosrun wallfollowing q_learning.py
```