# This is how far from our current rover position along the path that we want to be the point our rover is trying to go to
radius_from_goal_location = 2

# This is the speed we are set to travel at (.48m/s is max linear and 4.13rad/s is max angular)
goal_speed = .15
goal_hard_turn_speed = .01
# goal_hard_turn_ang_vel = 3

# This is the size we should treat the lander during navigation so we dont hit it
lander_size = 3

# These are max speeds but to reach require zero of the other speed according to documentation
max_linear_speed = .48
max_angular_speed = 4.13

# This is the time of the simulator, should be 20Hz
DT = .05