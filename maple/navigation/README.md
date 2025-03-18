
# Navigation structure breakdown

This is a breif description of the navigation breakdown



## File break down

- PythonRobotics -> Forked repo helper code for path planning when obstacles are in the way.

- static_path_planning -> Helper code to generate a static path we wish to travel.

- drive_control -> Code to generate the needed linear and angular velocity output needed based off of the controller needs.

- navigator -> Interface to switch between different states of drive control which will also call the drive_control.

- state folder -> A folder containing code for every state we can be in.

    - static -> Code to follow the static path (ran when there are no obstacles in the way).

    - dynamic -> Code to to follow a dynamic path generated to get around obstacles.

- helper_code -> Code used for testing and viewing the path, meant to run outside of sim.
## Input/Outputs

#### drive_control (IMPORTANT TODO: Add in kalman filtering)

| Parameters | Type     | Returns                | Description|
| :-------- | :------- | :------------------------- | :-- |
| `current_xy_location`, `current_goal_location` | `(x, y)`, `(x, y)` | `(goal linear vel, goal anguler vel)` | Take in the current predecited location and goal location and return the goal linear and angular velocity|

#### navigator

| Parameters | Type                    | Description|
| :-------- | :------- | :------------------------- |
| `agent` | `object` | This takes in the agent and is what should be used for calling this class, Ex: get_lin_vel_ang_vel gets the linear and angular velocity|

#### state folder

| Description|
|:------------------------- |
|The navigator will cycle through states within this folder for specific scenarios|

#### helper_code

| Description|
|:------------------------- |
|The code in here is completely used for testing and viewing the main code|
