"""This charging routine assumes that it could drive straight towards the antenna. It will arrive at
its final location, lower the drums, and then rotate to make contact with the charger. It will then
move back and forth if necessary to make a good connection."""

from maple.utils import pytransform_to_tuple, carla_to_pytransform
from pytransform3d.transformations import concat, transform_from, invert_transform
from maple import geometry
from pytransform3d.rotations import matrix_from_euler
import numpy as np
import carla
from maple.navigation.drive_control import DriveController

# TODO: Implement a way to keep track of mission time for the purposes of timing certain operations.


class ChargingNavigator:
    """This class is used to navigate the rover to the charging atennae. It is called by the agent
    when the charging process is initiated and will take over the navigation of the rover until such
    a time as charging is either completed, or cancelled."""

    def __init__(self, agent):
        self.agent = agent
        self.drive_control = DriveController()
        self.battery_level = None  # Needs to be set in the agent
        self.prev_battery_level = None
        # This is the start location for the rover
        self.rover_initial_position = carla_to_pytransform(agent.get_initial_position())

        # This is the start location for the lander
        lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_global = concat(self.rover_initial_position, lander_rover)

        # Store the location of the locator tag
        # Correct for the apriltag coordinate convention
        tag_correction = transform_from(
            matrix_from_euler(
                [np.pi / 2, 0, -np.pi / 2],
                2,
                1,
                0,
                False,
            ),
            [0, 0, 0],
        )
        tag = geometry.lander["locator"]
        translation = [tag["x"], tag["y"], tag["z"]]
        rotation = matrix_from_euler([np.deg2rad(90), 0, 0], 2, 1, 0, False)
        tag_lander = concat(tag_correction, transform_from(rotation, translation))
        self.locator_tag = concat(tag_lander, self.lander_global)
        # Also store the pose of the antenna itself
        antenna = geometry.lander["antenna"]
        translation = [antenna["x"], antenna["y"], antenna["z"]]
        rotation = matrix_from_euler([np.deg2rad(90), 0, 0], 2, 1, 0, False)
        rotation = matrix_from_euler([0, 0, 0], 2, 1, 0, False)
        translation = [0, 1.452, 0.509]
        # antenna_lander = concat(tag_correction, transform_from(rotation, translation))
        antenna_lander = transform_from(rotation, translation)
        self.antenna_pose = concat(self.lander_global, antenna_lander)

        # Keep track of the stage of navigation
        self.stage = "approach"
        self.stage_list = ["approach", "lower", "rotate", "back_and_forth", "done"]

    def navigate(self, rover_global):
        """This function is called by the agent to navigate the rover to the charging atenna.
        Inputs:
        - None
        Parameters:
        - current pose of the rover as a pytransform
        - current pose of the charging antenna as a pytransform
        - whether or not the charging fiduciary is in view of the rover
        - whether or not the rover is currently charging
        Outputs:
        - control commands for the rover as a tuple of (linear_velocity, angular_velocity)
        - a binary flag indicating whether or not the rover needs to continue with the charging routine"""
        if self.stage not in self.stage_list:
            raise ValueError("Invalid stage.")
        # Update the battery level on each iteration
        self.prev_battery_level = self.battery_level
        self.battery_level = self.agent.get_current_power()

        # Calculate the distance and yaw to the antenna
        # print("Antenna pose:", self.antenna_pose)
        # print("Rover global:", rover_global)
        # print("Invert rover global:", invert_transform(rover_global))
        # print("Antenna Pose", self.antenna_pose)
        rover2antenna = concat(invert_transform(rover_global), self.antenna_pose)
        # print("Rover2ant:", rover2antenna)
        rover2antenna_tuple = pytransform_to_tuple(
            rover2antenna
        )  # Weird...transforms don't work. My head still worts thinking about them!

        # rover2antenna_dist = np.linalg.norm(rover2antenna_tuple[:2])
        # print("rover2ant tuple shrunk:", rover2antenna_tuple[:2])
        # print("Rover2Antenna tuple:", rover2antenna_tuple)
        rover2antenna_yaw = rover2antenna_tuple[5]
        antenna_x, antenna_y, _, _, _, _ = pytransform_to_tuple(self.antenna_pose)
        goal_y = antenna_y + 0.208 + 0.5
        # goal1_x = antenna_x + 1.0
        goal_x = antenna_x
        rover_x, rover_y, _, _, _, rover_yaw = pytransform_to_tuple(rover_global)
        rover2goal_dist = np.sqrt((rover_x - goal_x) ** 2 + (rover_y - goal_y) ** 2)
        rover2antenna_y = rover2antenna_tuple[1]
        print("Stage:", self.stage)

        # Implement the charging routine
        if self.stage == "approach":
            # print("Rover location:", [rover_x, rover_y, rover_yaw])
            #print("Goal:", [goal_x, goal_y])
            print("rover2goal dist:", rover2goal_dist)
            print("Arm angle:", self.agent.get_front_arm_angle())
            # Drive straight towards the antenna.
            # This could be made more intelligent using the existing path navigator.
            if rover2goal_dist <= 1.0 and rover2goal_dist >= 0.1:
                print("Yaw:", rover2antenna_yaw)
                if np.abs(rover2antenna_yaw) <= np.deg2rad(50) or (np.abs(rover2antenna_yaw) >=130 and np.abs(rover2antenna_yaw <= 180)):
                    # TODO: Perhaps add a dependency on the actual distance from the antenna?
                    self.agent.set_front_arm_angle(np.deg2rad(30))
                    self.agent.set_back_arm_angle(np.deg2rad(30))
                else:
                    self.agent.set_front_arm_angle(np.deg2rad(5))
                    self.agent.set_back_arm_angle(np.deg2rad(5))
            if rover2goal_dist <= 0.1:
                self.stage = "lower"
                control = (0.0, 0.0)
                # TODO: check if the drums need to be lowered, or if they collide with the ground.
                self.agent.set_front_arm_angle(np.deg2rad(20))
                self.agent.set_back_arm_angle(np.deg2rad(20))
            else:
                # Ensure we are driving straight as possible
                pid_control = self.drive_control.get_lin_vel_ang_vel_drive_control(
                    rover_x, rover_y, rover_yaw, goal_x=antenna_x, goal_y=antenna_y
                )
                control = [0.1 * pid_control[0], pid_control[1]]
                if control[0] > 0.1:
                    control[0] = 0.1
                return control, True
        elif self.stage == "lower":
            # TODO: Add in some ability to check if the drums have been lowered. Could even be a timer.
            self.stage = "rotate"
            control = (0.0, 0.0)
        elif self.stage == "rotate":
            # Rotate to align with the charger
            control = (0.0, 0.5)
            if abs(rover2antenna_yaw) < np.deg2rad(50):
                self.agent.set_front_arm_angle(np.deg2rad(30))
                self.agent.set_back_arm_angle(np.deg2rad(30))
            if abs(rover2antenna_yaw) < 0.1:
                self.stage = "back_and_forth"
                self.current_time = self.agent.get_mission_time()
                self.charging_start_time = self.agent.get_mission_time()
        elif self.stage == "back_and_forth":
            self.current_time = self.agent.get_mission_time()
            # TODO: Implement a back and forth motion to make a good connection
            # Open the radiator cover
            self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Open)
            print("Radiator angle:", np.rad2deg(self.agent.get_radiator_cover_angle()))
            # Check if the rover is charging
            charged = self.check_charging()
            print("battery:",self.battery_level)
            if charged:
                self.stage = "done"
            if self.current_time - self.charging_start_time >= 10.:
                control = (np.random.randint(-1,2),0)
            else:
                control = (0.0, 0.0)
        elif self.stage == "done":
            #self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Closed)
            return (0, 0), False
        return control, True

    def check_charging(self):
        """This function is called by the agent to check the status of the charging process. It determines
        this by checking the current battery level of the rover.
        Inputs:
        - None
        Parameters:
        - whether or not the rover is currently charging
        Outputs:
        - whether or not the rover charged, True or False"""
        current_power = self.agent.get_current_power()
        if current_power > self.prev_battery_level:
            return True
        else:
            return False


if __name__ == "__main__":
    """This is a demonstration of how to incorporate the ChargingNavigator class into the agent."""

    class fake_agent:
        def __init__(self):
            self.battery_level = 10.0
            self.charging_flag = False

        def get_current_power(self):
            return self.battery_level

    agent = fake_agent()
    ChargeRoutine = ChargingNavigator(agent)
    if agent.get_current_power() <= 11:
        agent.charging_flag = True
    if agent.charging_flag:
        control, agent.charging_flag = ChargeRoutine.navigate()
