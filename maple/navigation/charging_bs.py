"""This charging routine assumes that it could drive straight towards the antenna. It will arrive at
its final location, lower the drums, and then rotate to make contact with the charger. It will then 
move back and forth if necessary to make a good connection."""

from maple.utils import pytransform_to_tuple, carla_to_pytransform
from pytransform3d.transformations import concat, transform_from, invert_transform
from maple import geometry
from pytransform3d.rotations import matrix_from_euler
import numpy as np
import carla
from math import radians
from maple.navigation.drive_control import DriveController

# TODO: Implement a way to keep track of mission time for the purposes of timing certain operations.

class ChargingNavigator:
    """This class is used to navigate the rover to the charging atennae. It is called by the agent
    when the charging process is initiated and will take over the navigation of the rover until such
    a time as charging is either completed, or cancelled."""
    def __init__(self, agent):
        self.agent = agent
        self.battery_level = None # Needs to be set in the agent
        # This is the start location for the rover
        self.rover_initial_position = carla_to_pytransform(agent.get_initial_position())


        # This is the start location for the lander
        lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_global = concat(lander_rover, self.rover_initial_position)

        self.lander_x, self.lander_y, _, self.lander_yaw, _, _ = pytransform_to_tuple(self.lander_global)

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
        antenna_lander = concat(tag_correction, transform_from(rotation, translation))
        self.antenna_pose = concat(antenna_lander, self.lander_global)

        # Keep track of the stage of navigation
        self.stage = 'initial_rotate'
        self.stage_list = ['initial_rotate', 'approach', 'drum grab', 'rotate', 'jiggle']

        # Lazily keeping track of time through this
        self.approach_time = 90 # hit with back drum
        # self.approach_time = 45 # hit with front drum
        self.drum_grab_time = 100
        self.flap_delay = 100

        self.jiggle_forward = 10
        self.jiggle_backward = 15
        self.jiggle_counter = 0
        self.is_jiggle_forward = True

        self.first_call = True

        self.drive_control = DriveController()
    
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
            raise ValueError('Invalid stage.')
        # Update the battery level on each iteration
        # self.battery_level = self.agent.get_current_power()

        if self.first_call:
            self.prev_battery = self.agent.get_current_power()

            self.first_call = False

        # Calculate the distance and yaw to the antenna
        rover2antenna = concat(invert_transform(rover_global), self.antenna_pose)
        rover2antenna_tuple = pytransform_to_tuple(rover2antenna)
        rover2antenna_dist = np.linalg.norm(rover2antenna_tuple[:3])
        rover2antenna_yaw = rover2antenna_tuple[5]

        # Calculate the rover yaw
        rover_x, rover_y, _, rover_yaw, _, _ = pytransform_to_tuple(rover_global)

        # Implement the charging routine
        if self.stage == 'initial_rotate':
            # Assuming we are in a good starting state rotate until we are properly facing the lander

            # IMPORTANT TODO: Add in cormac IMU angle control
            turn_angle = rover_yaw - self.lander_yaw
            
            # Turn until we are at a good enough angle to drive towards goal
            if turn_angle > .05:
                # IMPORTANT TODO: Add in the PID code in drive_control
                return (0, turn_angle), None
            else:
                # Change the state
                self.stage = 'approach'

        if self.stage == 'approach':
            
            # Just charge with everything lowered, will work out the details later
            self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Close)
            self.agent.set_front_arm_angle(radians(0))
            self.agent.set_back_arm_angle(radians(0))

            # Switch states when we are within the lander
            # TODO: Make a utils file for functions like this
            distance_squared = (rover_x - self.lander_x)**2 + (rover_y-self.lander_y)**2
            if distance_squared < 2:
                self.stage = 'drum grab'

            # Ensure we are driving straight as possible
            return self.drive_control.get_lin_vel_ang_vel_drive_control_straight(rover_x, rover_y, rover_yaw), None

        elif self.stage == 'drum grab':

            # Grab the charger with the drumes so we know where we are
            self.agent.set_front_arm_angle(radians(135))
            self.agent.set_back_arm_angle(radians(135))

            # Switch states when we run out of time
            self.drum_grab_time -= 1
            if self.drum_grab_time < 0:
                self.stage = 'rotate'

            return (0., 0.), None

        elif self.stage == 'rotate':

            if self.check_charging():
                print(f'hell ya boys!!! We charging!!')

            if self.agent.get_mission_time() > 25:
                self.stage = 'jiggle'

            # Try to charge
            if 19 > self.agent.get_mission_time() > 14:
                self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Open)
            
            if self.agent.get_mission_time() > 19:
                print(f'lowering everything')
                self.agent.set_front_arm_angle(radians(0))
                self.agent.set_back_arm_angle(radians(0))
                if self.agent.get_mission_time() > 21:
                    print(f'opening the cover')
                    self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Open)
                else:
                    print(f'closing the cover')
                    self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Close)
            else:
                self.agent.set_front_arm_angle(radians(90))
                self.agent.set_back_arm_angle(radians(90))

            if self.agent.get_mission_time() > 17:
                return (0, 0), None
            return (-1., 3.), None
        
        elif self.stage == 'jiggle':

            if self.check_charging():
                print(f'hell ya boys!!! We charging!!')


            # Always try to raise the cover
            self.agent.set_radiator_cover_state(carla.RadiatorCoverState.Open)

            # Go back and forth while raising the flap to push to get to charge
            if self.is_jiggle_forward:

                # Go forward a set amount before switching
                self.jiggle_counter += 1
                if self.jiggle_counter >= self.jiggle_forward:
                    self.jiggle_counter = 0
                    self.is_jiggle_forward = False

                return (.1, 0.), None
            
            else:
                # Go forward a set amount before switching
                self.jiggle_counter += 1
                if self.jiggle_counter >= self.jiggle_backward:
                    self.jiggle_counter = 0
                    self.is_jiggle_forward = True

                return (-.1, 0.), None


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
        if current_power > self.prev_battery:
            self.prev_battery = self.agent.get_current_power()
            return True
        else:
            self.agent.get_current_power()
            return False
        
if __name__ == "__main__":
    """This is a demonstration of how to incorporate the ChargingNavigator class into the agent."""
    class fake_agent:
        def __init__(self):
            self.battery_level = 10.
            self.charging_flag = False
        def get_current_power(self):
            return self.battery_level
        
    agent = fake_agent()
    ChargeRoutine = ChargingNavigator(agent)
    if agent.get_current_power() <= 11:
        agent.charging_flag = True
    if agent.charging_flag:
        control, agent.charging_flag = ChargeRoutine.navigate()
        