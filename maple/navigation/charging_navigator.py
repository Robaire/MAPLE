from maple.utils import pytransform_to_tuple, carla_to_pytransform
from pytransform3d.transformations import concat, transform_from
from maple import geometry
from pytransform3d.rotations import matrix_from_euler
import numpy as np

class ChargingNavigator:
    """This class is used to navigate the rover to the charging atennae. It is called by the agent
    when the charging process is initiated and will take over the navigation of the rover until such
    a time as charging is either completed, or cancelled."""
    def __init__(self, agent):
        self.agent = agent
        self.battery_level = None # Must be set only once the agent is running
        # This is the start location for the rover
        self.rover_initial_position = carla_to_pytransform(agent.get_initial_position())

        # This is the start location for the lander
        lander_rover = carla_to_pytransform(agent.get_initial_lander_position())
        self.lander_global = concat(lander_rover, self.rover_initial_position)

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
        self.antenna_tag = concat(antenna_lander, self.lander_global)
    
    def navigate(self):
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
        return (None, None), False

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
        if current_power > self.battery_level:
            return True
        else:
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
        