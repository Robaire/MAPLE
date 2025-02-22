
from maple.navigation.path import Path

class LocalPath(Path):
    """This is the local pathing that can remove points once we realise they are unreachable

    Args:
        Path (_type_): _description_
    """

    def __init__(self, target_locations):
        super().__init__(target_locations)

    def traverse(self, rover_position, obstacles=None, radius_from_goal_location=0.5):
        """Same traverse code except will remove points we cant reach

        Args:
            rover_position (_type_): _description_
            obstacles (_type_, optional): _description_. Defaults to None.
            radius_from_goal_location (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """

        # Gets the next possible goal location
        next_possible_goal_location = super().traverse(rover_position, radius_from_goal_location)

        # IMPORANT NOTE: This might result in an infinite loop
        # Repeat until we find a valid loop
        while not self.is_possible_to_reach(*rover_position, obstacles):
            next_possible_goal_location = super().traverse(rover_position, radius_from_goal_location)
        
        # Return the next possible path
        return next_possible_goal_location