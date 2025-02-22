
from maple.navigation.path import Path

class GlobalPath(Path):
    """This is the global path, it should never change as this is our main goal

    Args:
        Path (_type_): _description_
    """
    def __init__(self, target_locations):
        super().__init__(target_locations)

    def remove_current_goal_location(self):
        """This function should not work here because this is a permanet path
        """
        pass
        # return super().remove_current_goal_location()