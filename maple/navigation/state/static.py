import math
from typing import List

from maple.navigation.state.path import Path

class StaticPath(Path):
    """
    This is the static path, points that are unreachable are skipped
    """

    def __init__(self, target_locations):
        """ Only have 2 locations for the target locations, the start location and the end locations
        """
        super().__init__(target_locations)
        
        # The target locations are everthing we are going to try and hit
        self.path = target_locations