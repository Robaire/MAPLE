"""
Class to store all the constant values used throughout the Leaderboard
"""

class Constants(object):


    def __init__(self, qualifier, testing):

        # Geometric map
        if qualifier:
            self.map_size = 9                                           # m
        else:
            self.map_size = 27                                          # m
        self.cell_size = 0.15                                           # m
        self.cell_number = int(self.map_size/self.cell_size)            # units
        self.total_cells = self.cell_number * self.cell_number          # units

        # Battery refilling
        self.refill_duration = 2 * 3600                                 # s
        self.refill_x_threshold = 0.1                                   # m
        self.refill_y_threshold = 0.1                                   # m
        self.refill_yaw_threshold = 30                                  # deg
        self.refill_reset_distance = 1                                  # m

        # Off-nominal terminations
        if testing:
            self.max_mission_duration = 30                              # s
            self.max_simulation_duration = 10                           # s
        elif qualifier:
            self.max_mission_duration = 1 * 3600                        # s
            self.max_simulation_duration = 7.5 * 3600                   # s
        else:
            self.max_mission_duration = 24 * 3600                       # s
            self.max_simulation_duration = 180 * 3600                   # s

        self.min_vehicle_power = 5                                      # Wh. The vehicle stops moving at ~3
        self.blocked_min_speed = 0.1                                    # m/s
        self.blocked_max_time = 5 * 60                                  # s
        self.bounds_distance = 19.5                                     # m (Half a meter away from the actual mesh end)

        # Geometric map metric
        self.geometric_map_max_score = 300.0
        self.geometric_map_min_score = 0.0
        self.geometric_map_threshold = 0.05                             # m

        # Rocks map metric
        self.rock_max_score = 300.0
        self.rock_min_score = 0.0

        # Mapping productivity metric
        self.mapping_productivity_score_rate = self.total_cells / self.max_mission_duration        # cell / s
        self.mapping_productivity_max_score = 250.0
        self.mapping_productivity_min_score = 0.0

        # Fiducials metric
        self.fiducials_max_score = 150.0
        self.fiducials_min_score = 0.0
