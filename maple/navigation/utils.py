from math import hypot

def get_distance_between_points(x1, y1, x2, y2):
    return hypot(x1 - x2, y1 - y2)