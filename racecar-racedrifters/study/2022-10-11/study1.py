#!/usr/bin/python3
import sys
import math
import enum

import racecar_core

### import racecar_utils as rc_utils
### from racecar_utils import clamp, remap_range, find_contours, get_largest_contour, get_contour_center, crop, lidar_closest_point, print_warning, print_colored
from racecar_utils import *

rc = racecar_core.create_racecar()

class State(enum.Enum):
    INITIAL = enum.auto() # Initial: go to a predefined state
    GO_BRAKE = enum.auto() ###
    DRIVE_WITH_OBSTACLES = enum.auto() # obstacle_gonia
    BRAKE_TIGHT = enum.auto() # brake_distance
    BRAKE_LIDAR_EXACT = enum.auto() # state, stop_distance, prev_distance, limit_distance
    BRAKE_BLINDLY = enum.auto()
    WALL_FOLLOWING = enum.auto() # side
    HALL_FOLLOWING = enum.auto() # no side
    LANE_FOLLOWING = enum.auto() # color
    LINE_FOLLOWING = enum.auto() # color, side
    SLALOM = enum.auto() # side, radius
    SLALOM_AROUND = enum.auto() # side, radius
    SLALOM_MIDWAY = enum.auto() # side
    MANUAL = enum.auto() # Use manual controls to navigate and select state
    AUTO = enum.auto() # Select an obvious / common state
    DEMO = enum.auto() # Select an interesting state (by looking around)
    QUIT = enum.auto() # Stop the motor and wait.

TURN_DEGREE = 36 ### ### ###

BLUE = ((100, 150, 50), (110, 255, 255), 'blue')  # The HSV range for the color blue
RED = ((170, 50, 50), (10, 255, 255), 'red')  # The HSV range for the color red
GREEN = ((50, 220, 220), (80, 255, 255), 'green')  # The HSV range for the color green
ORANGE = ((10, 200, 200), (20, 255, 255), 'orange')  # The HSV range for the color orange
GRAY = ((0, 0, 65), (10, 10, 170), 'gray')  # The HSV range for the color gray
WHITE = ((0, 0, 240), (10, 10, 255), 'white')  # The HSV range for the color white
PURPLE = ((110,59,50), (165,255,255), 'purple') # The HSV range for the color purple

color_dict = {"orange": ORANGE, "blue": BLUE, "green": GREEN, "red": RED, "purple": PURPLE}

def echo(text):
    print_colored(text, TerminalColor.orange)

def sign1(x):
    if x < 0:
        return -1
    else:
        return 1

def g180(a):
    return (a+180) % 360 - 180

def degree_asin(val):
    return math.asin(val) / math.pi * 180

def dump(txt, n = 10, cc = [0]):
    if not (cc[0] % n):
        print(txt)
    cc[0] += 1

def remap_one(val, maxval):
    return clamp(val / maxval, -1, 1)

def remap_one_ang(val, maxval):
    global ang
    ang = val / maxval
    return clamp(ang, -1, 1)

def angle_asin(val, gonia):
    return remap_one(g180(degree_asin(val) - gonia), TURN_DEGREE)

def angle_asin_ang(val, gonia):
    return remap_one_ang(g180(degree_asin(val) - gonia), TURN_DEGREE)

def lidar_closest(window):
    return lidar_gonia_closest(window)[1]

def lidar_gonia_closest(window):
    global lidar_scan
    if lidar_scan is None: lidar_scan = rc.lidar.get_samples()
    return get_lidar_closest_point(lidar_scan, window)

def lidar_average(angle, window_angle):
    global lidar_scan
    if lidar_scan is None: lidar_scan = rc.lidar.get_samples()
    return get_lidar_average_distance(lidar_scan, angle, window_angle)

def autostate():
    '''
HERE WE WRITE WHAT THIS FUNCTION DOES
It returns ...
'''
    # lane
    for color in (ORANGE, PURPLE):
        contours = find_contours(full_image, color[0], color[1])
        if len(contours) == 2: return State.LANE_FOLLOWING, {'color': color}
    # line following / aiming
    for color in (GREEN, RED, BLUE):
        contours = find_contours(full_image, color[0], color[1])
        if contours:
            update_contour(CROP2_FLOOR)
            contours = find_contours(crop_image, color[0], color[1])
            if contours:
                return State.LINE_FOLLOWING, {'color': color}
            else:
                return State.LINE_AIMING, {'color': color}
    # wall
    if lidar_closest((0, 360)) < 50: return State.WALL_FOLLOWING, {} ### side? use lidar angle
    # orange cone
    contours = find_contours(full_image, ORANGE[0], ORANGE[1])
    if len(contours) == 1: return State.LINE_FOLLOWING, {'color': ORANGE} ## ?? parking?
    # red/blue cones
    if len(find_contours(full_image, RED[0], RED[1])) > 1 and len(find_contours(full_image, BLUE[0], BLUE[1])) > 1: return State.SLALOM, {}
    # white checkpoint
    contours = find_contours(full_image, WHITE[0], WHITE[1])
    if contours: return State.MANUAL, {} ##
    # if obstacle, move towards it
    if lidar_closest((0, 360)) < 500: return State.DRIVE_WITH_OBSTACLES, {}
    # move forward, braking if necessary
    return State.DRIVE_WITH_OBSTACLES, None
