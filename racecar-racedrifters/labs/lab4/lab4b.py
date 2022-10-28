"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 4B - LIDAR Wall Following
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here
NEAR_WALL = 40
RIGHT_WINDOW = (0, 75)
FRONT_WINDOW = (350, 10)
FRONT_RIGHT = (20, 50)
FRONT_LEFT = (310, 340)

########################################################################################
########################################################################################
# Functions
########################################################################################
########################################################################################

def clamp(value: float, vmin: float, vmax: float) -> float:
       
    if value > vmax:
        return vmax
    if value < vmin:
        return vmin
    return value

def remap_range(
    val: float,
    old_min: float,
    old_max: float,
    new_min: float,
    new_max: float,
    ) -> float:

    a = (new_max - new_min) / (old_max - old_min)
    b = (new_min * old_max - new_max * old_min) / (old_max - old_min)
    return a * val + b

########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 4B - LIDAR Wall Following")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    speed = 1
    angle = 0

    # TODO: Follow the wall to the right of the car without hitting anything.
    
    scan = rc.lidar.get_samples()
    dist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)[1]
    front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)[1]
    left_dist = rc_utils.get_lidar_closest_point(scan, FRONT_LEFT)[1]
    right_dist = rc_utils.get_lidar_closest_point(scan, FRONT_RIGHT)[1]
       
    dx = dist - NEAR_WALL
    desired_angle = remap_range(dx, 0, NEAR_WALL/2.5, 0, 1)
    angle = clamp(desired_angle, -1, 1)

    if front_dist < 140:
        angle = clamp(right_dist - left_dist, -1, 1)

    rc.drive.set_speed_angle(speed, angle)

    print(f'{front_dist=}')

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
