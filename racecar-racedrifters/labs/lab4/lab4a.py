"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 4A - LIDAR Safety Stop
"""

########################################################################################
# Imports
########################################################################################

from sre_parse import State
import sys
from turtle import back, distance
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The (min, max) degrees to consider when measuring forward and rear distances
FRONT_WINDOW = (-10, 10)
REAR_WINDOW = (170, 190)
SAFE_WINDOW = (135, 225)
CLOSE_WINDOW = (175, 185)
BRAKE_DISTANCE = 72
FAR_AWAY = 100000

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

    global danger
    global speed
    global angle
    global x_distance
    global min_distance
    global back_range

    # Initialize variables
    danger = 0
    speed = 0
    angle = 0
    x_distance = 1000
    min_distance = 10000
    back_range = REAR_WINDOW

    # Print start message
    print(
        ">> Lab 4A - LIDAR Safety Stop\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Right bumper = override forward safety stop\n"
        "    Left trigger = accelerate backward\n"
        "    Left bumper = override rear safety stop\n"
        "    Left joystick = turn front wheels\n"
        "    A button = print current speed and angle\n"
        "    B button = print forward and back distances"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    global danger
    global speed
    global angle
    global x_distance # internal
    global min_distance # internal
    global back_range

    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance in front of and behind the car
    scan = rc.lidar.get_samples()
    _, forward_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    _, back_dist = rc_utils.get_lidar_closest_point(scan, back_range)
    _, close_dist = rc_utils.get_lidar_closest_point(scan, CLOSE_WINDOW)

    # TODO (warmup): Prevent the car from hitting things in front or behind it.
    # Allow the user to override safety stop by holding the left or right bumper.

    if danger == 1 and back_dist > FAR_AWAY:
        back_range = SAFE_WINDOW

    if back_dist < 100:
        danger = 1
    if back_dist < BRAKE_DISTANCE:
        if back_dist > FAR_AWAY:
            if close_dist > 1000:
                speed = 1
            else:
                speed = 0.20
        elif back_dist > min_distance + 0:
            speed = clamp(remap_range(back_dist - min_distance - 2, 0, 10, 0, -0.2), -1, 1)
        else:
            speed = 1
    else:
        speed = -1

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    x_distance = back_dist

    if back_dist < min_distance:
        min_distance = back_dist

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the distance of the closest object in front of and behind the car
    if rc.controller.is_down(rc.controller.Button.B):
        print("Back distance:", back_dist, "Back distance:", back_dist)

    # Display the current LIDAR scan
    rc.display.show_lidar(scan)
    print(f'{back_dist= :.2f}   {min_distance = :.2f} {speed = :.2f}')


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
