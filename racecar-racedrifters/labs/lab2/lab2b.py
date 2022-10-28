"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 2B - Color Image Cone Parking
"""

########################################################################################
# Imports
########################################################################################

from operator import and_
from math import sqrt
import sys
import cv2 as cv
import numpy as np
import scipy as sp

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30
MAX_CONE_AREA = 27400.0
BRAKING_AREA = 5000.0
MAX_CONE_AREA_SQRT = sqrt(MAX_CONE_AREA)
BRAKING_AREA_SQRT = sqrt(BRAKING_AREA)
# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

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

def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle
    global go_back

    # Initialize variables
    speed = 0
    angle = 0
    go_back = False

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(">> Lab 2B - Color Image Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global go_back

    # Search for contours in the current color image
    update_contour()

    # Choose an angle based on contour_center
    # If we could not find a contour, keep the previous angle
    if contour_center is not None:
        # Current implementation: bang-bang control (very choppy)
        # TODO (warmup): Implement a smoother way to follow the line
        
        dx = contour_center[1] - 320
        dy = 480 - contour_center[0]
        TERMA_TIMONI = 0.6

        desired_angle = remap_range(dx/dy, 0, TERMA_TIMONI, 0, 1)
        angle = clamp(desired_angle, -1, 1)
   
    # Use the triggers to control the car's speed
    #forwardSpeed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    forwardSpeed = 1
    backSpeed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    
    # TODO: Park the car 30 cm away from the closest orange cone
    terma_speed = remap_range(contour_area, BRAKING_AREA, MAX_CONE_AREA, forwardSpeed, 0)

    if terma_speed < 0:
        terma_speed *= 10
    speed = clamp(terma_speed - backSpeed, -1, 1)

    if go_back:
        speed = -1
        if abs(angle) < go_back:
            go_back = False
    elif contour_area > MAX_CONE_AREA and (angle > 0.2 or angle < -0.2):
        go_back = abs(angle)/2
        speed = -1

    if speed < 0:
        angle = -angle

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)



def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the contour area and x position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
