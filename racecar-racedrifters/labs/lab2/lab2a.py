"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 2A - Color Image Line Following
"""

########################################################################################
# Imports
########################################################################################

from curses import color_content
from lib2to3.pgen2.token import GREATER
from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
import sys
import cv2 as cv
from cv2 import REDUCE_MAX
import numpy as np
from nptyping import NDArray
from typing import Any, Tuple, List, Optional

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
ALT_CONTOUR_AREA = 4 * MIN_CONTOUR_AREA

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Colors, stored as a pair (hsv_min, hsv_max)
BLUE = ((90, 50, 50), (120, 255, 255), 'blue')  # The HSV range for the color blue
RED1 = ((0, 50, 50), (10, 255, 255), 'red1')  # The HSV range for the color red
RED2 = ((170, 50, 50), (179, 255, 255), 'red2')  # The HSV range for the color red
GREEN = ((40, 50, 50), (80, 255, 255), 'green')  # The HSV range for the color green
ORANGE = ((15, 50, 50), (25, 255, 255), 'orange')  # The HSV range for the color orange
ANYCOLOR = ((0, 50, 50), (179, 255, 255), 'anycolor')  # The HSV range for all lines
GRAY = ((0, 0, 65), (10, 10, 170), 'gray')  # The HSV range for the color gray
WHITE = ((0, 0, 240), (10, 10, 255), 'white')  # The HSV range for the color white

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
contour_color = None

########################################################################################
########################################################################################
########################################################################################
# Functions
########################################################################################
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

def get_mask(
    image,
    hsv_lower,
    hsv_upper
): # -> NDArray[Any, Any]:

    hsv_lower = np.array(hsv_lower)
    hsv_upper = np.array(hsv_upper)
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv, hsv_lower, hsv_upper)
    return mask

def find_contours(mask):
    """
    Returns a list of contours around all objects in a mask.
    """
    return cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

def get_contour_center(contour):

    # Ask OpenCV to calculate the contour's moments
    M = cv.moments(contour)

    # Check that the contour is not empty
    if M["m00"] <= 0:
        return None

    # Compute the center of mass of the contour
    center_row = round(M["m01"] / M["m00"])
    center_column = round(M["m10"] / M["m00"])
    
    return (center_row, center_column)

def largest_color_contour(image, color):
    global contour_color
    # Find all of the contours in the color
    contours = rc_utils.find_contours(image, color[0], color[1])
    # Select the largest contour
    if contour_color == color[2]:
       min_area = MIN_CONTOUR_AREA
    else:
       min_area = ALT_CONTOUR_AREA
    contour = rc_utils.get_largest_contour(contours, min_area)
    if contour is not None:
        contour_color = color[2]
    return contour

def best_contour(image):
    # Find all of the blue contours
    global contour_color
    last_color = contour_color
    contour = largest_color_contour(image, RED1)
    if contour is None:
        contour = largest_color_contour(image, RED2)
    if contour is None:
        contour = largest_color_contour(image, GREEN)
    if contour is None:
        contour = largest_color_contour(image, ORANGE)
    if contour is None:
        contour = largest_color_contour(image, BLUE)
    if contour is None:
        contour = largest_color_contour(image, ANYCOLOR)
    if contour is None:
        contour_color = "No contour"
    if contour_color != last_color:
        print("New contour color:", contour_color)
    return contour

########################################################################################
########################################################################################
########################################################################################


def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area

    full_image = rc.camera.get_color_image()

    if full_image is None:
        contour_center = None
        contour_area = 0
    else:
        # TODO (challenge 1): Search for multiple tape colors with a priority order
        # (currently we only search for blue)

        # Crop the image to the floor directly in front of the car
        image = rc_utils.crop(full_image, CROP_FLOOR[0], CROP_FLOOR[1])

        # Select the largest contour of some color
        contour = best_contour(image)

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

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(
        ">> Lab 2A - Color Image Line Following\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    A button = print current speed and angle\n"
        "    B button = print contour center and area"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle

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
    forwardSpeed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    backSpeed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = forwardSpeed - backSpeed

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
    # Print a line of ascii text denoting the contour area and x-position
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
