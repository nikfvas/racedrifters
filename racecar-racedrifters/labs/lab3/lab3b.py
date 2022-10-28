"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3B - Depth Camera Cone Parking
"""

########################################################################################
# Imports
########################################################################################

from operator import and_
from math import sqrt
import sys
from turtle import st
import cv2 as cv
import numpy as np
import scipy as sp

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

from nptyping import NDArray
from typing import Any, Tuple, List, Optional

########################################################################################
# Global variables
########################################################################################

MIN_CONTOUR_AREA = 30
ORANGE = ((10, 100, 100), (20, 255, 255))
TERMA_TIMONI = 0.6

GOAL_DISTANCE = 29
BRAKE_DISTANCE = 38
KERNEL_SIZE = 5
MAX_DEPTH = 501
STOP_DISTANCE = 1
STILL_DISTANCE = 0.2

# Add any global variables here

speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
x_distance = 1000
state = 0 # 0=going, 1=taxing

rc = racecar_core.create_racecar()

########################################################################################
########################################################################################
# Functions
########################################################################################
########################################################################################


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

def show_depth_image(
    depth_image: NDArray[(Any, Any), np.float32],
    max_depth: int = 400,
    points: List[Tuple[int, int]] = []
 ) -> None:
    """
    Displays a color image in the Jupyter Notebook.
    
    Args:
        depth_image: The image to display.
        max_depth: The farthest depth to show in the image in cm. Anything past this depth is shown as black.
        points: A list of points in (pixel row, pixel column) format to show on the image colored dots.
    """
    # Clip anything above max_depth
    np.clip(depth_image, None, max_depth, depth_image)

    # Shift down slightly so that 0 (no data) becomes the "farthest" color
    depth_image = (depth_image - 1) % max_depth

    # Convert depth values to colors
    color_image = cv.applyColorMap(-cv.convertScaleAbs(depth_image, alpha=255/max_depth), cv.COLORMAP_INFERNO)
    
    # Draw a dot at each point in points
    for point in points:
        cv.circle(color_image, (point[1], point[0]), 6, (0, 255, 0), -1)

    # Show the image with Matplotlib
    plt.imshow(cv.cvtColor(color_image, cv.COLOR_BGR2RGB))
    plt.show()

def crop(
    image: NDArray[(Any, ...), Any],
    top_left_inclusive: Tuple[float, float],
    bottom_right_exclusive: Tuple[float, float]
 ) -> NDArray[(Any, ...), Any]:
    """
    Crops an image to a rectangle based on the specified pixel points.

    Args:
        image: The color or depth image to crop.
        top_left_inclusive: The (row, column) of the top left pixel of the crop rectangle.
        bottom_right_exclusive: The (row, column) of the pixel one past the bottom right corner of the crop rectangle.

    Returns:
        A cropped version of the image.

    Note:
        The top_left_inclusive pixel is included in the crop rectangle, but the
        bottom_right_exclusive pixel is not.
        
        If bottom_right_exclusive exceeds the bottom or right edge of the image, the
        full image is included along that axis.
    """
    # Extract the minimum and maximum pixel rows and columns from the parameters
    r_min, c_min = top_left_inclusive
    r_max, c_max = bottom_right_exclusive

    # Shorten the array to the specified row and column ranges
    return image[r_min:r_max, c_min:c_max]

def get_closest_pixel(
    depth_image: NDArray[(Any, Any), np.float32],
    kernel_size: int = 5
 ) -> Tuple[int, int]:
    """
    Finds the closest pixel in a depth image.

    Args:
        depth_image: The depth image to process.
        kernel_size: The size of the area to average around each pixel.

    Returns:
        The (row, column) of the pixel which is closest to the car.

    Warning:
        kernel_size be positive and odd.
        It is highly recommended that you crop off the bottom of the image, or else
        this function will likely return the ground directly in front of the car.

    Note:
        The larger the kernel_size, the more that the depth of each pixel is averaged
        with the distances of the surrounding pixels.  This helps reduce noise at the
        cost of reduced accuracy.
    """
    # Shift 0.0 values to 10,000 so they are not considered for the closest pixel
    depth_image = (depth_image - 0.01) % 10000
    
    # TODO: Apply a gaussian blur using kernel_size
    blurred_image = cv.GaussianBlur(depth_image, (kernel_size, kernel_size), 0)

    # TODO: Find the pixel location of the minimum depth and return it
    #mask = np.zeros(blurred_image.shape[:2],np.uint8)
    #mask[0:blurred_image.shape[0]*5//8,0:blurred_image.shape[1]] = 255
    #minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(blurred_image, mask)

    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(blurred_image)
    return minLoc[1], minLoc[0]


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

    color_image = rc.camera.get_color_image()

    if color_image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(color_image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(color_image, contour)
            rc_utils.draw_circle(color_image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(color_image)

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global speed
    global angle
    global x_distance
    global state

    # Initialize variables
    speed = 0
    angle = 0
    x_distance = 1000
    state = 0

    rc.drive.set_speed_angle(speed, angle)

    # Print start message
    print(">> Lab 3B - Depth Camera Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global x_distance
    global state

    # TODO: Park the car 30 cm away from the closest orange cone.
    # Use both color and depth information to handle cones of multiple sizes.
    # You may wish to copy some of your code from lab2b.py

    update_contour()

    if contour_center is not None:
        # Current implementation: bang-bang control (very choppy)
        # TODO (warmup): Implement a smoother way to follow the line
        
        dx = contour_center[1] - 320
        dy = 480 - contour_center[0]

        desired_angle = remap_range(dx/dy, 0, TERMA_TIMONI, 0, 1)
        angle = clamp(desired_angle, -1, 1)
    else:
        angle = 0

    depth_image = rc.camera.get_depth_image()

    # Clip anything above max_depth
    np.clip(depth_image, None, MAX_DEPTH, depth_image)

    # Shift down slightly so that 0 (no data) becomes the "farthest" color
    depth_image = (depth_image - 1) % MAX_DEPTH

    blurred_image = cv.GaussianBlur(depth_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    new_distance = blurred_image[contour_center[0]][contour_center[1]]
    ##print(distance)

    if state == 0:
    	if new_distance < BRAKE_DISTANCE:
            if abs(x_distance - new_distance) > STOP_DISTANCE:
                speed = -1
            else:
                state = 1
    elif state == 1:
        if abs(GOAL_DISTANCE - new_distance) < STILL_DISTANCE:
           speed = 0
        else:
            speed = remap_range(new_distance - GOAL_DISTANCE, old_min = 0, old_max = 10, new_min = 0, new_max = 1)
            speed = clamp(speed, -0.5, 0.5)

    if speed < 0:
        angle = -angle
    
    x_distance = new_distance
    rc.drive.set_speed_angle(speed, angle)

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
    print('distance =', x_distance)
    print('speed =', speed)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
