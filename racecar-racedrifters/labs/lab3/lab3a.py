"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3A - Depth Camera Safety Stop
"""

########################################################################################
# Imports
########################################################################################


from tkinter.tix import MAX
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import statistics
from nptyping import NDArray
from typing import Any, Tuple, List, Optional
########################################################################################
import sys
from turtle import distance, st
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

BRAKE_DISTANCE = 70
TOUCH_DISTANCE = 30
KERNEL_SIZE = 11
MAX_DEPTH = 250

rc = racecar_core.create_racecar()

# Add any global variables here

########################################################################################
# Functions
########################################################################################


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



########################################################################################
########################################################################################
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    global x_distance
    global state
    x_distance = 1000
    state = 0  # 0=go,decide 1=stop 2=run

    # Print start message
    print(
        ">> Lab 3A - Depth Camera Safety Stop\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Right bumper = override safety stop\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = print current speed and angle\n"
        "    B button = print the distance at the center of the depth image"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    global x_distance
    global state

    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance of the object directly in front of the car
    depth_image = rc.camera.get_depth_image()
    ##center_distance = rc_utils.get_depth_image_center_distance(depth_image)

    # TODO (warmup): Prevent forward movement if the car is about to hit something.
    # Allow the user to override safety stop by holding the right bumper.

    # Clip anything above max_depth
    np.clip(depth_image, None, MAX_DEPTH, depth_image)

    # Shift down slightly so that 0 (no data) becomes the "farthest" color
    depth_image = (depth_image - 1) % MAX_DEPTH

    blurred_image = cv.GaussianBlur(depth_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    top_left_inclusive = (rc.camera.get_height()*4//16, rc.camera.get_width()*1//8)
    bottom_right_exclusive = (rc.camera.get_height()*10//16, rc.camera.get_width()*7//8)

    cropped_image = crop(blurred_image, top_left_inclusive, bottom_right_exclusive)
    closest_pixel = get_closest_pixel(cropped_image)
    ##show_depth_image(cropped_image, points = [closest_pixel])

    distance = cropped_image[closest_pixel[0]][closest_pixel[1]]
    print(distance)
    below = blurred_image[rc.camera.get_height()*15//20][rc.camera.get_width()//2]
    extra_below = blurred_image[rc.camera.get_height()*19//20][rc.camera.get_width()//2]
    
    if state == 0 and below > 69 and extra_below > 37:
        state = 1
    if state == 0 and distance < TOUCH_DISTANCE:
        state = 1

    if state == 0 and below < 44 and extra_below < 28:
        state = 2

    if state == 1:
        if x_distance > distance:
            speed = -1
        else:
            speed = 0
    elif state == 0 and distance < BRAKE_DISTANCE:
        speed = 0.1
    
    if rc.controller.is_down(rc.controller.Button.RB):
        speed = -1
    
    x_distance = distance


    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

########################################################################################
# Functions
########################################################################################

    # TODO (stretch goal): Prevent forward movement if the car is about to drive off a
    # ledge.  ONLY TEST THIS IN THE SIMULATION, DO NOT TEST THIS WITH A REAL CAR.

    top_left_inclusive = (rc.camera.get_height()*5//8, 0)
    bottom_right_exclusive = (rc.camera.get_height(), rc.camera.get_width())
    
    cropped_image2 = crop(depth_image, top_left_inclusive, bottom_right_exclusive)



    # TODO (stretch goal): Tune safety stop so that the car is still able to drive up
    # and down gentle ramps.
    # Hint: You may need to check distance at multiple points.

    print('up:', blurred_image[rc.camera.get_height()*4//10][rc.camera.get_width()//2],
        'down:', blurred_image[rc.camera.get_height()*15//20][rc.camera.get_width()//2],
        'extra below: ', extra_below,
        'state =', state)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == '__main__':
    rc.set_start_update(start, update, None)
    rc.go()
