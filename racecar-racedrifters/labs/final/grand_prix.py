"""
Copyright MIT and Harvey Mudd College
MIT License
Fall 2020

Final Challenge - Grand Prix
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
from cv2 import REDUCE_MAX
import numpy as np

from nptyping import NDArray
from typing import Any, Tuple, List, Optional

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

import ar_solver
import enum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

class State(enum.Enum):
    INITIAL = enum.auto()
    TURN_LEFT = enum.auto() 
    TURN_RIGHT = enum.auto()
    WALL_FOLLOWING = enum.auto()
    LANE_FOLLOWING = enum.auto()
    LINE_FOLLOWING = enum.auto() 
    SLALOM = enum.auto()
    CANYON = enum.auto()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 80
ALT_CONTOUR_AREA = 4 * MIN_CONTOUR_AREA

RIGHT_WINDOW = (30, 150)
LEFT_WINDOW = (210, 330)

# A crop window for the floor directly in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# Colors, stored as a pair (hsv_min, hsv_max)
BLUE = ((100, 150, 50), (110, 255, 255), 'blue')  # The HSV range for the color blue
RED1 = ((0, 50, 50), (10, 255, 255), 'red1')  # The HSV range for the color red
RED2 = ((170, 50, 50), (179, 255, 255), 'red2')  # The HSV range for the color red
RED3 = ((170, 50, 50), (10, 255, 255), 'red3')  # The HSV range for the color red
GREEN = ((50, 220, 220), (80, 255, 255), 'green')  # The HSV range for the color green
ORANGE = ((10, 200, 200), (20, 255, 255), 'orange')  # The HSV range for the color orange
GRAY = ((0, 0, 65), (10, 10, 170), 'gray')  # The HSV range for the color gray
WHITE = ((0, 0, 240), (10, 10, 255), 'white')  # The HSV range for the color white
PURPLE = ((110,59,50), (165,255,255), 'purple') # The HSV range for the color purple


# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour
state_color = None
current_color = None


########################################################################################
########################################################################################
# Functions
########################################################################################
########################################################################################

# Usual functions

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

# State: LINE FOLLOWING - Lab2a

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

def find_contours(mask): #### UNUSED
    """
    Returns a list of contours around all objects in a mask.
    """
    return cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]

def largest_color_contour(image, color): #### UNUSED
    global current_color
    # Find all of the contours in the color
    contours = rc_utils.find_contours(image, color[0], color[1])
    # Select the largest contour
    if current_color == color:
       min_area = MIN_CONTOUR_AREA
    else:
       min_area = ALT_CONTOUR_AREA
    contour = rc_utils.get_largest_contour(contours, min_area)
    if contour is not None:
        current_color = color
    return contour

def best_contour(image):
    # Find all of the blue contours

    global current_color

    last_color = current_color
    if current_color != state_color is not None:
        contour = get_color_contour(image, state_color)
        if contour is not None:
            print(f'{current_color=} returns to {state_color=}')
            current_color = state_color
            return contour
    contour = None
    if current_color is not None:
        contour = get_color_contour(image, current_color)
    if contour is None:
        for color in color_list:
            if color != current_color and color != state_color:
                contour = get_color_contour(image, color)
                if contour is not None:
                    current_color = color
                    if current_color != last_color:
                        print(f'detected {current_color=}')
                        return contour

    if contour is None:
        current_color = None
    return contour

def get_color_contour(color_image, color):
    ## mask = get_mask(color_image, color[0], color[1])
    contours = rc_utils.find_contours(color_image, color[0], color[1])
    largest_contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return largest_contour

def get_color_contour_center(color_image, color):
    largest_contour = get_color_contour(color_image, color)
    if largest_contour is None:
        return None
    return rc_utils.get_contour_center(largest_contour)

def get_white_checkpoint():
    color_image = rc_utils.crop(full_image, (rc.camera.get_height()-80, 0), (rc.camera.get_height(), rc.camera.get_width()))
    contours = rc_utils.find_contours(color_image, WHITE[0], WHITE[1])
    return contours

def pair_color_contours_center(color): ####
    color_image = rc_utils.crop(full_image, (20, 0), (rc.camera.get_height(), rc.camera.get_width()))

    contours = rc_utils.find_contours(color_image, color[0], color[1])
    if not contours:
        return None
    count = 0
    sums = [0, 0]
    for contour in contours:
        center = rc_utils.get_contour_center(contour)
        if center is not None:
            sums[0] += center[0]
            sums[1] += center[1]
            count += 1
    if count != len(contours):
        print(f'pair {count} from {len(contours)}.')
    if not count:
        return None
    return [sums[0] / count, sums[1] / count]

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
    global contour
    global full_image
    global crop_image

    full_image = rc.camera.get_color_image()

    if full_image is None:
        contour_center = None
        contour_area = 0
    else:
        # TODO (challenge 1): Search for multiple tape colors with a priority order
        # (currently we only search for blue)

        # Crop the image to the floor directly in front of the car
        crop_image = rc_utils.crop(full_image, CROP_FLOOR[0], CROP_FLOOR[1])

        # Select the largest contour of some color
        contour = best_contour(crop_image)
        ### contour = get_color_contour(image, state_color)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(crop_image, contour)
            rc_utils.draw_circle(crop_image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(crop_image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global state

    # Initialize variables
    state = State.INITIAL

    # Set initial driving speed and angle
    rc.drive.stop()

    # Set update_slow to refresh every half second
    ## rc.set_update_slow_time(0.5)

    # Print start message
    print(">> Final Challenge - Grand Prix")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global state
    global state_color
    global state_side
    global state_next
    global state_next_color
    global state_retry
    global state_ready
    global color_list
    global last_info
    global current_color

    if state == State.INITIAL:
        speed = 0
        angle = 0
        last_info = None
        state_next = None
        state, state_color = State.LINE_FOLLOWING, GREEN
    info = ar_solver.get_markers_info()
    if not info or info is last_info:
        pass
    elif info[-1] == 'ORANGE Lane Following':
        state_next, state_next_color, state_ready, state_retry = State.LANE_FOLLOWING, ORANGE, 0, 3
    elif info[-1] == 'PURPLE Lane Following':
        state_next, state_next_color, state_ready, state_retry = State.LANE_FOLLOWING, PURPLE, 0, 3
    elif info[-1] == 'Follow BLUE line':
        state, state_color = State.LINE_FOLLOWING, BLUE
    elif info[-1] == 'Follow GREEN line':
        state, state_color = State.LINE_FOLLOWING, GREEN
    elif info[-1] == 'Follow RED line':
        state, state_color = State.LINE_FOLLOWING, RED3
    elif True:
        pass
    elif info[-1] == 'Turn Left':
        state = State.TURN_LEFT
    elif info[-1] == 'Turn Right':
        state = State.TURN_RIGHT
    elif info[-1] == 'Slalom':
        state = State.SLALOM

    update_contour()
    color_image = full_image # rc.camera.get_color_image()

    # Lidar stuff
    scan = rc.lidar.get_samples()
    _, right_dist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)
    _, left_dist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)

    if state == State.LINE_FOLLOWING:
        color_list = ()
        if contour_center is not None:        
            dx = contour_center[1] - 320
            dy = 480 - contour_center[0]
            TERMA_TIMONI = 0.8

            desired_angle = remap_range(dx/dy, 0, TERMA_TIMONI, 0, 1)
            angle = clamp(desired_angle, -1, 1)
            print(f'line\t{angle = :.2f} {state_next=}')
        if state_next is not None:
            if not get_white_checkpoint():
                if state_ready:
                    state, state_color = state_next, state_next_color
                    state_next = None
                else:
                    state_ready = 1
    elif state == State.LANE_FOLLOWING:
        color_list = (ORANGE, PURPLE)
        if contour_center is not None:
            if contour_center[1] < 320:
                dx = contour_center[1] - 20
            else:
                dx = contour_center[1] - (640 - 20)
            dy = 480 - contour_center[0]
            TERMA_TIMONI = 0.8

            desired_angle = remap_range(dx/dy, 0, TERMA_TIMONI, 0, 5)
            angle = clamp(desired_angle, -1, 1)
            print(f'lane\t{angle = :.2f}')
        else:
            state_retry -= 1
            if state_retry == 0:
                state, state_color = State.LINE_FOLLOWING, GREEN
                state_next = None

    elif False:
        center = pair_color_contours_center(state_color)
        if center is None:       
            center = pair_color_contours_center(PURPLE)
        if center is None:
            state = State.LINE_FOLLOWING
        if center is not None:       
            dx = center[1] - 320
            dy = 480 - center[0]
            TERMA_TIMONI = 0.8

            desired_angle = remap_range(dx/dy, 0, TERMA_TIMONI, 0, 1)
            angle = clamp(desired_angle, -1, 1)
            print(f'lane\t{angle = :.2f}')

    elif state == State.TURN_LEFT:
        for color in (BLUE, GREEN, RED3):
            new_center = get_color_contour_center(full_image, color)
            if new_center is not None and new_center[1] < 300:
                state_color = color
                break

        print(f'{state_color[2]=}')
        state = State.LINE_FOLLOWING

    elif state == State.TURN_RIGHT:
        for color in (BLUE, GREEN, RED3):
            new_center = get_color_contour_center(full_image, color)
            if new_center is not None and new_center[1] > 340:
                state_color = color
                break

        print(f'{state_color[2]=}')
        state = State.LINE_FOLLOWING
    
    elif state == State.CANYON:
        if left_dist < right_dist:
            angle = -0.8
        if left_dist > right_dist:
            angle = 0.8

    elif state == State.SLALOM:
        # p1challenge
        pass

    speed = 1

    if rc.controller.was_pressed(rc.controller.Button.A):
        speed -= 0.2

    if rc.controller.was_pressed(rc.controller.Button.Y):
        speed -= 0.4

    if info:
        print('info:  ', info)

    last_info = info # even when empty!
    ##print(f'{angle =  }')

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.X):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
