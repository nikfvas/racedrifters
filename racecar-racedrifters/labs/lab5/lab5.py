"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

import enum

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Add any global variables here

class State(enum.Enum):
    INITIAL = enum.auto()
    TURN_LEFT = enum.auto() 
    TURN_RIGHT = enum.auto()
    RIGHT_WALL = enum.auto()
    LEFT_WALL = enum.auto()
    LINE_FOLLOWING = enum.auto() 

BLUE = ((90, 50, 50), (120, 255, 255), 'blue')  # The HSV range for the color blue
RED1 = ((0, 50, 50), (10, 255, 255), 'red1')  # The HSV range for the color red
RED2 = ((170, 50, 50), (179, 255, 255), 'red2')  # The HSV range for the color red
GREEN = ((40, 50, 50), (80, 255, 255), 'green')  # The HSV range for the color green
ORANGE = ((15, 50, 50), (25, 255, 255), 'orange')  # The HSV range for the color orange
GRAY = ((0, 0, 65), (10, 10, 170), 'gray')  # The HSV range for the color gray
WHITE = ((0, 0, 240), (10, 10, 255), 'white')  # The HSV range for the color white

NEAR_WALL = 40
RIGHT_WINDOW = (0, 75)
LEFT_WINDOW = (285, 0)
FRONT_WINDOW = (350, 10)
FRONT_RIGHT = (20, 50)
FRONT_LEFT = (310, 340)

CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
current_color = None

########################################################################################
########################################################################################
# Functions
########################################################################################
########################################################################################

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
    global color
    global current_color

    last_color = current_color
    if current_color != color is not None:
        contour = get_color_contour(image, color)
        if contour is not None:
            print(f'{current_color=} returns to {color=}')
            current_color = color
            return contour
    contour = None
    if current_color is not None:
        contour = get_color_contour(image, current_color)
    if contour is None:
        for color in color_list:
            if color != current_color and color != color:
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
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    speed = 1
    angle = 0

    global state
    global color_list
    
    color_image = rc.camera.get_color_image()
    markers = rc_utils.get_ar_markers(color_image)

    if markers:
        marker = markers[0]
        id = marker.get_id()
    else:
        id = None

    # Lidar stuff
    scan = rc.lidar.get_samples()
    front_dist = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)[1]
    left_dist = rc_utils.get_lidar_closest_point(scan, FRONT_LEFT)[1]
    right_dist = rc_utils.get_lidar_closest_point(scan, FRONT_RIGHT)[1]
    
    # TODO: Turn left if we see a marker with ID 0 and right for ID 1
    if id is not None and id == 0:
        state = State.LEFT_WALL
    elif id == 1:
        state = State.RIGHT_WALL

    # TODO: If we see a marker with ID 199, turn left if the marker faces left and right
    # if the marker faces right

    elif id == 199 and marker.get_orientation() == rc_utils.Orientation.LEFT:
        state = State.LEFT_WALL
    elif id == 199 and marker.get_orientation() == rc_utils.Orientation.RIGHT:
        state = State.RIGHT_WALL

    # TODO: If we see a marker with ID 2, follow the color line which matches the color
    # border surrounding the marker (either blue or red). If neither color is found but
    # we see a green line, follow that instead.

    elif id == 2:
        marker.detect_colors(color_image, [RED2, BLUE, GREEN])
        color = marker.get_color
        best_contour(color_image)


########################################################################################

    if state == State.RIGHT_WALL:
        dist = rc_utils.get_lidar_closest_point(scan, RIGHT_WINDOW)[1]
        
        dx = dist - NEAR_WALL
        desired_angle = rc_utils.remap_range(dx, 0, NEAR_WALL/2.5, 0, 1)
        angle = rc_utils.clamp(desired_angle, -1, 1)

        ## if front_dist < 140:
        ##     angle = rc_utils.clamp(right_dist - left_dist, -1, 1)

    elif state == State.LEFT_WALL:
        dist = rc_utils.get_lidar_closest_point(scan, LEFT_WINDOW)[1]
        
        dx = dist - NEAR_WALL
        desired_angle = rc_utils.remap_range(dx, 0, NEAR_WALL/2.5, 0, 1)
        angle = rc_utils.clamp(desired_angle, -1, 1)

        ## if front_dist < 140:
        ##     angle = rc_utils.clamp(right_dist - left_dist, -1, 1)

    elif state == State.LINE_FOLLOWING:
        color_list = ()
        if contour_center is not None:

            dx = contour_center[1] - 320
            dy = 480 - contour_center[0]
            TERMA_TIMONI = 0.6

            desired_angle = rc_utils.remap_range(dx/dy, 0, TERMA_TIMONI, 0, 1)
            angle = rc_utils.clamp(desired_angle, -1, 1)
   
    rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
