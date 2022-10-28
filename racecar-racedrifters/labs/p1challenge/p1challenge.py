"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Phase 1 Challenge - Cone Slaloming
"""

########################################################################################
# Imports
########################################################################################

from operator import and_
from math import dist, sqrt
import sys
from turtle import distance, st
import cv2 as cv
import numpy as np
import scipy as sp
import enum

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

from nptyping import NDArray
from typing import Any, Tuple, List, Optional

########################################################################################
# Global variables
########################################################################################

MIN_CONTOUR_AREA = 50
TERMA_TIMONI = 0.6

TIMONI = 0.5
DIRECTION = 10
TURN = 300
MAIN_DISTANCE = 120 # Go straight until main
AWAY_DISTANCE = 200 # the cone disappeared
INSIST = 8 # Times to stay on states RIGHT/LEFT
INSIST_AWAY = 4
INSIST_REVERSE = 20
INSIST_AGAIN =10
RADIUS = 75
KERNEL_SIZE = 11
RED = ((170, 50, 50), (10, 255, 255))
BLUE = ((100, 150, 50), (110, 255, 255))

# Add any global variables here

speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels

rc = racecar_core.create_racecar()

class State(enum.Enum):
    INITIAL = enum.auto()
    SETUP = enum.auto()
    RED = enum.auto()
    RIGHT = enum.auto()
    BLUE_AWAY = enum.auto()
    RIGHT_LEFT = enum.auto()
    RIGHT_AGAIN = enum.auto()
    RIGHT_RIGHT = enum.auto()
    BLUE = enum.auto()
    LEFT = enum.auto()
    RED_AWAY = enum.auto()
    LEFT_RIGHT = enum.auto()
    LEFT_AGAIN = enum.auto()
    LEFT_LEFT = enum.auto()
    FINISH = enum.auto()
    BACK = enum.auto()

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
########################################################################################
########################################################################################

def get_color_contour(image, color):
    # Find all of the contours in the color
    contours = rc_utils.find_contours(image, color[0], color[1])
    # Select the largest contour
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    return contour


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    global speed
    global angle
    global state
    angle = 0
    state = 0
    state = State.INITIAL

    rc.drive.set_speed_angle(speed, angle)
    # Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")

def update():
    """    update_contour()

    """
    global speed
    global angle
    global state
    global insist # internal

    # TODO: Slalom between red and blue cones.  The car should pass to the right of
    # each red cone and the left of each blue cone.

    color_image = rc.camera.get_color_image()

    depth_image = rc.camera.get_depth_image()
    blurred_image = cv.GaussianBlur(depth_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    speed = 1
    if state == State.RED:
        contour = get_color_contour(color_image, RED) # Select the largest contour
        if contour is None:
            angle = -0.9
            print('<')
            ## state = State.RIGHT
            ## print(state)
        else:
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx} ({state}) {rc_utils.get_contour_area(contour)}')
            ## angle = remap_range(dx + distance, -TURN, TURN, 0, 1)
            ## angle = remap_range(dx, -TURN, 0, 0, 1)
            angle = remap_range(dx, -TURN, TURN, 0, 1)
            ## angle = remap_range(distance - RADIUS, 0, 10, -0.1, -1)
            angle = clamp(angle, -1, 1)
            if distance > AWAY_DISTANCE:
                insist = INSIST
                state = State.RIGHT
                print(f'{distance:.1f} * {dx} {state} {rc_utils.get_contour_area(contour)}')

    elif state == State.RIGHT:
        angle = 0.5
        insist -= 1
        if insist == 0:
            insist = INSIST_AWAY
            state = State.BLUE_AWAY
            print(state)

    elif state == State.BLUE_AWAY:
        insist -= 1
        if insist == 0:
            insist = INSIST_REVERSE
            state = State.RIGHT_LEFT
            print(state)
        contour = get_color_contour(color_image, BLUE) # Select the largest contour
        if contour is None:
            angle = -0.3
            print('<')
        else:
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx} ({state}) {rc_utils.get_contour_area(contour)}')
            angle = remap_range(dx, -TURN, TURN, 0, 1)
            angle = clamp(angle, -1, 1)

    elif state == State.RIGHT_LEFT:
        angle = -1
        insist -= 1
        if insist == 0:
            insist = INSIST_AGAIN
            state = State.RIGHT_AGAIN
            print(state)

    elif state == State.RIGHT_AGAIN:
        angle = 1
        insist -= 1
        if insist == 0:
            state = State.RIGHT_RIGHT
            print(state)

    elif state == State.RIGHT_RIGHT:
        angle = 1
        contour = get_color_contour(color_image, BLUE) # Select the largest contour
        if contour is not None:
            state = State.BLUE
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx}', state)

    elif state == State.BLUE:
        contour = get_color_contour(color_image, BLUE) # Select the largest contour
        if contour is None:
            angle = 0.3
            print('>')
            ## state = State.LEFT
            ## print(state)
        else:
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx} ({state}) {rc_utils.get_contour_area(contour)}')
            ## angle = remap_range(-dx + distance, -TURN, TURN, -1, 0)
            ## angle = remap_range(dx / distance, 0, DIRECTION, -TIMONI, 0)
            ## angle = remap_range(dx , TURN, 0, 0,-1)
            angle = remap_range(dx, -TURN, TURN, -1, 0)
            angle = clamp(angle, -1, 1)
            if distance > AWAY_DISTANCE:
                insist = INSIST
                state = State.LEFT
                print(f'{distance:.1f} * {dx} {state} {rc_utils.get_contour_area(contour)}')

    elif state == State.LEFT:
        angle = -0.5
        insist -= 1
        if insist == 0:
            insist = INSIST_AWAY
            state = State.RED_AWAY
            print(state)

    elif state == State.RED_AWAY:
        insist -= 1
        if insist == 0:
            insist = INSIST_REVERSE
            state = State.LEFT_RIGHT
            print(state)
        contour = get_color_contour(color_image, RED) # Select the largest contour
        if contour is None:
            angle = 0.9
            print('>')
        else:
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx} ({state}) {rc_utils.get_contour_area(contour)}')
            angle = remap_range(dx, -TURN, TURN, -1, 0)
            angle = clamp(angle, -1, 1)

    elif state == State.LEFT_RIGHT:
        angle = 1
        insist -= 1
        if insist == 0:
            insist = INSIST_REVERSE
            state = State.LEFT_AGAIN
            print(state)

    elif state == State.LEFT_AGAIN:
        angle = -1
        insist -= 1
        if insist == 0:
            state = State.LEFT_LEFT
            print(state)

    elif state == State.LEFT_LEFT:
        angle = -1
        contour = get_color_contour(color_image, RED) # Select the largest contour
        if contour is not None:
            state = State.RED
            contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            print(f'{distance:.1f} * {dx}', state)

    elif state == State.FINISH:
        angle = -0.1
    elif state == State.BACK:
        pass

    elif state == State.SETUP: # Turn Right
        contour = get_color_contour(color_image, RED) # Select the largest contour
        contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
        if contour_center is None:
            angle = -0.2
            print('<')
        else:
            distance = blurred_image[contour_center[0]][contour_center[1]]
            dx = contour_center[1] - rc.camera.get_width()//2
            ## print(f'{distance:.1f} * {dx}', state)
            angle = remap_range(dx, -TURN, TURN, 0, 1)
            angle = clamp(angle, -1, 1)
            if distance < RADIUS:
                state = State.RED
                print(f'{distance:.1f} * {dx}', state)

    else: # state == State.INITIAL # Go towards the first cone, from distance 180
        contour = get_color_contour(color_image, RED) # Select the largest contour
        contour_center = rc_utils.get_contour_center(contour) # Calculate contour information
        distance = blurred_image[contour_center[0]][contour_center[1]]
        dx = contour_center[1] - rc.camera.get_width()//2
        ## print(f'{distance:.1f} * {dx}', state)
        angle = remap_range(dx, 0, 500, 0, 1)
        angle = clamp(angle, -1, 1)
        if distance < MAIN_DISTANCE:
            state = State.SETUP
            print(f'{distance:.1f} * {dx}', state)

    if speed < 0:
        angle = -angle
    
    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
