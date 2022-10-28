"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3C - Depth Camera Wall Parking
"""
########################################################################################
# Imports
########################################################################################


import cv2 as cv
import numpy as np
import sys

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

from nptyping import NDArray
from typing import Any, Tuple, List, Optional

import enum

########################################################################################
# Global variables
########################################################################################

GOAL_DISTANCE = 20
BRAKE_DISTANCE = GOAL_DISTANCE + 25
MIN_LEGAL_DISTANCE = 12
FULL_TURN_DIFFERENCE = 10
MAX_TURN_RATIO = 0.1
MIN_SURE_TURN = 0.3
GOAL_PARK_RATIO = 0.2
PLAIN_PARK_DISTANCE = 4
KERNEL_SIZE = 15
Round = 0

rc = racecar_core.create_racecar()

# Add any global variables here
class State(enum.Enum):
    FAST_MOVE = enum.auto() # run towards the wall, braking (fast) if close enough
    SLOW_PARK = enum.auto() # we are close to the wall, adjust the position to the GOAL_DISTANCE - in low speed
    TOO_CLOSE = enum.auto() # go backwards, to align the car towards the wall.
    BACK_TURN = enum.auto() # go backwards, we are too close to the wall to trust depth measurements
    
########################################################################################
# Functions
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

def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()
    global x_distance
    global state
    x_distance = 0
    state = State.FAST_MOVE

    # Print start message
    global Round
    Round += 1
    print(f">> Lab 3C - Depth Camera Wall Parking (Round {Round})")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    global state
    global x_distance
    global angle_back, park_goal # remember it
    global left, right # for printing only

    # Calculate the distance of the object directly in front of the car
    depth_image = rc.camera.get_depth_image()
    blurred_image = cv.GaussianBlur(depth_image, (KERNEL_SIZE, KERNEL_SIZE), 0)

    # TODO: Park the car 20 cm away from the closest wall with the car directly facing
    # the wall

    yy, xx = rc.camera.get_height()*2//5, rc.camera.get_width()//2
    m_distance = blurred_image[yy][xx]
    distance = rc_utils.get_depth_image_center_distance(depth_image)
    left = blurred_image[yy][10]
    right = blurred_image[yy][-10]

    if left < MIN_LEGAL_DISTANCE:
        if right < MIN_LEGAL_DISTANCE:
            trust_depth, trust_angle, angle = False, False, 0
        else:
            trust_depth, trust_angle, angle = False, True, 1
    elif right < MIN_LEGAL_DISTANCE:
        trust_depth, trust_angle, angle = False, True, -1
    else:
        trust_depth, trust_angle = True, True
        if left - right > FULL_TURN_DIFFERENCE:
            angle = 1
        elif right - left > FULL_TURN_DIFFERENCE:
            angle = -1
        elif abs(left - right) / distance < MAX_TURN_RATIO:
            angle = 0
        elif abs(distance - x_distance) < 1 and abs(left - right) / distance < MIN_SURE_TURN:
            angle = 0
        else:
            angle = remap_range(left - right, 0, MAX_TURN_RATIO * GOAL_DISTANCE, 0, 0.5)
            angle = clamp(angle, -1, 1)
    trust_depth = distance > MIN_LEGAL_DISTANCE # do not use left/right

    if state == State.FAST_MOVE:
    
        if distance > BRAKE_DISTANCE:
            speed = 1
        else:
            speed = -1
            
            if trust_depth and distance > x_distance > MIN_LEGAL_DISTANCE:
                state = State.SLOW_PARK
                if abs(distance - GOAL_DISTANCE) < PLAIN_PARK_DISTANCE:
                    park_goal = GOAL_DISTANCE
                else:
                    park_goal = GOAL_DISTANCE + (distance - GOAL_DISTANCE) * GOAL_PARK_RATIO
                print(f'{m_distance:.2f}\t{distance:.1f}\t{x_distance:.1f}\t{state} {left:.0f} {right:.0f}')
            
    elif state == State.SLOW_PARK: # use forward or backward movement to park

        if trust_depth and abs(distance - x_distance) < 0.1:
            if abs(distance - GOAL_DISTANCE) < PLAIN_PARK_DISTANCE:
                park_goal = GOAL_DISTANCE
            else:
                park_goal = GOAL_DISTANCE + (distance - GOAL_DISTANCE) * GOAL_PARK_RATIO
        speed = remap_range(distance - park_goal, old_min = 0, old_max = 10, new_min = 0, new_max = 1)
        speed = clamp(speed, -0.5, 0.5)

        if not trust_depth:
            state = State.TOO_CLOSE
            print(f'{m_distance:.2f}\t{distance:.2f}\t{x_distance:.2f}\t{state} {left:.0f} {right:.0f}')
        elif trust_angle and abs(left - right) / distance > MAX_TURN_RATIO:
            angle_back = (left - right) / abs(left - right) 
            state = State.BACK_TURN
            print(f'{m_distance:.2f}\t{distance:.2f}\t{x_distance:.2f}\t{state} {left:.0f} {right:.0f}')
        
        if trust_angle and trust_depth and distance > x_distance > MIN_LEGAL_DISTANCE:
            angle = -angle

    elif state == State.BACK_TURN:
            
        speed = -1
        angle = - angle_back
            
        if not trust_depth or (left - right) * angle_back < 2:
            state = State.FAST_MOVE
            print(f'{m_distance:.2f}\t{distance:.2f}\t{x_distance:.2f}\t{state} {left:.0f} {right:.0f}')
            
    else: ## state == State.TOO_CLOSE: # do not use depth measurements (usually "None")
            
        speed = -1
        angle = - angle
    
        if trust_depth:
            state = State.SLOW_PARK
            print(f'{m_distance:.2f}\t{distance:.2f}\t{x_distance:.2f}\t{state} {left:.0f} {right:.0f}')
     
    if rc.controller.is_down(rc.controller.Button.RB):
        # Use the triggers to control the car's speed
        rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        speed = rt - lt

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    x_distance = distance


def update_slow():
    print(f'{x_distance:.1f}\t{state} {int(left)} {int(right)}')


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == '__main__':
    rc.set_start_update(start, update, update_slow and None)
    rc.go()
