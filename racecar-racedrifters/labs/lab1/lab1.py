"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 1 - Driving in Shapes
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()
activities = []

# Put any global variables here

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Begin at a full stop
    rc.drive.stop()
    activities.clear()

    # Print start message
    # DANE (main challenge): add a line explaining what the Y button does
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = drive in a circle\n"
        "    B button = drive in a square\n"
        "    X button = drive in a figure eight\n"
        "    Y button = drive in a 'Γ' \n"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # DONE (warmup): Implement acceleration and steering
    global activities
    speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT) - rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    if not(-0.01 < speed < 0.01):
        angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0] 
        rc.drive.set_speed_angle(speed, angle)
        activities.clear()
        return
    
    if rc.controller.was_pressed(rc.controller.Button.A):
        print("Driving in a circle...")
        # DONE (main challenge): Drive in a circle
        activities = [[1, 1, 5.4], [-1, 1, 0.3]]
        rc.drive.set_speed_angle(activities[0][0], activities[0][1])

    if rc.controller.was_pressed(rc.controller.Button.B):
        print("Driving in a square...")
        # DONE (main challenge): Drive in a square when the B button is pressed
        activities = [[1, 0, 1.17],[1, 1, 1.3]]
        for i in range(3):
            activities.extend([[1, 0, 0.8],[1, 1, 1.3]])
        activities.append([-1, 0, 0.5])
        rc.drive.set_speed_angle(activities[0][0], activities[0][1])
      
    if rc.controller.was_pressed(rc.controller.Button.X):
        print("Driving in a 8 (eight)...")
        # DONE (main challenge): Drive in a figure eight when the X button is pressed
        activities = [[1, 0, 3.6], [1, 1, 3.35], [1, 0, 2.8], [1, -1, 3.20], [-1, 0, 0.5]]
        rc.drive.set_speed_angle(activities[0][0], activities[0][1])

    if rc.controller.was_pressed(rc.controller.Button.Y):
        print("Driving in a Γ ...")
        # DONE (main challenge): Drive in a shape of your choice when the Y button is pressed #
        activities = [[1, 0, 2],[1, 1, 1.25]]
        rc.drive.set_speed_angle(activities[0][0], activities[0][1])
        
    if activities:
       activities[0][2] -= rc.get_delta_time()
       if activities[0][2] > 0:
           return
       activities.pop(0)
       if activities:
            rc.drive.set_speed_angle(activities[0][0], activities[0][1])
            return
    rc.drive.stop()    

        
########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
