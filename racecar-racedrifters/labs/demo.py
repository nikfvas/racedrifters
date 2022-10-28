"""
Copyright Harvey Mudd College
MIT License
Spring 2020

Demo RACECAR program
"""

########################################################################################
# Imports
########################################################################################

from curses import get_tabsize
from lib2to3.pgen2 import driver
from socket import gaierror
import sys
from tkinter import Button
from turtle import speed

from numpy import angle

sys.path.insert(0, '../library')
import racecar_core

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here
counter = 0
isDriving = False
state = 0
drift_button = 0
activities = []

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

########################################################################################
########################################################################################
########################################################################################

def start():
    """
    This function is run once every time the start button is pressed
    """
    # If we use a global variable in our function, we must list it at
    # the beginning of our function like this
    global counter
    global isDriving
    global speed
    global angle
    global drift

    # The start function is a great place to give initial values to global variables
    counter = 0
    isDriving = False
    drift = 0
    speed = 0
    angle = 0

    # This tells the car to begin at a standstill
    rc.drive.stop()

def update():
  """
  After start() is run, this function is run every frame until the back button
  is pressed
  """
  global counter
  global isDriving
  global speed
  global angle
  global drift
  global state
  global drift_button
  global activities

  forwardSpeed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
  backSpeed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
  speed = forwardSpeed - backSpeed

  # This prints a message every time the A button is pressed on the controller

  if rc.controller.was_pressed(rc.controller.Button.A):
    if drift_button == 0:
      drift_button = 1
      state = 1
      print("drift mode:   ON")
      print('Press B when you want to drift...')
      print('Then press X')
    if drift_button != 1:
      print("drift mode:   OFF")
      state = 0
      speed = 0
      angle = 0
      print('press Y to reset')
    
  if rc.controller.was_pressed(rc.controller.Button.B):
    if drift_button == 1:
      print('drift in process...')
      state = 2
      print('drift COMPLETED')
      print('Press A to exit...')
      drift_button = 2

  if rc.controller.was_pressed(rc.controller.Button.X):
    if drift_button == 2:
      state = 3
      print('drift COMPLETED')
      print('Press A to exit...')

  if rc.controller.was_pressed(rc.controller.Button.Y):
    drift_button = 0
    speed = 0
    angle = 0
    print('Racecar is now resetted')
  

  if state == 0:
    pass

  if state == 1:
    speed = 1
    rc.drive.set_speed_angle(speed, angle)
  
  if state == 2:
    speed = 1
    angle = -1
    rc.drive.set_speed_angle(speed, angle)

  if state == 3:
    speed = 0
    angle = -1
    rc.drive.set_speed_angle(speed, angle)

  # Reset the counter and start driving in an L every time the B button is pressed on
  # the controller

  if isDriving:
    # rc.get_delta_time() gives the time in seconds since the last time
    # the update function was called
    counter += rc.get_delta_time()

    if counter < 1:
      # Drive forward at full speed for one second
      rc.drive.set_speed_angle(1, 0)
    elif counter < 2:
      # Turn left at full speed for the next second
      rc.drive.set_speed_angle(1, 1)
    else:
      # Otherwise, stop the car
      rc.drive.stop()
      isDriving = False

# update_slow() is similar to update() but is called once per second by default.
# It is especially useful for printing debug messages, since printing a message
# every frame in update is computationally expensive and creates clutter

def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # This prints a message every time that the right bumper is pressed during
    # a call to to update_slow.  If we press and hold the right bumper, it
    # will print a message once per second
    if rc.controller.is_down(rc.controller.Button.RB):
        print("The right bumper is currently down (update_slow)")

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
