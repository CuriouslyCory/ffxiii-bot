#!/usr/bin/env python3
"""
Debug utility for displaying gamepad button presses.

This script helps identify which button indices correspond to which physical
buttons on your gamepad controller.

Usage:
    python src/debug-controller.py
"""

import pygame
import sys
from evdev import ecodes
from typing import Dict

# Button name mappings for Nintendo Switch controller
# Maps pygame button indices to physical button names
BUTTON_NAMES: Dict[int, str] = {
    # Nintendo Switch controller mapping
    0: "B / Cross",      # B button (south)
    1: "A / Circle",      # A button (east)
    2: "X / Square",      # X button (north)
    3: "Y / Triangle",    # Y button (west)
    4: "Left Bumper",
    5: "Right Bumper",
    6: "Back / Select",
    7: "Start",
    8: "Left Stick Press",
    9: "Right Stick Press",
}

# Evdev code mappings matching controller.py's gamepad_button_map exactly
# Maps button names to evdev key codes (Nintendo Switch: B=south, A=east, X=north, Y=west)
EVDEV_BUTTON_MAP = {
    "a": ecodes.BTN_SOUTH,  # A button -> BTN_SOUTH (B button)
    "b": ecodes.BTN_EAST,   # B button -> BTN_EAST (A button)
    "x": ecodes.BTN_NORTH,  # X button -> BTN_NORTH
    "y": ecodes.BTN_WEST,   # Y button -> BTN_WEST
}

# Mapping from pygame button index to controller button name
# Nintendo Switch: pygame 0=B, 1=A, 2=X, 3=Y
PYGAME_TO_CONTROLLER_BUTTON: Dict[int, str] = {
    0: "b",  # Pygame button 0 = B button
    1: "a",  # Pygame button 1 = A button
    2: "x",  # Pygame button 2 = X button
    3: "y",  # Pygame button 3 = Y button
}

# Reverse mapping: evdev code -> button name
EVDEV_CODE_TO_NAME = {v: k for k, v in EVDEV_BUTTON_MAP.items()}

def main():
    """
    Main function that initializes pygame and monitors gamepad inputs.
    
    Displays button presses, releases, and axis movements in real-time.
    """
    print("=" * 60)
    print("Gamepad Debug Utility")
    print("=" * 60)
    print("\nThis utility will display all gamepad button presses and axis movements.")
    print("Press Ctrl+C to exit.\n")
    
    # Initialize pygame
    pygame.init()
    pygame.joystick.init()
    
    # Check for connected gamepads
    joystick_count = pygame.joystick.get_count()
    
    if joystick_count == 0:
        print("ERROR: No gamepad detected!")
        print("Please connect a gamepad and try again.")
        sys.exit(1)
    
    print(f"Found {joystick_count} gamepad(s):\n")
    
    # Initialize all connected gamepads
    joysticks = []
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        joysticks.append(joystick)
        print(f"  [{i}] {joystick.get_name()}")
        print(f"      Buttons: {joystick.get_numbuttons()}")
        print(f"      Axes: {joystick.get_numaxes()}")
        print(f"      Hats: {joystick.get_numhats()}")
        print()
    
    # Use the first gamepad
    if joystick_count > 1:
        print(f"Monitoring gamepad 0: {joysticks[0].get_name()}\n")
    else:
        print(f"Monitoring: {joysticks[0].get_name()}\n")
    
    print("-" * 60)
    print("Press buttons on your gamepad to see their mappings.")
    print("Button indices will be displayed when pressed/released.")
    print()
    print("Current controller.py button mappings:")
    for btn_name, evdev_code in EVDEV_BUTTON_MAP.items():
        evdev_code_name = {
            ecodes.BTN_SOUTH: "BTN_SOUTH",
            ecodes.BTN_EAST: "BTN_EAST",
            ecodes.BTN_NORTH: "BTN_NORTH",
            ecodes.BTN_WEST: "BTN_WEST",
        }.get(evdev_code, f"evdev_code_{evdev_code}")
        print(f"  'gamepad_{btn_name}' -> {evdev_code_name} ({evdev_code})")
    print("-" * 60)
    print()
    
    # Track button states to detect changes
    previous_button_states = {}
    for joystick in joysticks:
        previous_button_states[joystick] = {}
        for i in range(joystick.get_numbuttons()):
            previous_button_states[joystick][i] = False
    
    # Track axis states
    previous_axis_states = {}
    for joystick in joysticks:
        previous_axis_states[joystick] = {}
        for i in range(joystick.get_numaxes()):
            previous_axis_states[joystick][i] = 0.0
    
    try:
        while True:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                # Handle button presses/releases
                if event.type == pygame.JOYBUTTONDOWN:
                    joystick = joysticks[event.joy]
                    button_id = event.button
                    button_name = BUTTON_NAMES.get(button_id, "Unknown")
                    
                    # Map pygame button index to controller button name
                    controller_button = PYGAME_TO_CONTROLLER_BUTTON.get(button_id)
                    
                    # Check if this button maps to an evdev code in controller.py
                    evdev_info = ""
                    if controller_button and controller_button in EVDEV_BUTTON_MAP:
                        evdev_code = EVDEV_BUTTON_MAP[controller_button]
                        evdev_code_name = {
                            ecodes.BTN_SOUTH: "BTN_SOUTH",
                            ecodes.BTN_EAST: "BTN_EAST",
                            ecodes.BTN_NORTH: "BTN_NORTH",
                            ecodes.BTN_WEST: "BTN_WEST",
                        }.get(evdev_code, f"evdev_code_{evdev_code}")
                        evdev_info = f" -> evdev: {evdev_code_name} ({evdev_code})"
                    
                    controller_mapping = f"'gamepad_{controller_button}'" if controller_button else "Not mapped"
                    print(f"[BUTTON PRESS]   Gamepad {event.joy}, Pygame Button {button_id:2d} ({button_name})")
                    print(f"                 Controller mapping: {controller_mapping}{evdev_info}")
                    
                elif event.type == pygame.JOYBUTTONUP:
                    joystick = joysticks[event.joy]
                    button_id = event.button
                    button_name = BUTTON_NAMES.get(button_id, "Unknown")
                    
                    # Map pygame button index to controller button name
                    controller_button = PYGAME_TO_CONTROLLER_BUTTON.get(button_id)
                    
                    evdev_info = ""
                    if controller_button and controller_button in EVDEV_BUTTON_MAP:
                        evdev_code = EVDEV_BUTTON_MAP[controller_button]
                        evdev_code_name = {
                            ecodes.BTN_SOUTH: "BTN_SOUTH",
                            ecodes.BTN_EAST: "BTN_EAST",
                            ecodes.BTN_NORTH: "BTN_NORTH",
                            ecodes.BTN_WEST: "BTN_WEST",
                        }.get(evdev_code, f"evdev_code_{evdev_code}")
                        evdev_info = f" -> evdev: {evdev_code_name} ({evdev_code})"
                    
                    controller_mapping = f"'gamepad_{controller_button}'" if controller_button else "Not mapped"
                    print(f"[BUTTON RELEASE] Gamepad {event.joy}, Pygame Button {button_id:2d} ({button_name})")
                    print(f"                 Controller mapping: {controller_mapping}{evdev_info}")
                
                # Handle axis movements (analog sticks, triggers)
                elif event.type == pygame.JOYAXISMOTION:
                    joystick = joysticks[event.joy]
                    axis_id = event.axis
                    value = event.value
                    
                    # Only print if value changed significantly (avoid spam)
                    prev_value = previous_axis_states[joystick].get(axis_id, 0.0)
                    if abs(value - prev_value) > 0.1:
                        axis_name = "Left Stick" if axis_id < 2 else "Right Stick" if axis_id < 4 else "Trigger/Other"
                        direction = ""
                        if axis_id % 2 == 0:  # X axis
                            direction = "X (Left/Right)"
                        else:  # Y axis
                            direction = "Y (Up/Down)"
                        
                        print(f"[AXIS]          Gamepad {event.joy}, Axis {axis_id} ({axis_name} {direction}): {value:6.3f}")
                        previous_axis_states[joystick][axis_id] = value
                
                # Handle hat (D-pad) movements
                elif event.type == pygame.JOYHATMOTION:
                    joystick = joysticks[event.joy]
                    hat_id = event.hat
                    value = event.value
                    
                    if value != (0, 0):  # Only print non-neutral positions
                        print(f"[HAT/DPAD]      Gamepad {event.joy}, Hat {hat_id}: {value}")
            
            # Small delay to prevent excessive CPU usage
            pygame.time.wait(10)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()

