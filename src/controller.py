from pynput.keyboard import Controller as KController, Key, KeyCode
import time
import pygame
import vgamepad as vg
from typing import Set, Union, Optional
import os

class Controller:
    """
    Controller handles keyboard and gamepad input emulation using vgamepad.
    """
    
    def __init__(self):
        self.keyboard = KController()
        self.active_keys: Set[Union[str, Key, KeyCode]] = set()
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        self.gamepad: Optional[pygame.joystick.Joystick] = None
        self.active_gamepad_buttons: Set[int] = set()
        self.current_joystick_state = {
            "rx": 0.0, "ry": 0.0
        }
        
        # Try to initialize the first available gamepad
        gamepad_name = None
        if pygame.joystick.get_count() > 0:
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            gamepad_name = self.gamepad.get_name()
            print(f"Gamepad detected: {gamepad_name}")
        else:
            print("No gamepad detected. Keyboard-only mode.")
        
        # Initialize virtual gamepad
        try:
            self.virtual_gamepad = vg.VX360Gamepad()
            
            # Reset to default state
            self.virtual_gamepad.reset()
            self.virtual_gamepad.update()
            
            print("Virtual input device initialized: vgamepad VX360Gamepad")
            
        except Exception as e:
            print(f"Warning: Could not initialize virtual input device: {e}")
            self.virtual_gamepad = None
        
        # Maps
        self.gamepad_button_map = {
            "a": vg.XUSB_BUTTON.XUSB_GAMEPAD_A,     # A button -> BTN_SOUTH
            "b": vg.XUSB_BUTTON.XUSB_GAMEPAD_B,     # B button -> BTN_EAST
            "x": vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,     # X button -> BTN_NORTH (Y on Xbox)
            "y": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,     # Y button -> BTN_WEST (X on Xbox)
        }
        self.numpad_map = {
            "numpad_2": 80, "numpad_4": 75, "numpad_6": 77, "numpad_8": 72,
        }

    def reset_joysticks(self):
        """Forces all analog sticks to center position."""
        if not self.virtual_gamepad: return
        try:
            self.virtual_gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.virtual_gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            self.virtual_gamepad.update()
        except Exception as e:
            print(f"Error resetting joysticks: {e}")

    def press(self, key: Union[str, Key, KeyCode]):
        button_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                button_code = self.gamepad_button_map.get(name)
            # numpad is handled by keyboard controller usually, but let's check legacy map
            # The legacy numpad_map had integers (scancodes?), but pynput takes Keys.
            # However, looking at original code, numpad_map values were 80, 75 etc.
            # And `evdev_code = self.numpad_map.get(key)` was used.
            # But the new implementation shouldn't mix vgamepad and evdev.
            # If it's numpad, it should probably be keyboard press unless mapped to gamepad.
            
        if button_code and self.virtual_gamepad:
            if button_code not in self.active_gamepad_buttons:
                try:
                    self.virtual_gamepad.press_button(button=button_code)
                    self.virtual_gamepad.update()
                    self.active_gamepad_buttons.add(button_code)
                except Exception as e:
                    print(f"Error pressing {key}: {e}")
            return
        
        if key not in self.active_keys:
            try:
                if isinstance(key, str) and key == '1': return
                self.keyboard.press(key)
                self.active_keys.add(key)
            except Exception as e:
                print(f"Error pressing key {key}: {e}")

    def release(self, key: Union[str, Key, KeyCode]):
        button_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                button_code = self.gamepad_button_map.get(name)

        if button_code and self.virtual_gamepad:
            if button_code in self.active_gamepad_buttons:
                try:
                    self.virtual_gamepad.release_button(button=button_code)
                    self.virtual_gamepad.update()
                    self.active_gamepad_buttons.remove(button_code)
                except Exception as e:
                    print(f"Error releasing {key}: {e}")
            return
        
        if key in self.active_keys:
            try:
                self.keyboard.release(key)
                self.active_keys.remove(key)
            except Exception as e:
                print(f"Error releasing key {key}: {e}")

    def tap(self, key: Union[str, Key, KeyCode], duration: float = 0.05):
        # print(f"Tapping {key}")
        button_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                button_code = self.gamepad_button_map.get(name)

        if button_code and self.virtual_gamepad:
            try:
                self.virtual_gamepad.press_button(button=button_code)
                self.virtual_gamepad.update()
                self.active_gamepad_buttons.add(button_code)
                time.sleep(duration)
                self.virtual_gamepad.release_button(button=button_code)
                self.virtual_gamepad.update()
                self.active_gamepad_buttons.discard(button_code)
            except Exception as e:
                print(f"Error tapping {key}: {e}")
            return
        
        try:
            if isinstance(key, str) and key == '1': return
            self.keyboard.press(key)
            time.sleep(duration)
            self.keyboard.release(key)
        except Exception as e:
            print(f"Error tapping key {key}: {e}")

    def release_all(self):
        for key in list(self.active_keys):
            self.release(key)
        
        if self.virtual_gamepad:
            try:
                # Release buttons
                for code in list(self.active_gamepad_buttons):
                    self.virtual_gamepad.release_button(button=code)
                
                self.reset_joysticks()
                
                self.virtual_gamepad.update()
                self.active_gamepad_buttons.clear()
            except Exception as e:
                print(f"Error releasing buttons: {e}")

    def move_camera(self, x: float, y: float):
        """
        Moves the camera in FFXIII.
        
        :param x: -1.0 (Left) to 1.0 (Right) - horizontal pan
        :param y: -1.0 (Up) to 1.0 (Down) - vertical tilt
        """
        if not self.virtual_gamepad: return

        print(f"Moving camera: x={x}, y={y}")
        
        # Standard Xbox 360 mapping:
        # Right Stick X: Horizontal
        # Right Stick Y: Vertical
        # Previous implementation used weird axis mapping (RX for Y input, RY for X input).
        # We will try standard mapping first.
        
        try:
            self.virtual_gamepad.right_joystick_float(x_value_float=x, y_value_float=y)
            self.virtual_gamepad.update()
        except Exception as e:
            print(f"Error moving camera: {e}")

    def move_character(self, x: float, y: float):
        """
        Moves the character using the left joystick.
        
        :param x: -1.0 (Left) to 1.0 (Right)
        :param y: -1.0 (Backward) to 1.0 (Forward) - Note: standard gamepad Y is inverted (up is negative), 
                  but user requested 1.0 = forward. We will map accordingly.
        """
        if not self.virtual_gamepad: return

        # Standard Xbox 360 mapping:
        # Left Stick Y: -1.0 is Up (Forward), 1.0 is Down (Backward)
        # User requested: 1.0 is Forward, -1.0 is Backward
        # So we need to invert the Y input from the user request.
        gamepad_y = -y 
        
        try:
            self.virtual_gamepad.left_joystick_float(x_value_float=x, y_value_float=gamepad_y)
            self.virtual_gamepad.update()
        except Exception as e:
            print(f"Error moving character: {e}")
