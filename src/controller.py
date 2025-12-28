from pynput.keyboard import Controller as KController, Key, KeyCode
import time
import pygame
from evdev import UInput, ecodes, InputDevice, list_devices, AbsInfo
from typing import Set, Union, Optional
import os

class Controller:
    """
    Controller handles keyboard and gamepad input emulation.
    """
    
    def __init__(self):
        self.keyboard = KController()
        self.active_keys: Set[Union[str, Key, KeyCode]] = set()
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        self.gamepad: Optional[pygame.joystick.Joystick] = None
        self.physical_gamepad_device: Optional[InputDevice] = None
        self.uinput_device: Optional[UInput] = None
        self.active_gamepad_buttons: Set[int] = set()
        self.current_joystick_state = {
            "rx": 0, "ry": 0
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
        
        # Find physical device
        if gamepad_name:
            try:
                devices = [InputDevice(path) for path in list_devices()]
                for device in devices:
                    name_lower = device.name.lower()
                    if "nintendo" in name_lower or "switch" in name_lower or \
                       "gamepad" in name_lower or "joystick" in name_lower:
                        self.physical_gamepad_device = device
                        break
            except Exception:
                pass
        
        # Initialize uinput with Axes
        try:
            capabilities = {
                ecodes.EV_KEY: [
                    ecodes.BTN_SOUTH, ecodes.BTN_EAST, ecodes.BTN_NORTH, ecodes.BTN_WEST,
                    ecodes.KEY_KP2, ecodes.KEY_KP4, ecodes.KEY_KP6, ecodes.KEY_KP8
                ],
                ecodes.EV_ABS: [
                    (ecodes.ABS_X,  AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
                    (ecodes.ABS_Y,  AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
                    (ecodes.ABS_RX, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
                    (ecodes.ABS_RY, AbsInfo(value=128, min=0, max=255, fuzz=0, flat=15, resolution=0)),
                ]
            }
            
            device_name = gamepad_name if gamepad_name else "ffxiii-bot-virtual-gamepad"
            self.uinput_device = UInput(capabilities, name=device_name, vendor=0x045e, product=0x028e)
            
            # CRITICAL: Write center values IMMEDIATELY after creation
            # This prevents axes from defaulting to 0 (up+left) before the OS registers the device
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_X, 128)
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_Y, 128)
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RX, 128)
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RY, 128)
            self.uinput_device.syn()
            
            # Let the OS register the device
            time.sleep(0.5)
            
            # Reset joysticks to center again after registration
            self.reset_joysticks()
            
            print(f"Virtual input device initialized: {device_name}")
            
        except Exception as e:
            print(f"Warning: Could not initialize virtual input device: {e}")
        
        # Maps
        self.gamepad_button_map = {
            "a": ecodes.BTN_SOUTH,
            "b": ecodes.BTN_EAST,
            "x": ecodes.BTN_NORTH,
            "y": ecodes.BTN_WEST,
        }
        self.numpad_map = {
            "numpad_2": 80, "numpad_4": 75, "numpad_6": 77, "numpad_8": 72,
        }

    def reset_joysticks(self):
        """Forces all analog sticks to center position (128)."""
        if not self.uinput_device: return
        try:
            # Write multiple times to ensure it registers
            for _ in range(3):
                self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_X, 128)
                self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_Y, 128)
                self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RX, 128)
                self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RY, 128)
                self.uinput_device.syn()
                time.sleep(0.01)
        except Exception as e:
            print(f"Error resetting joysticks: {e}")

    # ... (rest of methods: press, release, tap, release_all, move_camera)
    # Be sure to include them or I will break the file if I overwrite fully.
    # I will write the FULL file content below to be safe.

    def press(self, key: Union[str, Key, KeyCode]):
        evdev_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                evdev_code = self.gamepad_button_map.get(name)
            elif key.startswith("numpad_"):
                evdev_code = self.numpad_map.get(key)
        
        if evdev_code and self.uinput_device:
            if evdev_code not in self.active_gamepad_buttons:
                try:
                    self.uinput_device.write(ecodes.EV_KEY, evdev_code, 1)
                    self.uinput_device.syn()
                    self.active_gamepad_buttons.add(evdev_code)
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
        evdev_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                evdev_code = self.gamepad_button_map.get(name)
            elif key.startswith("numpad_"):
                evdev_code = self.numpad_map.get(key)

        if evdev_code and self.uinput_device:
            if evdev_code in self.active_gamepad_buttons:
                try:
                    self.uinput_device.write(ecodes.EV_KEY, evdev_code, 0)
                    self.uinput_device.syn()
                    self.active_gamepad_buttons.remove(evdev_code)
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
        print(f"Tapping {key}")
        evdev_code = None
        if isinstance(key, str):
            if key.startswith("gamepad_"):
                name = key.replace("gamepad_", "").lower()
                evdev_code = self.gamepad_button_map.get(name)
            elif key.startswith("numpad_"):
                evdev_code = self.numpad_map.get(key)

        if evdev_code and self.uinput_device:
            try:
                self.uinput_device.write(ecodes.EV_KEY, evdev_code, 1)
                self.uinput_device.syn()
                self.active_gamepad_buttons.add(evdev_code)
                time.sleep(duration)
                self.uinput_device.write(ecodes.EV_KEY, evdev_code, 0)
                self.uinput_device.syn()
                self.active_gamepad_buttons.discard(evdev_code)
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
        
        if self.uinput_device:
            try:
                # Release buttons
                for code in list(self.active_gamepad_buttons):
                    self.uinput_device.write(ecodes.EV_KEY, code, 0)
                
                self.reset_joysticks()
                
                self.active_gamepad_buttons.clear()
            except Exception as e:
                print(f"Error releasing buttons: {e}")

    def move_camera(self, x: float, y: float):
        """
        Moves the camera in FFXIII.
        
        FFXIII camera control mapping (discovered through testing):
        - Horizontal pan: ABS_RY (analog right stick Y) - smooth analog control
        - Vertical tilt: Numpad 8 (up) / Numpad 2 (down) - digital only
        
        :param x: -1.0 (Left) to 1.0 (Right) - horizontal pan (analog)
        :param y: -1.0 (Up) to 1.0 (Down) - vertical tilt (uses keyboard numpad)
        """
        if not self.uinput_device: return

        print(f"Moving camera: x={x}, y={y}")
        
        # Horizontal pan - analog via ABS_RY
        val_x = int((x + 1.0) * 127.5)
        val_x = max(0, min(255, val_x))
        val_y = int((y + 1.0) * 127.5)
        val_y = max(0, min(255, val_y))
        
        try:
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RY, val_x)
            self.uinput_device.write(ecodes.EV_ABS, ecodes.ABS_RX, val_y)
            
            # # Vertical tilt - digital via numpad keys
            # # Release both first, then press the appropriate one
            # if abs(y) > 0.1:  # Deadzone for vertical
            #     if y < 0:  # Look up
            #         self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP8, 1)
            #         self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP2, 0)
            #     else:  # Look down
            #         self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP2, 1)
            #         self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP8, 0)
            # else:
            #     # Release both vertical keys when centered
            #     self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP8, 0)
            #     self.uinput_device.write(ecodes.EV_KEY, ecodes.KEY_KP2, 0)
            
            self.uinput_device.syn()
        except Exception as e:
            print(f"Error moving camera: {e}")
