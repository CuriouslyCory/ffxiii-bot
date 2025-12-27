from pynput.keyboard import Controller as KController, Key
import time
from typing import Set, Union

class Controller:
    """
    Controller handles keyboard input emulation.
    
    It provides methods for pressing, releasing, and tapping keys,
    maintaining compatibility with game input processing.
    """
    
    def __init__(self):
        """Initializes the keyboard controller."""
        self.keyboard = KController()
        self.active_keys: Set[Union[str, Key]] = set()

    def press(self, key: Union[str, Key]):
        """
        Presses and holds a key.
        
        :param key: The character (e.g., 'w') or Key object (e.g., Key.space) to press.
        """
        if key not in self.active_keys:
            try:
                self.keyboard.press(key)
                self.active_keys.add(key)
            except Exception as e:
                print(f"Error pressing key {key}: {e}")

    def release(self, key: Union[str, Key]):
        """
        Releases a held key.
        
        :param key: The character or Key object to release.
        """
        if key in self.active_keys:
            try:
                self.keyboard.release(key)
                self.active_keys.remove(key)
            except Exception as e:
                print(f"Error releasing key {key}: {e}")

    def tap(self, key: Union[str, Key], duration: float = 0.05):
        """
        Taps a key (press and release).
        
        :param key: The character or Key object to tap.
        :param duration: Time in seconds to hold the key down.
        """
        try:
            self.keyboard.press(key)
            time.sleep(duration)
            self.keyboard.release(key)
        except Exception as e:
            print(f"Error tapping key {key}: {e}")

    def release_all(self):
        """Releases all currently active keys."""
        for key in list(self.active_keys):
            self.release(key)

