"""
Button interaction skills for pressing and tapping buttons.
"""
import time
from typing import Union
from pynput.keyboard import Key, KeyCode
from .base import Skill


class ButtonPressSkill(Skill):
    """
    Skill for pressing and releasing buttons.
    
    Maintains state of currently pressed buttons.
    """
    
    def __init__(self, controller):
        """
        Initialize button press skill.
        
        Args:
            controller: Controller instance for input
        """
        super().__init__("Button Press Skill", "Presses and releases buttons")
        self.controller = controller
        self._pressed_keys: set = set()
    
    def press(self, key: Union[str, Key, KeyCode]):
        """
        Press a button/key.
        
        Args:
            key: Key to press (string, Key, or KeyCode)
        """
        if key not in self._pressed_keys:
            self.controller.press(key)
            self._pressed_keys.add(key)
    
    def release(self, key: Union[str, Key, KeyCode]):
        """
        Release a button/key.
        
        Args:
            key: Key to release (string, Key, or KeyCode)
        """
        if key in self._pressed_keys:
            self.controller.release(key)
            self._pressed_keys.discard(key)
    
    def release_all(self):
        """Release all currently pressed keys."""
        for key in list(self._pressed_keys):
            self.release(key)
    
    def execute(self, context: dict) -> None:
        """
        Execute button press/release from context.
        
        Context should contain:
        - 'action': 'press' or 'release'
        - 'key': Key to press/release
        - 'release_all': Optional boolean to release all keys
        """
        if context.get('release_all', False):
            self.release_all()
        else:
            action = context.get('action', 'press')
            key = context.get('key')
            if key:
                if action == 'press':
                    self.press(key)
                elif action == 'release':
                    self.release(key)


class ButtonTapSkill(Skill):
    """
    Skill for tapping buttons (press and release quickly).
    """
    
    def __init__(self, controller):
        """
        Initialize button tap skill.
        
        Args:
            controller: Controller instance for input
        """
        super().__init__("Button Tap Skill", "Taps buttons with duration")
        self.controller = controller
    
    def tap(self, key: Union[str, Key, KeyCode], duration: float = 0.05):
        """
        Tap a button (press and release with duration).
        
        Args:
            key: Key to tap
            duration: Duration in seconds to hold the key
        """
        self.controller.tap(key, duration)
    
    def execute(self, context: dict) -> None:
        """
        Execute button tap from context.
        
        Context should contain:
        - 'key': Key to tap
        - 'duration': Optional duration in seconds (default 0.05)
        """
        key = context.get('key')
        duration = context.get('duration', 0.05)
        if key:
            self.tap(key, duration)
