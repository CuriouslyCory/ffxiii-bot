"""
Input handler for processing keyboard input and converting to action requests.
"""

from typing import Optional, Set
from pynput import keyboard
from enum import Enum


class Action(Enum):
    """Actions that can be triggered by keyboard input."""
    RECORD_MODE = "record_mode"
    CAPTURE = "capture"
    NEXT_STEP = "next_step"
    RETAKE = "retake"
    SAVE_EXIT = "save_exit"
    PLAYBACK = "playback"
    LIST_ROUTES = "list_routes"
    SELECT_ROUTE = "select_route"
    STOP = "stop"
    DELETE_CURRENT_IMAGE = "delete_current_image"
    DELETE_NEXT_IMAGE = "delete_next_image"
    DELETE_NEXT_NODE = "delete_next_node"
    RANDOM_MOVEMENT = "random_movement"


class InputHandler:
    """
    Handles keyboard input processing and converts key presses to action requests.
    
    Provides thread-safe action queue for communication between keyboard listener
    and main execution thread.
    """
    
    def __init__(self, on_action_callback=None):
        """
        Initialize the input handler.
        
        Args:
            on_action_callback: Optional callback function called when actions are triggered.
        """
        self.listening = False
        self.blocking_input = False
        self.on_action_callback = on_action_callback
        
        # Pending actions (set to avoid duplicates)
        self._pending_actions: Set[Action] = set()
        self._select_route_value: Optional[int] = None
        
        # Setup keyboard listener (non-blocking)
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
    
    def _on_press(self, key):
        """Internal keyboard press handler."""
        if not self.listening or self.blocking_input:
            return
        
        try:
            # Handle escape key
            if key == keyboard.Key.esc:
                self._add_action(Action.STOP)
                return
            
            # Get character if available
            k_char = None
            if hasattr(key, 'char') and key.char:
                k_char = key.char.lower()
            elif hasattr(key, 'vk') and key.vk in [87, 88, 89, 83, 84, 85, 79, 80, 81, 90]:
                pass  # Numpad keys
            
            if k_char:
                self._process_char(k_char)
                
        except AttributeError:
            pass
    
    def _process_char(self, k_char: str):
        """Process a character key press."""
        if k_char == 'r':
            self._add_action(Action.RECORD_MODE)
        elif k_char == 't':
            self._add_action(Action.CAPTURE)
        elif k_char == 'n':
            self._add_action(Action.NEXT_STEP)
        elif k_char == 'g':
            self._add_action(Action.RETAKE)
        elif k_char == 'y':
            self._add_action(Action.SAVE_EXIT)
        elif k_char == 'u':
            self._add_action(Action.PLAYBACK)
        elif k_char == 'p':
            self._add_action(Action.LIST_ROUTES)
        elif k_char == '2':
            self._add_action(Action.DELETE_CURRENT_IMAGE)
        elif k_char == '3':
            self._add_action(Action.DELETE_NEXT_IMAGE)
        elif k_char == '4':
            self._add_action(Action.DELETE_NEXT_NODE)
        elif k_char == '0':
            self._add_action(Action.RANDOM_MOVEMENT)
        elif k_char in [str(i) for i in range(1, 10)]:
            self._select_route_value = int(k_char)
            self._add_action(Action.SELECT_ROUTE)
    
    def _add_action(self, action: Action):
        """Add an action to the pending set."""
        self._pending_actions.add(action)
        if self.on_action_callback:
            self.on_action_callback(action)
    
    def get_pending_actions(self) -> Set[Action]:
        """
        Get all pending actions and clear the queue.
        
        Returns:
            Set of pending actions.
        """
        actions = self._pending_actions.copy()
        self._pending_actions.clear()
        return actions
    
    def has_action(self, action: Action) -> bool:
        """Check if a specific action is pending."""
        return action in self._pending_actions
    
    def clear_action(self, action: Action):
        """Remove a specific action from pending set."""
        self._pending_actions.discard(action)
    
    def get_select_route_value(self) -> Optional[int]:
        """
        Get the route selection value (1-9) and clear it.
        
        Returns:
            Route number (1-9) or None if not set.
        """
        value = self._select_route_value
        self._select_route_value = None
        return value
    
    def set_listening(self, listening: bool):
        """Enable or disable input listening."""
        self.listening = listening
    
    def set_blocking(self, blocking: bool):
        """Enable or disable blocking input (for dialogs)."""
        self.blocking_input = blocking
    
    def stop(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()
