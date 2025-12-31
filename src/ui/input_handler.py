"""
Generalized input handler for processing keyboard input.
"""
from typing import Optional, Set, Dict, Callable
from pynput import keyboard


class InputHandler:
    """
    Generalized keyboard input handler that supports dynamic key bindings.
    
    Provides thread-safe action queue for communication between keyboard listener
    and main execution thread. Supports per-state key bindings and sub-state
    key binding layering.
    """
    
    def __init__(self, on_action_callback: Optional[Callable] = None):
        """
        Initialize the input handler.
        
        Args:
            on_action_callback: Optional callback function called when actions are triggered.
        """
        self.listening = False
        self.blocking_input = False
        self.on_action_callback = on_action_callback
        
        # Dynamic key bindings: key -> action identifier (string or enum)
        self._key_bindings: Dict[str, str] = {}
        
        # Pending actions (set to avoid duplicates)
        self._pending_actions: Set[str] = set()
        
        # Additional data storage for actions (e.g., route selection number)
        self._action_data: Dict[str, any] = {}
        
        # Setup keyboard listener (non-blocking)
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()
    
    def set_key_bindings(self, bindings: Dict[str, str]):
        """
        Set key bindings for the current state.
        
        Args:
            bindings: Dictionary mapping key characters to action identifiers
                     Example: {'r': 'RECORD_MODE', 'p': 'PLAYBACK'}
        """
        self._key_bindings = bindings.copy()
    
    def add_key_bindings(self, bindings: Dict[str, str]):
        """
        Add key bindings to existing bindings (for sub-state layering).
        
        Args:
            bindings: Dictionary mapping key characters to action identifiers
        """
        self._key_bindings.update(bindings)
    
    def clear_key_bindings(self):
        """Clear all key bindings."""
        self._key_bindings = {}
    
    def _on_press(self, key):
        """Internal keyboard press handler."""
        if not self.listening or self.blocking_input:
            return
        
        try:
            # Handle escape key (always bound to STOP)
            if key == keyboard.Key.esc:
                self._add_action('STOP')
                return
            
            # Get character if available
            k_char = None
            if hasattr(key, 'char') and key.char:
                k_char = key.char.lower()
            
            if k_char and k_char in self._key_bindings:
                action = self._key_bindings[k_char]
                self._add_action(action, data={'key': k_char})
            
        except AttributeError:
            pass
    
    def _add_action(self, action: str, data: Optional[Dict] = None):
        """
        Add an action to the pending set.
        
        Args:
            action: Action identifier
            data: Optional additional data for the action
        """
        self._pending_actions.add(action)
        if data:
            if action not in self._action_data:
                self._action_data[action] = {}
            self._action_data[action].update(data)
        
        if self.on_action_callback:
            self.on_action_callback(action)
    
    def get_pending_actions(self) -> Set[str]:
        """
        Get all pending actions and clear the queue.
        
        Returns:
            Set of pending action identifiers
        """
        actions = self._pending_actions.copy()
        self._pending_actions.clear()
        return actions
    
    def get_action_data(self, action: str) -> Optional[Dict]:
        """
        Get additional data for an action and clear it.
        
        Args:
            action: Action identifier
            
        Returns:
            Action data dictionary, or None if no data
        """
        data = self._action_data.pop(action, None)
        return data
    
    def has_action(self, action: str) -> bool:
        """
        Check if a specific action is pending.
        
        Args:
            action: Action identifier
            
        Returns:
            True if action is pending
        """
        return action in self._pending_actions
    
    def clear_action(self, action: str):
        """
        Remove a specific action from pending set.
        
        Args:
            action: Action identifier
        """
        self._pending_actions.discard(action)
        self._action_data.pop(action, None)
    
    def set_listening(self, listening: bool):
        """
        Enable or disable input listening.
        
        Args:
            listening: True to enable, False to disable
        """
        self.listening = listening
    
    def set_blocking(self, blocking: bool):
        """
        Enable or disable blocking input (for dialogs).
        
        Args:
            blocking: True to block input, False to allow
        """
        self.blocking_input = blocking
    
    def stop(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()
