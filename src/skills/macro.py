"""
Macro skills for executing sequences of actions.
"""
from typing import List, Dict, Any, Callable, Optional
import time
from .base import Skill


class MacroSkill(Skill):
    """
    Skill for executing sequences of actions (macros).
    
    Stores action sequences and executes them in order.
    Supports conditional execution and delays.
    """
    
    def __init__(self, controller, name: str = "Macro Skill"):
        """
        Initialize macro skill.
        
        Args:
            controller: Controller instance for input
            name: Name of the macro skill
        """
        super().__init__(name, "Executes sequences of actions")
        self.controller = controller
        self._sequence: List[Dict[str, Any]] = []
        self._current_step: int = 0
        self._is_running: bool = False
    
    def add_action(self, action: Dict[str, Any]):
        """
        Add an action to the macro sequence.
        
        Action format:
        {
            'type': 'tap' | 'press' | 'release' | 'delay' | 'move' | 'camera',
            'key': key (for tap/press/release),
            'duration': duration (for tap/delay),
            'x': x value (for move/camera),
            'y': y value (for move/camera),
            'condition': optional callable that returns bool
        }
        
        Args:
            action: Action dictionary
        """
        self._sequence.append(action)
    
    def set_sequence(self, sequence: List[Dict[str, Any]]):
        """
        Set the entire macro sequence.
        
        Args:
            sequence: List of action dictionaries
        """
        self._sequence = sequence
        self._current_step = 0
    
    def start(self):
        """Start executing the macro."""
        if not self._sequence:
            return
        self._is_running = True
        self._current_step = 0
    
    def stop(self):
        """Stop executing the macro."""
        self._is_running = False
        self._current_step = 0
    
    def execute(self, context: dict) -> None:
        """
        Execute the macro (call this repeatedly until complete).
        
        Context should contain:
        - 'start': Optional boolean to start the macro
        - 'stop': Optional boolean to stop the macro
        """
        if context.get('stop', False):
            self.stop()
            return
        
        if context.get('start', False):
            self.start()
        
        if not self._is_running:
            return
        
        if self._current_step >= len(self._sequence):
            self.stop()
            return
        
        action = self._sequence[self._current_step]
        
        # Check condition if present
        condition = action.get('condition')
        if condition and not condition():
            # Skip this action if condition fails
            self._current_step += 1
            return
        
        # Execute action based on type
        action_type = action.get('type')
        
        if action_type == 'tap':
            key = action.get('key')
            duration = action.get('duration', 0.05)
            if key:
                self.controller.tap(key, duration)
            self._current_step += 1
        
        elif action_type == 'press':
            key = action.get('key')
            if key:
                self.controller.press(key)
            self._current_step += 1
        
        elif action_type == 'release':
            key = action.get('key')
            if key:
                self.controller.release(key)
            self._current_step += 1
        
        elif action_type == 'delay':
            duration = action.get('duration', 0.1)
            time.sleep(duration)
            self._current_step += 1
        
        elif action_type == 'move':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            self.controller.move_character(x, y)
            self._current_step += 1
        
        elif action_type == 'camera':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            self.controller.move_camera(x, y)
            self._current_step += 1
        
        else:
            # Unknown action type, skip
            self._current_step += 1
    
    def is_running(self) -> bool:
        """Check if macro is currently running."""
        return self._is_running
    
    def is_complete(self) -> bool:
        """Check if macro has completed."""
        return not self._is_running and self._current_step >= len(self._sequence)
    
    def reset(self):
        """Reset the macro to the beginning."""
        self._current_step = 0
        self._is_running = False
