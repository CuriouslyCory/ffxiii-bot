"""
Movement skills for character and camera control.
"""
from typing import Optional
from .base import Skill


class MovementSkill(Skill):
    """
    Skill for handling character movement with state management.
    
    Maintains current movement vector and provides methods to move
    and stop the character.
    """
    
    def __init__(self, controller):
        """
        Initialize movement skill.
        
        Args:
            controller: Controller instance for gamepad input
        """
        super().__init__("Movement Skill", "Handles character movement")
        self.controller = controller
        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._active: bool = False
    
    def move(self, x: float, y: float):
        """
        Move the character with the given vector.
        
        Args:
            x: X movement (-1.0 to 1.0, negative=left, positive=right)
            y: Y movement (-1.0 to 1.0, negative=backward, positive=forward)
        """
        self._current_x = max(-1.0, min(1.0, x))
        self._current_y = max(-1.0, min(1.0, y))
        self._active = True
        self.controller.move_character(self._current_x, self._current_y)
    
    def stop(self):
        """Stop character movement."""
        self._current_x = 0.0
        self._current_y = 0.0
        self._active = False
        self.controller.move_character(0.0, 0.0)
    
    def execute(self, context: dict) -> None:
        """
        Execute movement from context.
        
        Context should contain:
        - 'x': X movement value (float)
        - 'y': Y movement value (float)
        - 'stop': Optional boolean to stop movement
        """
        if context.get('stop', False):
            self.stop()
        else:
            x = context.get('x', 0.0)
            y = context.get('y', 0.0)
            self.move(x, y)
    
    @property
    def current_x(self) -> float:
        """Get current X movement value."""
        return self._current_x
    
    @property
    def current_y(self) -> float:
        """Get current Y movement value."""
        return self._current_y
    
    @property
    def is_active(self) -> bool:
        """Check if movement is currently active."""
        return self._active


class CameraSkill(Skill):
    """
    Skill for handling camera movement.
    """
    
    def __init__(self, controller):
        """
        Initialize camera skill.
        
        Args:
            controller: Controller instance for gamepad input
        """
        super().__init__("Camera Skill", "Handles camera movement")
        self.controller = controller
        self._current_x: float = 0.0
        self._current_y: float = 0.0
        self._active: bool = False
    
    def move(self, x: float, y: float):
        """
        Move the camera with the given vector.
        
        Args:
            x: X camera movement (-1.0 to 1.0)
            y: Y camera movement (-1.0 to 1.0)
        """
        self._current_x = max(-1.0, min(1.0, x))
        self._current_y = max(-1.0, min(1.0, y))
        self._active = True
        self.controller.move_camera(self._current_x, self._current_y)
    
    def stop(self):
        """Stop camera movement."""
        self._current_x = 0.0
        self._current_y = 0.0
        self._active = False
        self.controller.move_camera(0.0, 0.0)
    
    def execute(self, context: dict) -> None:
        """
        Execute camera movement from context.
        
        Context should contain:
        - 'x': X camera movement value (float)
        - 'y': Y camera movement value (float)
        - 'stop': Optional boolean to stop camera
        """
        if context.get('stop', False):
            self.stop()
        else:
            x = context.get('x', 0.0)
            y = context.get('y', 0.0)
            self.move(x, y)
    
    @property
    def current_x(self) -> float:
        """Get current X camera movement value."""
        return self._current_x
    
    @property
    def current_y(self) -> float:
        """Get current Y camera movement value."""
        return self._current_y
    
    @property
    def is_active(self) -> bool:
        """Check if camera movement is currently active."""
        return self._active
