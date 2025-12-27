from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.vision import VisionEngine
    from src.controller import Controller
    from src.states.manager import StateManager

class State(ABC):
    """
    Base class for all game states.
    
    Each state defines its own logic for detection and execution of instructions.
    """
    
    def __init__(self, manager: 'StateManager'):
        """
        Initializes the state with a reference to the manager.
        
        :param manager: The StateManager instance.
        """
        self.manager = manager
        self.vision = manager.vision
        self.controller = manager.controller

    @abstractmethod
    def is_active(self, image) -> bool:
        """
        Determines if this state is currently active based on vision.
        
        :param image: The current screen capture.
        :return: True if the state is detected, False otherwise.
        """
        pass

    @abstractmethod
    def execute(self, image):
        """
        Executes the state's instructions.
        
        :param image: The current screen capture.
        """
        pass

    def on_enter(self):
        """Called when entering the state. Override if needed."""
        pass

    def on_exit(self):
        """Called when exiting the state. Override if needed."""
        pass

