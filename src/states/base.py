from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from src.ui.menu import MenuDefinition, MenuManager
from src.ui.input_handler import InputHandler

if TYPE_CHECKING:
    from src.core.vision import VisionEngine
    from src.core.controller import Controller
    from src.core.manager import StateManager

class State(ABC):
    """
    Base class for all game states.
    
    Each state defines its own logic for detection and execution of instructions.
    Supports sub-states (hierarchical state management) and menu systems.
    """
    
    def __init__(self, manager: 'StateManager'):
        """
        Initialize the state with a reference to the manager.
        
        :param manager: The StateManager instance.
        """
        self.manager = manager
        self.vision = manager.vision
        self.controller = manager.controller
        
        # Sub-state support
        self._sub_state: Optional['State'] = None
        
        # Menu and input handler support
        self.menu_manager: Optional[MenuManager] = None
        self.input_handler: Optional[InputHandler] = None

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
        # Attach input listeners if menu/input handler is configured
        self.attach_input_listeners()
    
    def on_exit(self):
        """Called when exiting the state. Override if needed."""
        # Detach input listeners
        self.detach_input_listeners()
        
        # Exit sub-state if active
        if self._sub_state:
            self._sub_state.on_exit()
            self._sub_state = None
    
    # Sub-state management
    
    def set_sub_state(self, sub_state: 'State'):
        """
        Set a sub-state for this state.
        
        Sub-states are managed hierarchically - the parent state manages
        the sub-state lifecycle. Sub-states can override parent menu items.
        
        :param sub_state: The sub-state to set
        """
        # Exit current sub-state if any
        if self._sub_state:
            self._sub_state.on_exit()
        
        self._sub_state = sub_state
        sub_state.manager = self.manager
        sub_state.vision = self.vision
        sub_state.controller = self.controller
        
        # Enter the sub-state
        sub_state.on_enter()
    
    def clear_sub_state(self):
        """Clear the current sub-state."""
        if self._sub_state:
            self._sub_state.on_exit()
            self._sub_state = None
    
    def get_active_sub_state(self) -> Optional['State']:
        """
        Get the active sub-state (recursive).
        
        Returns the most deeply nested active sub-state.
        
        :return: The active sub-state, or None if no sub-state
        """
        if self._sub_state:
            # Check if sub-state has its own sub-state (recursive)
            nested = self._sub_state.get_active_sub_state()
            return nested if nested else self._sub_state
        return None
    
    @property
    def sub_state(self) -> Optional['State']:
        """Get the current sub-state."""
        return self._sub_state
    
    # Menu and input handling
    
    def get_menu(self) -> Optional[MenuDefinition]:
        """
        Get the menu definition for this state.
        
        Override this method to define the menu for the state.
        
        :return: MenuDefinition instance, or None if no menu
        """
        return None
    
    def attach_input_listeners(self):
        """
        Attach keyboard/controller listeners for this state.
        
        This is called automatically in on_enter(). Override if you need
        custom listener attachment logic, but be sure to call super().
        """
        # Setup menu if defined
        menu = self.get_menu()
        if menu and self.menu_manager:
            self.menu_manager.attach_menu(menu)
            self.menu_manager.print_menu()
        
        # Enable input handler if available
        if self.input_handler:
            self.input_handler.set_listening(True)
    
    def detach_input_listeners(self):
        """
        Detach keyboard/controller listeners for this state.
        
        This is called automatically in on_exit(). Override if you need
        custom listener detachment logic, but be sure to call super().
        """
        # Detach menu
        if self.menu_manager:
            self.menu_manager.detach_menu()
        
        # Disable input handler
        if self.input_handler:
            self.input_handler.set_listening(False)
