import time
from typing import List, Optional, Type
import numpy as np
from src.core.vision import VisionEngine
from src.core.controller import Controller
from src.core.roi_cache import ROICache
from src.states.base import State

class StateManager:
    """
    StateManager coordinates state transitions and execution.
    
    It maintains a list of possible states and checks which one is active.
    """
    
    def __init__(self, vision: VisionEngine, controller: Controller):
        """
        Initializes the StateManager.
        
        :param vision: VisionEngine instance.
        :param controller: Controller instance.
        """
        self.vision = vision
        self.controller = controller
        self.states: List[State] = []
        self.current_state: Optional[State] = None
        
        # Initialize ROI cache
        self.roi_cache = ROICache(vision)
        
        # Register common ROIs
        # Minimap ROI: (x, y, width, height) - no stretching for base extraction
        self.roi_cache.register_roi("minimap", (1375, 57,  425, 320))
        self.roi_cache.register_roi("minimap_center_arrow", (1575, 202, 30, 30))
        
        # Hostile clock ROI: bottom 50% of screen, middle 33% horizontally
        # Use vision engine's resolution: (width, height)
        width, height = vision.resolution
        hostile_clock_width = int(width * 0.33)
        hostile_clock_x = (width - hostile_clock_width) // 2
        hostile_clock_y = height // 2  # Bottom 50%
        hostile_clock_height = height // 2
        self.roi_cache.register_roi("hostile_clock", (hostile_clock_x, hostile_clock_y, hostile_clock_width, hostile_clock_height))

    def add_state(self, state: State):
        """
        Adds a state instance to the manager.
        
        :param state: An instance of a State subclass.
        """
        self.states.append(state)

    def update(self):
        """
        Captures the screen and updates the current state logic.
        
        Checks if the current state is still active, otherwise searches for a new one.
        """
        # Clear ROI cache at start of each frame
        self.roi_cache.clear_cache()
        
        image = self.vision.capture_screen()
        
        # Check if current state is still active
        if self.current_state:
            # Check sub-state first if it exists
            sub_state = self.current_state.get_active_sub_state()
            if sub_state and sub_state.is_active(image):
                sub_state.execute(image)
                return
            
            if self.current_state.is_active(image):
                self.current_state.execute(image)
                return
            else:
                print(f"Exiting state: {self.current_state.__class__.__name__}")
                self.current_state.on_exit()
                self.current_state = None

        # No state currently active, find one
        for state in self.states:
            if state.is_active(image):
                print(f"Entering state: {state.__class__.__name__}")
                self.current_state = state
                self.current_state.on_enter()
                self.current_state.execute(image)
                return

    def run(self, frequency: float = 0.1):
        """
        Main loop for the state manager.
        
        :param frequency: Delay between updates in seconds.
        """
        print("Starting StateManager loop...")
        try:
            while True:
                self.update()
                time.sleep(frequency)
        except KeyboardInterrupt:
            print("Stopping StateManager...")
            if self.current_state:
                self.current_state.on_exit()
            self.controller.release_all()
