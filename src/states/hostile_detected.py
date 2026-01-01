from src.states.base import State
from src.sensors.minimap_state import MinimapStateSensor


class HostileDetectedState(State):
    """
    HostileDetectedState handles logic when enemies are detected nearby.
    
    Detected when the minimap frame is red, indicating enemies are nearby.
    This state is a placeholder for future hostile detection behavior.
    """
    
    def __init__(self, manager):
        """Initialize hostile detected state."""
        super().__init__(manager)
        
        # Initialize minimap state sensor
        self.minimap_state_sensor = MinimapStateSensor(manager.roi_cache)
    
    def is_active(self, image) -> bool:
        """
        Checks if hostile_detected state is active.
        
        Uses MinimapStateSensor to detect red minimap frame.
        
        :param image: Current screen capture.
        :return: True if hostile_detected state is detected.
        """
        # Enable sensor temporarily for detection
        self.minimap_state_sensor.enable()
        
        # Read minimap state
        state = self.minimap_state_sensor.read(image)
        
        # Disable sensor after reading (will be re-enabled in execute if needed)
        self.minimap_state_sensor.disable()
        
        return state == "hostile_detected"
    
    def execute(self, image):
        """
        Executes hostile detected state instructions.
        
        This is a stub implementation - will be expanded in the future.
        """
        # Enable sensor for execution
        self.minimap_state_sensor.enable()
        
        # Placeholder: Future logic will go here
        # For now, just keep the state active
        
        # Read sensor to verify we're still in hostile_detected state
        state = self.minimap_state_sensor.read(image)
        if state != "hostile_detected":
            # State changed, will transition on next update
            pass
    
    def on_enter(self):
        """Called when entering hostile detected state."""
        super().on_enter()
        print("--- Hostile Detected State ---")
        self.minimap_state_sensor.enable()
    
    def on_exit(self):
        """Called when exiting hostile detected state."""
        super().on_exit()
        self.minimap_state_sensor.disable()
