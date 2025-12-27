from src.states.base import State
import time

class MovementState(State):
    """
    MovementState handles navigation in the open world.
    
    Detected by the presence of a minimap in the upper right quadrant.
    """
    
    def is_active(self, image) -> bool:
        """
        Checks for the minimap outline in the top-right ROI.
        
        :param image: Current screen capture.
        :return: True if minimap is detected.
        """
        # ROI: Upper right quadrant for 1920x1080
        roi = (960, 0, 960, 540)
        match = self.vision.find_template("minimap_outline", image, threshold=0.6, roi=roi)
        return match is not None

    def execute(self, image):
        """
        Navigates the character along a set path.
        
        Placeholder logic: Hold 'w' to move forward.
        """
        self.controller.press('w')
        # Here we could add more complex logic to follow a path, 
        # such as checking orientation or specific landmarks.
        pass

    def on_exit(self):
        """Ensure keys are released when leaving the movement state."""
        self.controller.release('w')

