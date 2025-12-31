from src.states.base import State
import time

class ResultsState(State):
    """
    ResultsState handles the post-battle screens.
    
    Detected by "Battle Results" and "Stats" or "Spoils" text.
    """
    
    def is_active(self, image) -> bool:
        """
        Checks for post-battle UI elements in the top-left quadrant.
        
        :param image: Current screen capture.
        :return: True if results screen is detected.
        """
        roi = (0, 0, 960, 540)
        has_results_header = self.vision.find_template("battle_results", image, threshold=0.9, roi=roi) is not None
        
        return has_results_header

    def execute(self, image):
        """
        Instructions for skipping through battle results.
        """
        # Determine specific results type if needed for logging
        # is_stats = self.vision.find_template("stats_label", image, threshold=0.8, roi=(0, 0, 960, 540))
        # is_spoils = self.vision.find_template("spoils_label", image, threshold=0.8, roi=(0, 0, 960, 540))
        
        # Tap gamepad A button to progress
        self.controller.tap("gamepad_a")
        time.sleep(0.2)

