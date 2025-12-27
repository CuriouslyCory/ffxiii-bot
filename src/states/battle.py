from src.states.base import State
import time
import numpy as np

class BattleState(State):
    """
    BattleState handles logic during combat.
    
    Detected by HP/status bars in the lower right and "Paradigm Shift" in the lower left.
    """
    
    def is_active(self, image) -> bool:
        """
        Checks for battle-specific UI elements.
        
        :param image: Current screen capture.
        :return: True if in battle.
        """
        # ROI: Lower left for "Paradigm Shift"
        roi_paradigm = (0, 540, 960, 540)
        # ROI: Lower right for HP bars
        roi_hp = (960, 540, 960, 540)
        
        has_paradigm = self.vision.find_template("paradigm_shift", image, threshold=0.7, roi=roi_paradigm) is not None
        # We might also check for the HP bar container specifically
        has_hp_ui = self.vision.find_template("hp_container", image, threshold=0.7, roi=roi_hp) is not None
        
        return has_paradigm or has_hp_ui

    def execute(self, image):
        """
        Executes battle instructions, like auto-battle or macros.
        """
        # Monitor health bars for emergency macros
        if self.is_low_health(image):
            self.execute_healing_macro()
        else:
            # Standard battle action: tap 'e' for Auto-battle or confirm
            self.controller.tap('e')
        
        time.sleep(0.5)

    def is_low_health(self, image) -> bool:
        """
        Checks if any character's health is below a threshold.
        
        :param image: Current screen capture.
        :return: True if low health detected.
        """
        # Placeholder ROIs for character HP bars (needs calibration for actual game window)
        # These are examples of where HP bars might be located in the lower right
        hp_bar_rois = [
            (1450, 850, 200, 10), # Char 1
            (1450, 880, 200, 10), # Char 2
            (1450, 910, 200, 10), # Char 3
        ]
        
        for roi in hp_bar_rois:
            percent = self.calculate_hp_percentage(image, roi)
            if 0 < percent < 30: # 30% threshold
                return True
        return False

    def calculate_hp_percentage(self, image, roi) -> float:
        """
        Estimates HP percentage based on filled pixels in a given ROI.
        """
        try:
            hp_slice = self.vision.get_roi_slice(image, roi)
            gray = cv2.cvtColor(hp_slice, cv2.COLOR_BGR2GRAY)
            # Thresholding to isolate the health bar fill (typically bright white/green)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            total = thresh.size
            if total == 0: return 0.0
            filled = np.count_nonzero(thresh)
            return (filled / total) * 100
        except Exception:
            return 0.0

    def execute_healing_macro(self):
        """
        Executes a sequence of keys to shift paradigms or use a potion.
        """
        print("Executing healing macro!")
        # Example: Paradigm Shift (L1 on controller, maybe 'q' on keyboard)
        self.controller.tap('q')
        time.sleep(0.2)
        self.controller.tap('e') # Confirm selection
        time.sleep(1.0) # Wait for animation

