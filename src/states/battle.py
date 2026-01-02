from src.states.base import State
from src.skills.button import ButtonTapSkill
from src.sensors.health import HealthSensor
from src.ui.menu import MenuDefinition, MenuItem
from src.ui.input_handler import InputHandler
import time
import numpy as np
import cv2


class BattleState(State):
    """
    BattleState handles logic during combat.
    
    Detected by HP/status bars in the lower right and "Paradigm Shift" in the lower left.
    Now uses skills and sensors for cleaner architecture.
    """
    
    def __init__(self, manager):
        """Initialize battle state."""
        super().__init__(manager)
        
        # Skills
        self.tap_skill = ButtonTapSkill(self.controller)
        
        # Sensors
        self.health_sensor = HealthSensor(
            self.vision,
            hp_bar_rois=[
                (1450, 850, 200, 10),  # Char 1
                (1450, 880, 200, 10),  # Char 2
                (1450, 910, 200, 10),  # Char 3
            ]
        )
        
        # Pause state
        self.paused = False
        
        # Input handler for pause hotkey
        self.input_handler = InputHandler()
        self.input_handler.set_key_bindings({'p': 'TOGGLE_PAUSE'})
        
        # Sensors can be managed directly or via registry if sub-states are added
    
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
        has_hp_ui = self.vision.find_template("hp_container", image, threshold=0.7, roi=roi_hp) is not None
        
        return has_paradigm or has_hp_ui
    
    def execute(self, image):
        """
        Executes battle instructions using skills.
        """
        # Check for pause toggle action
        actions = self.input_handler.get_pending_actions()
        if 'TOGGLE_PAUSE' in actions:
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            print(f"Battle state {status}")
        
        # If paused, skip button pressing
        if self.paused:
            return
        
        # Check health using sensor (only if enabled)
        if self.health_sensor.is_enabled:
            is_low = self.health_sensor.is_low_health(image, threshold_percent=30.0)
            if is_low:
                # Could trigger healing macro here
                pass
        
        # Standard battle action: tap gamepad A button for Auto-battle or confirm
        self.tap_skill.tap("gamepad_a", duration=0.2)
        
        time.sleep(0.2)
    
    def get_menu(self) -> MenuDefinition:
        """
        Get menu definition for battle state.
        
        Returns:
            MenuDefinition with battle state controls
        """
        # Example menu - can be extended with more options
        return MenuDefinition(
            title="Battle State",
            items=[
                MenuItem('p', 'Toggle Pause', 'TOGGLE_PAUSE'),
                # Could add menu items here if needed
                # MenuItem('h', 'Toggle Health Sensor', 'TOGGLE_HEALTH'),
            ]
        )
    
    def on_enter(self):
        """Called when entering battle state."""
        super().on_enter()
        # Enable health sensor if needed
        # self.health_sensor.enable()
    
    def on_exit(self):
        """Called when exiting battle state."""
        super().on_exit()
        # Disable health sensor
        self.health_sensor.disable()
