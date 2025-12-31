"""
Compass sensor for determining player orientation/direction.
"""
from typing import Optional, Tuple
import numpy as np
from .base import Sensor


class CompassSensor(Sensor):
    """
    Sensor for determining compass direction/orientation.
    
    Uses ORB/feature matching to calculate compass direction.
    This is a placeholder implementation - full implementation would
    use visual odometry or feature matching with a reference map.
    """
    
    def __init__(self, vision_engine):
        """
        Initialize compass sensor.
        
        Args:
            vision_engine: VisionEngine instance for feature matching
        """
        super().__init__("Compass Sensor", "Determines player compass direction")
        self.vision = vision_engine
        self._last_direction: Optional[float] = None
    
    def read(self, image: np.ndarray) -> Optional[float]:
        """
        Read current compass direction.
        
        Args:
            image: Current screen capture
            
        Returns:
            Compass direction in degrees (0-360), or None if unavailable
        """
        if not self.is_enabled:
            return None
        
        # TODO: Implement actual compass detection using ORB/feature matching
        # This would require:
        # 1. Reference minimap or landmark features
        # 2. Feature matching to determine orientation
        # 3. Calculation of compass direction from orientation
        
        # For now, return None (sensor not fully implemented)
        return self._last_direction
    
    def is_available(self, image: np.ndarray) -> bool:
        """
        Check if compass data is available.
        
        Args:
            image: Current screen capture
            
        Returns:
            True if compass data is available
        """
        # Compass requires minimap to be visible
        # Check if minimap outline is present
        h, w = image.shape[:2]
        roi = (w // 2, 0, w // 2, h // 2)
        
        if "minimap_outline" not in self.vision.templates:
            return False
        
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=roi)
        return match is not None
