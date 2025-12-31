"""
Minimap sensor for extracting and processing minimap data.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from .base import Sensor


class MinimapSensor(Sensor):
    """
    Sensor for extracting minimap ROI from the game screen.
    
    Provides minimap extraction with optional filtering.
    """
    
    def __init__(self, vision_engine, roi: Tuple[int, int, int, int] = (1530, 0, 400, 400)):
        """
        Initialize minimap sensor.
        
        Args:
            vision_engine: VisionEngine instance for template matching
            roi: Minimap ROI as (x, y, width, height)
        """
        super().__init__("Minimap Sensor", "Extracts minimap ROI from screen")
        self.vision = vision_engine
        self.roi = roi
        self._calibrated = False
    
    def calibrate(self, image: np.ndarray) -> bool:
        """
        Attempt to auto-calibrate the minimap ROI using template matching.
        
        Args:
            image: Current screen capture
            
        Returns:
            True if calibration succeeded, False otherwise
        """
        # Search in the top-right quadrant
        h, w = image.shape[:2]
        search_roi = (w // 2, 0, w // 2, h // 2)
        
        if "minimap_outline" not in self.vision.templates:
            return False
        
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=search_roi)
        
        if match:
            mx, my, conf = match
            template = self.vision.templates["minimap_outline"]
            th, tw = template.shape[:2]
            
            # Center of the match
            center_x = mx + tw // 2
            center_y = my + th // 2
            
            # Define new ROI centered on this
            roi_w, roi_h = 400, 400
            new_x = max(0, center_x - roi_w // 2)
            new_y = max(0, center_y - roi_h // 2)
            
            # Ensure within bounds
            if new_x + roi_w > w:
                new_x = w - roi_w
            if new_y + roi_h > h:
                new_y = h - roi_h
            
            self.roi = (new_x, new_y, roi_w, roi_h)
            self._calibrated = True
            return True
        
        return False
    
    def read(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract minimap ROI from the image.
        
        Args:
            image: Current screen capture
            
        Returns:
            Minimap ROI image, or None if extraction failed
        """
        if not self.is_enabled:
            return None
        
        # Auto-calibrate if not calibrated
        if not self._calibrated:
            self.calibrate(image)
        
        return self.vision.get_roi_slice(image, self.roi)
    
    def is_available(self, image: np.ndarray) -> bool:
        """
        Check if minimap is available (visible) in the current image.
        
        Args:
            image: Current screen capture
            
        Returns:
            True if minimap is available
        """
        if not self._calibrated:
            return self.calibrate(image)
        
        # Check if ROI is valid
        x, y, w, h = self.roi
        img_h, img_w = image.shape[:2]
        return (x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h)
