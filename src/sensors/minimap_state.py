"""
Minimap state sensor for detecting movement vs hostile_detected states.
"""
from typing import Optional
import numpy as np
import cv2
from .base import Sensor
from src.core.roi_cache import ROICache


class MinimapStateSensor(Sensor):
    """
    Sensor for detecting minimap frame color to determine game state.
    
    Detects if the minimap frame is blue (movement state) or red (hostile_detected state)
    by analyzing the frame border color using HSV color space.
    """
    
    def __init__(self, roi_cache: ROICache):
        """
        Initialize minimap state sensor.
        
        Args:
            roi_cache: ROICache instance for accessing cached minimap ROI
        """
        super().__init__("Minimap State Sensor", "Detects minimap frame color (blue/red) for state detection")
        self.roi_cache = roi_cache
        
        # Blue filter HSV ranges (movement state)
        self.blue_lower = np.array([84, 75, 100])
        self.blue_upper = np.array([97, 245, 245])
        
        # Red/Alert filter HSV ranges (hostile_detected state)
        # Two ranges for HSV wheel wrap
        self.red_lower1 = np.array([0, 40, 50])
        self.red_upper1 = np.array([13, 255, 255])
        self.red_lower2 = np.array([170, 40, 50])
        self.red_upper2 = np.array([180, 255, 255])
    
    def read(self, image: np.ndarray) -> Optional[str]:
        """
        Read minimap state from the cached minimap ROI.
        
        Args:
            image: Current screen capture (full image)
            
        Returns:
            "movement" if blue frame detected,
            "hostile_detected" if red frame detected,
            None if state cannot be determined
        """
        if not self.is_enabled:
            return None
        
        # Get cached minimap ROI
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        if minimap_roi is None:
            return None
        
        return self._detect_frame_color(minimap_roi)
    
    def _detect_frame_color(self, minimap_image: np.ndarray) -> Optional[str]:
        """
        Detect frame color by analyzing the border region of the minimap.
        
        Args:
            minimap_image: Extracted minimap ROI image
            
        Returns:
            "movement", "hostile_detected", or None
        """
        h, w = minimap_image.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(minimap_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for frame border region
        # The minimap is elliptical (320x425), so we create an elliptical mask
        # that excludes the center region to focus on the frame border
        center_x, center_y = w // 2, h // 2
        radius_x, radius_y = w // 2, h // 2
        
        # Inner ellipse radius (to exclude center, keeping only border)
        # Use ~70% of radius to focus on outer 30% border region
        inner_radius_x = int(radius_x * 0.7)
        inner_radius_y = int(radius_y * 0.7)
        
        # Create mask for outer border region
        border_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(border_mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(inner_mask, (center_x, center_y), (inner_radius_x, inner_radius_y), 0, 0, 360, 255, -1)
        border_mask = cv2.subtract(border_mask, inner_mask)
        
        # Apply mask to HSV image to get only border pixels
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=border_mask)
        
        # Count blue pixels
        blue_mask = cv2.inRange(masked_hsv, self.blue_lower, self.blue_upper)
        blue_count = np.count_nonzero(blue_mask)
        
        # Count red pixels (combine both ranges)
        red_mask1 = cv2.inRange(masked_hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(masked_hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_count = np.count_nonzero(red_mask)
        
        # Determine state based on dominant color
        # Use a threshold to avoid false positives (need at least some pixels)
        total_border_pixels = np.count_nonzero(border_mask)
        if total_border_pixels == 0:
            return None
        
        blue_ratio = blue_count / total_border_pixels
        red_ratio = red_count / total_border_pixels
        
        # Threshold for detection (at least 5% of border pixels must match)
        threshold = 0.05
        
        if blue_ratio > threshold and blue_ratio > red_ratio:
            return "movement"
        elif red_ratio > threshold and red_ratio > blue_ratio:
            return "hostile_detected"
        
        return None
    
    def is_available(self, image: np.ndarray) -> bool:
        """
        Check if minimap ROI is available.
        
        Args:
            image: Current screen capture
            
        Returns:
            True if minimap ROI can be extracted
        """
        if not self.roi_cache.has_roi("minimap"):
            return False
        
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        return minimap_roi is not None
