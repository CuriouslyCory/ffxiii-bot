"""
Player direction sensor for detecting player arrow orientation on minimap.
"""
from typing import Optional
import numpy as np
import cv2
from .base import Sensor
from src.core.roi_cache import ROICache


class PlayerDirectionSensor(Sensor):
    """
    Sensor for detecting player direction from the white V-shaped arrow on minimap.
    
    Detects the white arrow indicator in the center 24x24 ROI and returns
    the player's facing direction in degrees (0-360, where 0° = North/Up).
    """
    
    def __init__(self, roi_cache: ROICache):
        """
        Initialize player direction sensor.
        
        Args:
            roi_cache: ROICache instance for accessing cached minimap center arrow ROI
        """
        super().__init__("Player Direction Sensor", "Detects player arrow direction on minimap")
        self.roi_cache = roi_cache
        
        # Grayscale threshold for white arrow (high value = white)
        self.white_threshold = 200
    
    def read(self, image: np.ndarray) -> Optional[float]:
        """
        Read player direction from the cached minimap center arrow ROI.
        
        Args:
            image: Current screen capture (full image)
            
        Returns:
            Player direction in degrees (0-360, where 0° = North/Up),
            or None if arrow cannot be detected.
        """
        if not self.is_enabled:
            return None
        
        # Clear debug outputs from previous frame
        self.clear_debug_outputs()
        
        # Get cached minimap center arrow ROI
        arrow_roi = self.roi_cache.get_roi("minimap_center_arrow", image)
        if arrow_roi is None:
            return None
        
        # Register raw ROI for debug
        self.register_debug_output("arrow_roi_raw", arrow_roi)
        
        # Convert to grayscale
        gray = cv2.cvtColor(arrow_roi, cv2.COLOR_BGR2GRAY)
        
        # Threshold for white pixels
        _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        # Convert grayscale to BGR for debug output
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.register_debug_output("thresholded_arrow", binary_bgr)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Select largest contour (the arrow)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Check minimum area to avoid noise
        if cv2.contourArea(main_contour) < 10:
            return None
        
        # Calculate centroid
        M = cv2.moments(main_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
        
        # Find tip (furthest point from centroid) - this is the V-shape tip
        max_d = -1
        tip = centroid
        for pt in main_contour:
            px, py = pt[0]
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d > max_d:
                max_d = d
                tip = (px, py)
        
        # Calculate angle from centroid to tip
        # In image coordinates: X increases right, Y increases down
        # Arrow tip points in the direction the player is facing
        dx = tip[0] - cx
        dy = tip[1] - cy
        
        # Convert to angle in degrees
        # atan2(dx, -dy) gives angle where:
        #   -dx=0, -dy=-1 (up) = 0°
        #   -dx=1, -dy=0 (right) = 90°
        # Then normalize to [0, 360)
        angle_rad = np.arctan2(dx, -dy)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to [0, 360)
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg
    
    def is_available(self, image: np.ndarray) -> bool:
        """
        Check if minimap center arrow ROI is available.
        
        Args:
            image: Current screen capture
            
        Returns:
            True if minimap center arrow ROI can be extracted
        """
        if not self.roi_cache.has_roi("minimap_center_arrow"):
            return False
        
        arrow_roi = self.roi_cache.get_roi("minimap_center_arrow", image)
        return arrow_roi is not None
