"""
Enemy position sensor for detecting nearest enemy on minimap.
"""
from typing import Optional, Tuple
import numpy as np
import cv2
from .base import Sensor
from src.core.roi_cache import ROICache
from src.filters.color import HSVFilter


class EnemyPositionSensor(Sensor):
    """
    Sensor for detecting the nearest enemy position on the minimap.
    
    Detects red enemy dots using a saturated red HSV filter and returns
    the pixel offset of the nearest enemy relative to the minimap center.
    """
    
    def __init__(self, roi_cache: ROICache):
        """
        Initialize enemy position sensor.
        
        Args:
            roi_cache: ROICache instance for accessing cached minimap ROI
        """
        super().__init__("Enemy Position Sensor", "Detects nearest enemy position on minimap")
        self.roi_cache = roi_cache
        
        # Custom HSV filter for minimap enemy red (more saturated than AlertFilter)
        # AlertFilter uses: H=0-12, S=26-255, V=140-255
        # For more saturated red: higher saturation threshold
        # Using H=0-12 (red range), S=150-255 (higher saturation), V=150-255 (bright)
        self.enemy_red_filter = HSVFilter(
            lower=(0, 150, 150),
            upper=(12, 255, 255),
            name="Enemy Red Filter",
            description="Filters for saturated red enemy dots on minimap"
        )
        self.register_filter("enemy_red_filter", self.enemy_red_filter)
        
        # Minimum contour area to filter noise
        self.min_contour_area = 5
    
    def read(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Read nearest enemy position from the cached minimap ROI.
        
        Args:
            image: Current screen capture (full image)
            
        Returns:
            Tuple of (dx, dy) pixel offset of nearest enemy relative to minimap center,
            or None if no enemies detected.
            Coordinates: dx > 0 = East, dx < 0 = West
                         dy < 0 = North, dy > 0 = South (in image coordinates, but minimap is north-up)
        """
        if not self.is_enabled:
            return None
        
        # Clear debug outputs from previous frame
        self.clear_debug_outputs()
        
        # Get cached minimap ROI
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        if minimap_roi is None:
            return None
        
        h, w = minimap_roi.shape[:2]
        minimap_center = (w // 2, h // 2)  # Center of minimap
        
        # Apply red filter to detect enemy dots
        filtered = self.enemy_red_filter.apply(minimap_roi)
        self.register_debug_output("filtered_enemies", filtered)
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by minimum area and calculate distances
        valid_enemies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance from center
            dx = cx - minimap_center[0]
            dy = cy - minimap_center[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            valid_enemies.append({
                'centroid': (cx, cy),
                'offset': (dx, dy),
                'distance': distance
            })
        
        if not valid_enemies:
            return None
        
        # Find nearest enemy
        nearest = min(valid_enemies, key=lambda e: e['distance'])
        
        return nearest['offset']
    
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
