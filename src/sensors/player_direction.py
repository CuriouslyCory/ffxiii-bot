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
        
        # Create debug output with visualization dots
        arrow_roi_with_dots = arrow_roi.copy()
        # Green dot at the center of the minimap_center_arrow ROI
        center_x = arrow_roi.shape[1] // 2
        center_y = arrow_roi.shape[0] // 2
        cv2.circle(arrow_roi_with_dots, (center_x, center_y), 2, (0, 255, 0), -1)
        # Blue dot at the centroid (cx, cy)
        cv2.circle(arrow_roi_with_dots, (cx, cy), 2, (255, 0, 0), -1)
        
        # Find tip using convex hull and angle analysis
        # The tip should be the sharpest vertex (smallest internal angle) on the convex hull
        hull = cv2.convexHull(main_contour, returnPoints=True)
        
        if len(hull) < 3:
            # Fallback to furthest point from centroid if hull is degenerate
            max_d = -1
            tip = centroid
            for pt in main_contour:
                px, py = pt[0]
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d > max_d:
                    max_d = d
                    tip = (px, py)
        else:
            # Find the vertex with the sharpest angle (smallest internal angle)
            min_angle = float('inf')
            tip_idx = 0
            tip = (hull[0][0][0], hull[0][0][1])
            
            for i in range(len(hull)):
                # Get three consecutive points on the hull
                p1 = hull[(i - 1) % len(hull)][0]  # Previous point
                p2 = hull[i][0]                     # Current point (candidate tip)
                p3 = hull[(i + 1) % len(hull)][0]  # Next point
                
                # Calculate vectors from p2 to p1 and p2 to p3
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Calculate the angle between the two vectors
                # Use dot product: cos(angle) = (v1 · v2) / (|v1| * |v2|)
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
                norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = dot_product / (norm1 * norm2)
                    # Clamp to [-1, 1] to avoid numerical errors
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)  # Angle in radians
                    
                    # Smaller angle = sharper point = more likely to be the tip
                    if angle < min_angle:
                        min_angle = angle
                        tip_idx = i
                        tip = (int(p2[0]), int(p2[1]))
        
        cv2.circle(arrow_roi_with_dots, tip, 2, (0, 0, 255), -1)
        self.register_debug_output("arrow_roi_with_markers", arrow_roi_with_dots)
        
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

