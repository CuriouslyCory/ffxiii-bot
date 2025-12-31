"""
Health sensor for detecting character health percentages.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
from .base import Sensor


class HealthSensor(Sensor):
    """
    Sensor for detecting character health percentages from HP bars.
    
    Calculates HP percentage based on filled pixels in HP bar ROIs.
    """
    
    def __init__(
        self,
        vision_engine,
        hp_bar_rois: List[Tuple[int, int, int, int]],
        threshold: int = 127
    ):
        """
        Initialize health sensor.
        
        Args:
            vision_engine: VisionEngine instance for ROI extraction
            hp_bar_rois: List of HP bar ROIs as (x, y, width, height) tuples
            threshold: Threshold value for binary thresholding
        """
        super().__init__("Health Sensor", "Detects character HP percentages")
        self.vision = vision_engine
        self.hp_bar_rois = hp_bar_rois
        self.threshold = threshold
    
    def read(self, image: np.ndarray) -> List[float]:
        """
        Read health percentages for all characters.
        
        Args:
            image: Current screen capture
            
        Returns:
            List of HP percentages (0-100) for each character
        """
        if not self.is_enabled:
            return []
        
        health_percentages = []
        for roi in self.hp_bar_rois:
            percent = self._calculate_hp_percentage(image, roi)
            health_percentages.append(percent)
        
        return health_percentages
    
    def _calculate_hp_percentage(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
        """
        Calculate HP percentage from a single HP bar ROI.
        
        Args:
            image: Current screen capture
            roi: HP bar ROI as (x, y, width, height)
            
        Returns:
            HP percentage (0-100)
        """
        try:
            hp_slice = self.vision.get_roi_slice(image, roi)
            gray = cv2.cvtColor(hp_slice, cv2.COLOR_BGR2GRAY)
            # Thresholding to isolate the health bar fill (typically bright white/green)
            _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
            
            total = thresh.size
            if total == 0:
                return 0.0
            
            filled = np.count_nonzero(thresh)
            return (filled / total) * 100
        except Exception:
            return 0.0
    
    def is_low_health(self, image: np.ndarray, threshold_percent: float = 30.0) -> bool:
        """
        Check if any character has low health.
        
        Args:
            image: Current screen capture
            threshold_percent: Health percentage threshold (default 30%)
            
        Returns:
            True if any character has health below threshold
        """
        if not self.is_enabled:
            return False
        
        percentages = self.read(image)
        return any(0 < percent < threshold_percent for percent in percentages)
