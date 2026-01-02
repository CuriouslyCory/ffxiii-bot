"""
Color filters for image processing using HSV color space.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from .base import Filter


class HSVFilter(Filter):
    """
    Generic HSV color range filter with configurable ranges.
    """
    
    def __init__(
        self,
        lower: Tuple[int, int, int],
        upper: Tuple[int, int, int],
        name: str = "HSV Filter",
        description: str = ""
    ):
        """
        Initialize HSV filter.
        
        Args:
            lower: Lower HSV bounds (H, S, V)
            upper: Upper HSV bounds (H, S, V)
            name: Filter name
            description: Filter description
        """
        super().__init__(name, description)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HSV color filtering to the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image (only pixels in HSV range are visible)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result


class BlueFilter(Filter):
    """
    Blue color filter for minimap features (normal state).
    """
    
    def __init__(self):
        """Initialize blue filter with default HSV ranges."""
        super().__init__("Blue Filter", "Filters for blue minimap features")
        # Blue filter (normal minimap state)
        self.lower = np.array([90, 81, 72])
        self.upper = np.array([108, 245, 255])
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply blue color filtering.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image with only blue pixels visible
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result


class GoldFilter(Filter):
    """
    Gold color filter for player arrow indicator.
    """
    
    def __init__(self):
        """Initialize gold filter with default HSV ranges."""
        super().__init__("Gold Filter", "Filters for gold/arrow features")
        # Gold filter (player arrow indicator)
        self.lower = np.array([15, 100, 150])
        self.upper = np.array([45, 255, 255])
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gold color filtering.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image with only gold pixels visible
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result


class AlertFilter(Filter):
    """
    Alert/Red color filter for minimap in enemy nearby state.
    Handles HSV wheel wrap for red color.
    """
    
    def __init__(self):
        """Initialize alert filter with default HSV ranges."""
        super().__init__("Alert Filter", "Filters for red/alert minimap features")
        # Alert/Red filter (enemy nearby state - two ranges for HSV wheel wrap)
        self.lower1 = np.array([0, 26, 140])
        self.upper1 = np.array([12, 255, 255])

    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply alert/red color filtering (handles HSV wheel wrap).
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image with only red/alert pixels visible
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower1, self.upper1)
        result = cv2.bitwise_and(image, image, mask=mask1)
        return result


class MinimapColorFilter(Filter):
    """
    Combined color filter for minimap features (blue, alert/red, and optionally gold).
    This combines BlueFilter, AlertFilter, and optionally GoldFilter.
    """
    
    def __init__(self, include_gold: bool = True):
        """
        Initialize minimap color filter.
        
        Args:
            include_gold: Whether to include gold filter for arrow detection
        """
        super().__init__(
            "Minimap Color Filter",
            "Combined filter for minimap features (blue, alert, gold)"
        )
        self.include_gold = include_gold
        self.blue_filter = BlueFilter()
        self.alert_filter = AlertFilter()
        self.gold_filter = GoldFilter() if include_gold else None
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply combined minimap color filtering.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image with minimap features visible
        """
        # Apply blue filter
        blue_result = self.blue_filter.apply(image)
        
        # Apply alert filter
        alert_result = self.alert_filter.apply(image)
        
        # Combine blue and alert (additive)
        combined = cv2.bitwise_or(blue_result, alert_result)
        
        # Optionally add gold
        if self.include_gold and self.gold_filter:
            gold_result = self.gold_filter.apply(image)
            combined = cv2.bitwise_or(combined, gold_result)
        
        return combined
