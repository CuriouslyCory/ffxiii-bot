"""
Edge detection filters for image processing.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from .base import Filter


class EdgeDetectionFilter(Filter):
    """
    Generic edge detection filter using Canny algorithm.
    """
    
    def __init__(
        self,
        low_threshold: int = 50,
        high_threshold: int = 150,
        name: str = "Edge Detection Filter",
        description: str = ""
    ):
        """
        Initialize edge detection filter.
        
        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            name: Filter name
            description: Filter description
        """
        super().__init__(name, description)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge detection to the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Edge-detected BGR image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Convert edges to BGR and apply as mask
        edge_mask = edges.astype(np.uint8)
        result = cv2.bitwise_and(image, image, mask=edge_mask)
        return result


class CannyFilter(Filter):
    """
    Adaptive Canny edge detection filter with Otsu thresholding.
    """
    
    def __init__(
        self,
        otsu_low_factor: float = 0.5,
        otsu_high_factor: float = 1.5,
        min_low: int = 30,
        max_high: int = 200,
        dilate_kernel: Tuple[int, int] = (2, 2),
        dilate_iterations: int = 1,
        name: str = "Canny Filter",
        description: str = "Adaptive Canny edge detection with Otsu thresholding"
    ):
        """
        Initialize adaptive Canny filter.
        
        Args:
            otsu_low_factor: Factor for lower threshold (multiplies Otsu value)
            otsu_high_factor: Factor for upper threshold (multiplies Otsu value)
            min_low: Minimum lower threshold
            max_high: Maximum upper threshold
            dilate_kernel: Kernel size for edge dilation
            dilate_iterations: Number of dilation iterations
            name: Filter name
            description: Filter description
        """
        super().__init__(name, description)
        self.otsu_low_factor = otsu_low_factor
        self.otsu_high_factor = otsu_high_factor
        self.min_low = min_low
        self.max_high = max_high
        self.dilate_kernel = np.ones(dilate_kernel, np.uint8)
        self.dilate_iterations = dilate_iterations
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive Canny edge detection.
        
        Args:
            image: Input BGR image
            
        Returns:
            Edge-detected BGR image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding using Otsu's method
        otsu_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        
        # Calculate Canny thresholds based on Otsu value
        canny_low = max(self.min_low, int(otsu_val * self.otsu_low_factor))
        canny_high = min(self.max_high, int(otsu_val * self.otsu_high_factor))
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Dilate edges slightly to make them more visible
        edges = cv2.dilate(edges, self.dilate_kernel, iterations=self.dilate_iterations)
        
        # Convert edges to BGR and apply as mask
        edge_mask = edges.astype(np.uint8)
        result = cv2.bitwise_and(image, image, mask=edge_mask)
        return result
