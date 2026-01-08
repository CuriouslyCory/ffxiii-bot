"""
Health sensor for detecting character health percentages.
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
from .base import Sensor
from src.filters.color import HSVFilter


class HealthSensor(Sensor):
    """
    Sensor for detecting character health percentages from HP bars.
    
    Uses HSV color filtering to detect green/yellow/red health bars and
    calculates HP percentage based on filled bar length.
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
            threshold: Threshold value (kept for backward compatibility, not used in color-based detection)
        """
        super().__init__("Health Sensor", "Detects character HP percentages using color filtering")
        self.vision = vision_engine
        self.hp_bar_rois = hp_bar_rois
        self.threshold = threshold
        
        # Set up color filters for health bars
        self._setup_filters()
    
    def _setup_filters(self):
        """
        Set up HSV color filters for green, yellow, and red health bars.
        
        Default HSV ranges (to be tuned via debug utility):
        - Green: (40, 50, 50) to (80, 255, 255)
        - Yellow: (15, 100, 100) to (35, 255, 255)
        - Red: Two ranges for HSV wheel wrap (0-10 and 170-180)
        """
        # Green HP bar filter
        green_filter = HSVFilter(
            lower=(145, 52, 40),
            upper=(150, 60, 80),
            name="Green HP Filter",
            description="Filters for green health bar color"
        )
        self.register_filter("green_hp_filter", green_filter)
        
        # Yellow HP bar filter
        yellow_filter = HSVFilter(
            lower=(15, 100, 100),
            upper=(35, 255, 255),
            name="Yellow HP Filter",
            description="Filters for yellow health bar color"
        )
        self.register_filter("yellow_hp_filter", yellow_filter)
        
        # Red HP bar filter
        red_filter = HSVFilter(
            lower=(0, 50, 50),
            upper=(10, 255, 255),
            name="Red HP Filter",
            description="Filters for red health bar color"
        )
        self.register_filter("red_hp_filter", red_filter)
    
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
        
        # Clear debug outputs from previous frame
        self.clear_debug_outputs()
        
        health_percentages = []
        for idx, roi in enumerate(self.hp_bar_rois):
            percent = self._calculate_hp_percentage(image, roi, idx)
            health_percentages.append(percent)
        
        return health_percentages
    
    def _calculate_hp_percentage(self, image: np.ndarray, roi: Tuple[int, int, int, int], bar_index: int = 0) -> float:
        """
        Calculate HP percentage from a single HP bar ROI using color-based detection.
        
        Detects which color (green/yellow/red) the health bar is, then measures
        the filled length of the bar to calculate percentage.
        
        Args:
            image: Current screen capture
            roi: HP bar ROI as (x, y, width, height)
            bar_index: Index of the bar (for debug output labeling)
            
        Returns:
            HP percentage (0-100)
        """
        try:
            # Extract HP bar ROI
            hp_slice = self.vision.get_roi_slice(image, roi)
            h, w = hp_slice.shape[:2]
            
            # Register raw ROI for debug (all bars)
            self.register_debug_output(f"hp_bar_{bar_index}_raw", hp_slice)
            
            # Get registered filters
            filters = self.get_registered_filters()
            green_filter = filters.get("green_hp_filter")
            yellow_filter = filters.get("yellow_hp_filter")
            red_filter = filters.get("red_hp_filter")
            
            if not all([green_filter, yellow_filter, red_filter]):
                # Fallback to old method if filters not set up
                return self._calculate_hp_percentage_fallback(hp_slice)
            
            # Apply each color filter
            green_result = green_filter.apply(hp_slice)
            yellow_result = yellow_filter.apply(hp_slice)
            red_result = red_filter.apply(hp_slice)
            
            # Register filtered results for debug (all bars)
            self.register_debug_output(f"hp_bar_{bar_index}_green", green_result)
            self.register_debug_output(f"hp_bar_{bar_index}_yellow", yellow_result)
            self.register_debug_output(f"hp_bar_{bar_index}_red", red_result)
            
            # Convert to grayscale for pixel counting
            green_gray = cv2.cvtColor(green_result, cv2.COLOR_BGR2GRAY)
            yellow_gray = cv2.cvtColor(yellow_result, cv2.COLOR_BGR2GRAY)
            red_gray = cv2.cvtColor(red_result, cv2.COLOR_BGR2GRAY)
            
            # Count pixels in each color
            green_pixels = np.count_nonzero(green_gray)
            yellow_pixels = np.count_nonzero(yellow_gray)
            red_pixels = np.count_nonzero(red_gray)
            
            # Determine which color has the most pixels (the bar color)
            max_pixels = max(green_pixels, yellow_pixels, red_pixels)
            
            if max_pixels == 0:
                # No color detected - bar is empty
                return 0.0
            
            # Select the detected color
            if green_pixels == max_pixels:
                detected_color = green_gray
                color_name = "green"
            elif yellow_pixels == max_pixels:
                detected_color = yellow_gray
                color_name = "yellow"
            else:
                detected_color = red_gray
                color_name = "red"
            
            # Find the rightmost edge of the filled region
            # Use horizontal projection to find the rightmost column with significant pixel density
            filled_width = self._find_filled_width(detected_color, w)
            
            # Calculate percentage
            percentage = (filled_width / w) * 100
            
            # Create debug visualization with detected bar and measurement overlay (all bars)
            detected_vis = hp_slice.copy()
            # Draw detected color region
            color_bgr = {
                "green": (0, 255, 0),
                "yellow": (0, 255, 255),
                "red": (0, 0, 255)
            }
            overlay = detected_vis.copy()
            overlay[detected_color > 0] = color_bgr[color_name]
            cv2.addWeighted(overlay, 0.5, detected_vis, 0.5, 0, detected_vis)
            
            # Draw measurement line
            if filled_width > 0: 
                cv2.line(detected_vis, (int(filled_width), 0), (int(filled_width), h - 1), (255, 255, 255), 2)
            
            # Add text annotation
            cv2.putText(detected_vis, f"Bar {bar_index + 1}: {percentage:.1f}% ({color_name})", (5, h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            self.register_debug_output(f"hp_bar_{bar_index}_detected", detected_vis)
            
            return percentage
            
        except Exception:
            # Fallback to old method on error
            try:
                hp_slice = self.vision.get_roi_slice(image, roi)
                return self._calculate_hp_percentage_fallback(hp_slice)
            except Exception:
                return 0.0
    
    def _find_filled_width(self, binary_image: np.ndarray, total_width: int) -> float:
        """
        Find the rightmost edge of the filled region in a binary image.
        
        Uses horizontal projection to find the rightmost column with significant
        pixel density, providing a more robust measurement than simple rightmost pixel.
        
        Args:
            binary_image: Binary grayscale image (0 or 255)
            total_width: Total width of the image
            
        Returns:
            Filled width (0 to total_width)
        """
        h, w = binary_image.shape
        
        # Calculate horizontal projection (sum of pixels in each column)
        horizontal_projection = np.sum(binary_image > 0, axis=0)
        
        # Find the rightmost column with at least 10% of row height as pixels
        # This threshold helps avoid noise at the edges
        threshold = max(1, int(h * 0.1))
        
        # Find rightmost column that meets threshold
        filled_width = 0
        for col in range(w - 1, -1, -1):
            if horizontal_projection[col] >= threshold:
                filled_width = col + 1
                break
        
        return float(filled_width)
    
    def _calculate_hp_percentage_fallback(self, hp_slice: Optional[np.ndarray]) -> float:
        """
        Fallback method using old thresholding approach.
        
        Args:
            hp_slice: HP bar ROI image (may be None)
            
        Returns:
            HP percentage (0-100)
        """
        if hp_slice is None:
            return 0.0
        
        try:
            gray = cv2.cvtColor(hp_slice, cv2.COLOR_BGR2GRAY)
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
