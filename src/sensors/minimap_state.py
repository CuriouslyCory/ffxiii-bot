"""
Minimap state sensor for detecting movement vs hostile_detected states.
"""
from typing import Optional
import numpy as np
import cv2
from .base import Sensor
from src.core.roi_cache import ROICache
from src.filters.mask import EllipseMaskFilter
from src.filters.color import HSVFilter, AlertFilter, BlueFilter
from src.filters.composite import CompositeFilter


class MinimapStateSensor(Sensor):
    """
    Sensor for detecting minimap frame color to determine game state.
    
    Detects if the minimap frame is blue (movement state) or red (hostile_detected state)
    by analyzing the frame border color using HSV color space.
    
    First verifies that a minimap template match exists before checking colors,
    to avoid false positives from blue sky or red lava. Template matching uses the
    same mask and grayscale preprocessing as color detection for consistency.
    
    Uses composable filters:
    - minimap_frame: Masks to isolate the border region (outer ellipse - inner ellipse)
    - minimap_detected: minimap_frame + blue color filter (movement state)
    - minimap_hostile_detected: minimap_frame + red/alert color filter (hostile_detected state)
    """
    
    def __init__(self, roi_cache: ROICache):
        """
        Initialize minimap state sensor.
        
        Args:
            roi_cache: ROICache instance for accessing cached minimap ROI
        """
        super().__init__("Minimap State Sensor", "Detects minimap frame color (blue/red) for state detection")
        self.roi_cache = roi_cache
        
        # Default minimap dimensions (320x425)
        # These will be adjusted based on actual ROI size in _detect_frame_color
        self._setup_filters()
    
    def _setup_filters(self, width: int = 430, height: int = 320):
        """
        Set up filter pipelines for minimap processing.
        
        Args:
            width: Width of minimap ROI
            height: Height of minimap ROI
        """
        center_x, center_y = 215, 160 # width // 2, height // 2
        radius_x, radius_y = 212, 160 # width // 2, height // 2
        
        # Inner ellipse radius (to exclude center, keeping only border)
        # Use ~70% of radius to focus on outer 30% border region
        inner_radius_x = 181 #int(radius_x * 0.7)
        inner_radius_y = 131 #int(radius_y * 0.7)
        
        # Create outer ellipse mask (keep inside)
        outer_mask = EllipseMaskFilter(
            center=(center_x, center_y),
            axes=(radius_x, radius_y),
            mask_inside=True,
            name="Minimap Outer Mask"
        )
        
        # Create inner ellipse mask (keep outside = mask inside)
        inner_mask = EllipseMaskFilter(
            center=(center_x, center_y),
            axes=(inner_radius_x, inner_radius_y),
            mask_inside=False,  # Keep outside (border region)
            name="Minimap Inner Mask"
        )
        
        # Create minimap_frame filter: outer mask then inner mask (progressive)
        # This results in: outer ellipse - inner ellipse = border region
        minimap_frame = CompositeFilter(
            [outer_mask, inner_mask],
            mode="progressive",
            name="Minimap Frame",
            description="Isolates minimap border region using nested ellipses"
        )
        self.register_filter("minimap_frame", minimap_frame)
        
        # Create blue detection filter: minimap_frame + blue color filter
        blue_filter = BlueFilter()
        minimap_detected = CompositeFilter(
            [minimap_frame, blue_filter],
            mode="progressive",
            name="Minimap Detected (Blue)",
            description="Minimap border filtered for blue (movement state)"
        )
        self.register_filter("minimap_detected", minimap_detected)
        
        # Create red/alert detection filter: minimap_frame + alert color filter
        alert_filter = AlertFilter()
        minimap_hostile_detected = CompositeFilter(
            [minimap_frame, alert_filter],
            mode="progressive",
            name="Minimap Hostile Detected (Red)",
            description="Minimap border filtered for red/alert (hostile_detected state)"
        )
        self.register_filter("minimap_hostile_detected", minimap_hostile_detected)
        
        # Store mask filters for potential parameter adjustment
        self.register_filter("outer_mask", outer_mask)
        self.register_filter("inner_mask", inner_mask)
        self.register_filter("blue_filter", blue_filter)
        self.register_filter("alert_filter", alert_filter)
    
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
        
        # Clear debug outputs from previous frame
        self.clear_debug_outputs()
        
        # Get cached minimap ROI
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        if minimap_roi is None:
            return None
        
        # Update filter dimensions if ROI size changed
        h, w = minimap_roi.shape[:2]
        # Re-setup filters if dimensions changed (this is a simple approach;
        # in production, filters might need to be more dynamic)
        if not hasattr(self, '_last_width') or self._last_width != w or self._last_height != h:
            self._setup_filters(width=w, height=h)
            self._last_width = w
            self._last_height = h
        
        # First verify we're actually looking at a minimap before checking colors
        if not self._detect_minimap(minimap_roi):
            return None
        
        return self._detect_frame_color(minimap_roi)
    
    def _detect_minimap(self, minimap_image: np.ndarray) -> bool:
        """
        Detect if the ROI actually contains a minimap by template matching.
        
        Uses the same mask and grayscale preprocessing for both the ROI and template
        to ensure consistent matching. This prevents false positives from blue sky
        or red lava that might match color filters.
        
        Args:
            minimap_image: Extracted minimap ROI image
            
        Returns:
            True if minimap template is detected, False otherwise
        """
        # Get the minimap_frame filter (same masks used in color detection)
        minimap_frame_filter = self._registered_filters.get("minimap_frame")
        if not minimap_frame_filter:
            # If filters aren't set up, can't verify minimap - return True to allow color check
            return True
        
        # Apply the same mask to the ROI image
        masked_roi = minimap_frame_filter.apply(minimap_image)
        
        # Convert to grayscale
        roi_gray = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2GRAY)
        
        # Get template if available
        if not hasattr(self.roi_cache, 'vision') or not self.roi_cache.vision:
            # Vision engine not available - can't verify, allow color check
            return True
        
        vision = self.roi_cache.vision
        if "minimap_outline" not in vision.templates:
            # Template not loaded - can't verify, allow color check
            return True
        
        template = vision.templates["minimap_outline"]
        
        # Resize template to match ROI dimensions if needed
        h, w = minimap_image.shape[:2]
        template_h, template_w = template.shape[:2]
        
        # If template is larger than ROI, we can't match it
        if template_h > h or template_w > w:
            print("Template is larger than ROI, can't match it")
            return False
        
        # Apply the same mask and grayscale to template
        # First resize template to ROI size if different
        if template_h != h or template_w != w:
            template_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            template_resized = template
        
        # Apply the same minimap_frame mask to template
        masked_template = minimap_frame_filter.apply(template_resized)
        
        # Convert to grayscale
        template_gray = cv2.cvtColor(masked_template, cv2.COLOR_BGR2GRAY)
        
        # Check for match using template matching
        # Since both images are already masked and grayscale, we can use normalized correlation
        result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Use threshold of 0.7 as it should be a high confidence match
        threshold = 0.3
        return max_val >= threshold
    
    def _detect_frame_color(self, minimap_image: np.ndarray) -> Optional[str]:
        """
        Detect frame color by analyzing the border region of the minimap.
        
        Uses registered filters to process the minimap and register debug outputs.
        Should only be called after _detect_minimap confirms a minimap is present.
        
        Args:
            minimap_image: Extracted minimap ROI image
            
        Returns:
            "movement", "hostile_detected", or None
        """
        
        # Get registered filters
        minimap_frame_filter = self._registered_filters.get("minimap_frame")
        minimap_detected_filter = self._registered_filters.get("minimap_detected")
        minimap_hostile_detected_filter = self._registered_filters.get("minimap_hostile_detected")
        
        if not minimap_frame_filter or not minimap_detected_filter or not minimap_hostile_detected_filter:
            # Fallback to old logic if filters not set up
            return None
        
        # Apply minimap_frame filter to get masked border region
        masked_frame = minimap_frame_filter.apply(minimap_image)
        self.register_debug_output("masked_frame", masked_frame)
        
        # Apply blue filter to get blue pixels in border
        blue_filtered = minimap_detected_filter.apply(minimap_image)
        self.register_debug_output("blue_filtered", blue_filtered)
        
        # Apply red/alert filter to get red pixels in border
        red_filtered = minimap_hostile_detected_filter.apply(minimap_image)
        self.register_debug_output("red_filtered", red_filtered)
        
        # Count pixels in filtered results
        # Convert to grayscale and count non-zero pixels
        blue_gray = cv2.cvtColor(blue_filtered, cv2.COLOR_BGR2GRAY)
        blue_count = np.count_nonzero(blue_gray)
        
        red_gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
        red_count = np.count_nonzero(red_gray)
        
        # Count total border pixels from masked frame
        masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        total_border_pixels = np.count_nonzero(masked_gray)
        
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
