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
    
    def __init__(
        self,
        roi_cache: ROICache,
        template_match_threshold: float = 0.09,
        template_match_method: str = "auto"
    ):
        """
        Initialize minimap state sensor.
        
        Args:
            roi_cache: ROICache instance for accessing cached minimap ROI
            template_match_threshold: Threshold for template matching (0.0-1.0)
                                     Default 0.12 for normalized grayscale matching with opacity variations
            template_match_method: Matching method to use (currently unused, kept for compatibility)
                                  "auto" selects best method automatically
        """
        super().__init__("Minimap State Sensor", "Detects minimap frame color (blue/red) for state detection")
        self.roi_cache = roi_cache
        self.template_match_threshold = template_match_threshold
        self.template_match_method = template_match_method
        
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
        inner_radius_x = 50 #int(radius_x * 0.7)
        inner_radius_y = 50 #int(radius_y * 0.7)
        
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
        
        # Create combined blue+alert filter for pre-check (additive mode)
        minimap_color_filter = CompositeFilter(
            [blue_filter, alert_filter],
            mode="additive",
            name="Minimap Color Filter (Blue+Alert)",
            description="Combined blue and alert color filter for minimap pre-check"
        )
        self.register_filter("minimap_color_filter", minimap_color_filter)
        
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
        
        Uses normalized grayscale and edge detection preprocessing on masked images
        (no color filtering) to handle opacity variations while preserving structure.
        The minimap border can have variable saturation and value due to opacity,
        so normalization handles brightness variations better than color filtering.
        
        Processing pipeline:
        1. Apply mask to isolate border region (for both ROI and template)
        2. Convert to grayscale and normalize for brightness/opacity invariance
        3. Optionally apply edge detection for structure-based matching
        4. Match template using normalized grayscale or edge-detected images
        
        Args:
            minimap_image: Extracted minimap ROI image
            
        Returns:
            True if minimap template is detected, False otherwise
        """
        # Get the minimap_frame filter and color filter (same masks used in color detection)
        minimap_frame_filter = self._registered_filters.get("minimap_frame")
        minimap_color_filter = self._registered_filters.get("minimap_color_filter")
        if not minimap_frame_filter or not minimap_color_filter:
            # If filters aren't set up, can't verify minimap - return True to allow color check
            return True
        
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
        
        # Apply the same mask to both ROI and template (no color filtering - it removes too much info)
        # First resize template to ROI size if different
        if template_h != h or template_w != w:
            template_resized = cv2.resize(template, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            template_resized = template
        
        # Apply the same minimap_frame mask to both (don't color filter - preserves structure)
        masked_roi = minimap_frame_filter.apply(minimap_image)
        masked_template = minimap_frame_filter.apply(template_resized)
        
        self.register_debug_output("minimap_precheck_template_masked", masked_template)
        self.register_debug_output("minimap_precheck_roi_masked", masked_roi)
        
        # Preprocess for matching: normalize grayscale to handle opacity variations
        # Try multiple methods and use the best match
        def preprocess_normalized_grayscale(color_image: np.ndarray) -> np.ndarray:
            """
            Preprocess image using normalized grayscale (no edge detection).
            
            This method preserves more detail and works better when both images
            have similar structure but different intensities.
            
            Args:
                color_image: Input BGR image
                
            Returns:
                Normalized grayscale image
            """
            # Convert to grayscale
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Normalize to handle brightness/opacity variations
            # This helps when the minimap has variable opacity
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Optional: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This can help with varying opacity levels
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(normalized)
            
            return normalized
        
        def preprocess_edge_detection(color_image: np.ndarray) -> np.ndarray:
            """
            Preprocess image using edge detection (Canny with adaptive thresholds).
            
            This method focuses on structure/shape rather than intensity.
            
            Args:
                color_image: Input BGR image
                
            Returns:
                Edge-detected grayscale image
            """
            # Convert to grayscale
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Normalize first to handle brightness variations
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply adaptive edge detection using Otsu thresholding
            otsu_val = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            
            # Calculate adaptive Canny thresholds based on Otsu value
            canny_low = max(30, int(otsu_val * 0.5))
            canny_high = min(200, int(otsu_val * 1.5))
            
            # Apply Canny edge detection
            edges = cv2.Canny(normalized, canny_low, canny_high)
            
            # Lightly dilate edges to make them more robust to small variations
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            return edges
        
        # Preprocess masked ROI and template directly (no color filtering - preserves structure)
        # The mask already isolates the border region, and normalization handles opacity variations
        roi_processed = preprocess_normalized_grayscale(masked_roi)
        template_processed = preprocess_normalized_grayscale(masked_template)
        
        # Also try edge detection as alternative
        roi_edges = preprocess_edge_detection(masked_roi)
        template_edges = preprocess_edge_detection(masked_template)
        
        # Register debug outputs for visualization
        roi_processed_bgr = cv2.cvtColor(roi_processed, cv2.COLOR_GRAY2BGR)
        template_processed_bgr = cv2.cvtColor(template_processed, cv2.COLOR_GRAY2BGR)
        roi_edges_bgr = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
        template_edges_bgr = cv2.cvtColor(template_edges, cv2.COLOR_GRAY2BGR)
        self.register_debug_output("minimap_precheck_roi_normalized", roi_processed_bgr)
        self.register_debug_output("minimap_precheck_template_normalized", template_processed_bgr)
        self.register_debug_output("minimap_precheck_roi_edges", roi_edges_bgr)
        self.register_debug_output("minimap_precheck_template_edges", template_edges_bgr)
        
        # Try matching with normalized grayscale (primary method)
        result_normalized = cv2.matchTemplate(roi_processed, template_processed, cv2.TM_CCOEFF_NORMED)
        _, max_val_normalized, _, max_loc_normalized = cv2.minMaxLoc(result_normalized)
        
        # Try edge detection as alternative
        result_edges = cv2.matchTemplate(roi_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val_edges, _, max_loc_edges = cv2.minMaxLoc(result_edges)
        
        # Try alternative template matching methods (different algorithms work better for different images)
        result_ccorr = cv2.matchTemplate(roi_processed, template_processed, cv2.TM_CCORR_NORMED)
        _, max_val_ccorr, _, max_loc_ccorr = cv2.minMaxLoc(result_ccorr)
        
        # Use the best match between normalized and ccorr (both preserve structure)
        # ccorr can sometimes work better for images with varying brightness
        if max_val_ccorr > max_val_normalized:
            max_val = max_val_ccorr
            max_loc = max_loc_ccorr
            result = result_ccorr
            method_used = "normalized_ccorr"
        elif max_val_normalized >= max_val_edges:
            max_val = max_val_normalized
            max_loc = max_loc_normalized
            result = result_normalized
            method_used = "normalized_grayscale"
        else:
            max_val = max_val_edges
            max_loc = max_loc_edges
            result = result_edges
            method_used = "edge_detection"
        
        # print(f"[Minimap State Sensor] Match scores - normalized: {max_val_normalized:.3f}, ccorr: {max_val_ccorr:.3f}, edges: {max_val_edges:.3f}, using: {method_used} ({max_val:.3f})")
        
        # Normalize match result to 0-255 range for visualization
        match_result_viz = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        match_result_viz_bgr = cv2.cvtColor(match_result_viz, cv2.COLOR_GRAY2BGR)
        # Draw match location
        cv2.circle(match_result_viz_bgr, max_loc, 5, (0, 255, 0), 2)
        self.register_debug_output("minimap_precheck_match_result", match_result_viz_bgr)
        
        # Create annotated visualization with sensor data
        # Use configured threshold (edge-based matching typically needs lower thresholds)
        threshold = self.template_match_threshold
        match_passed = max_val >= threshold
        status_color = (0, 255, 0) if match_passed else (0, 0, 255)
        status_text = "PASS" if match_passed else "FAIL"
        
        # Create visualization image showing match data
        # Use the winning method's image for display
        if method_used in ("normalized_grayscale", "normalized_ccorr"):
            roi_display = roi_processed_bgr
        else:
            roi_display = roi_edges_bgr
        viz_height = max(roi_display.shape[0], 150)
        viz_width = roi_display.shape[1]
        sensor_data_viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
        
        # Place processed ROI on top
        sensor_data_viz[:roi_display.shape[0], :roi_display.shape[1]] = roi_display
        
        # Add text annotations with sensor data
        y_offset = roi_display.shape[0] + 20
        cv2.putText(sensor_data_viz, f"Match Value: {max_val:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(sensor_data_viz, f"Threshold: {threshold:.3f}", (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(sensor_data_viz, f"Status: {status_text}", (10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(sensor_data_viz, f"Match Location: {max_loc}", (10, y_offset + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sensor_data_viz, f"Method: {method_used}", (10, y_offset + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        self.register_debug_output("minimap_precheck_sensor_data", sensor_data_viz)
        
        return match_passed
    
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

        # print(f"Blue Ratio: {blue_ratio:.3f}, Red Ratio: {red_ratio:.3f}")
        # print(f"Blue Count: {blue_count}, Red Count: {red_count}")
        # print(f"Total Border Pixels: {total_border_pixels}")    
        # Threshold for detection (at least 5% of border pixels must match)
        threshold = 0.005
        
        # Determine detected state
        detected_state = None
        if blue_ratio > threshold and blue_ratio > red_ratio:
            detected_state = "movement"
        elif red_ratio > threshold and red_ratio > blue_ratio:
            detected_state = "hostile_detected"
        
        # Create sensor data visualization
        viz_height = 200
        viz_width = max(masked_frame.shape[1], 400)
        sensor_data_viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
        
        # Add text annotations with sensor data
        y_offset = 20
        cv2.putText(sensor_data_viz, f"Blue Count: {blue_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(sensor_data_viz, f"Red Count: {red_count}", (10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(sensor_data_viz, f"Total Border Pixels: {total_border_pixels}", (10, y_offset + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sensor_data_viz, f"Blue Ratio: {blue_ratio:.3f}", (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(sensor_data_viz, f"Red Ratio: {red_ratio:.3f}", (10, y_offset + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(sensor_data_viz, f"Threshold: {threshold:.3f}", (10, y_offset + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show detected state with color
        state_color = (0, 255, 0) if detected_state else (128, 128, 128)
        state_text = f"Detected State: {detected_state}" if detected_state else "Detected State: None"
        cv2.putText(sensor_data_viz, state_text, (10, y_offset + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        self.register_debug_output("frame_color_sensor_data", sensor_data_viz)
        
        return detected_state
    
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
