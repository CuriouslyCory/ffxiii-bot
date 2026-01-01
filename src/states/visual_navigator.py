import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple, Dict
from .movement.constants import (
    PHASE_CORR_CONFIDENCE_THRESHOLD,
    ARRIVAL_DISTANCE_THRESHOLD,
    ARRIVAL_ANGLE_THRESHOLD,
    ARRIVAL_BUFFER_SIZE,
    ENABLE_ADVANCED_STITCHING,
    ENABLE_MESH_WARPING,
    SUPERPOINT_N_FEATURES,
    SUPERPOINT_DEVICE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_DEVICE,
    MAGSAC_METHOD,
    MAGSAC_THRESHOLD,
    MAGSAC_CONFIDENCE,
    MAGSAC_MAX_ITERS,
    BUNDLE_ADJUSTMENT_MAX_ITERS,
    BUNDLE_ADJUSTMENT_FTOL,
    BUNDLE_ADJUSTMENT_XTOL,
    BUNDLE_ADJUSTMENT_GTOL,
    MESH_WARPING_GRID_SIZE,
    MESH_WARPING_BLEND_RADIUS,
    RESOLUTION
)

class VisualNavigator:
    """
    Handles Visual Odometry and Hybrid Navigation logic.
    Separated from MovementState to keep file sizes manageable.
    
    Key Features:
    - Hybrid Visual Odometry: Combines visual features (ORB) and phase correlation for navigation
    - Minimap Processing: Extracts, stretches (400x520), and filters minimap images
    - North-Aligned SLAM: Stitches minimap images in consistent North-up orientation
    - Dynamic Circle Masking: Removes transient elements (enemies, allies) from feature matching
    - Gold Arrow Detection: Identifies player orientation using contour analysis and PCA
    
    Image Processing Pipeline:
    1. Raw minimap ROI (400x400) is extracted from screen
    2. Image is stretched to 400x520 to correct UI geometry (ONLY transformation during recording)
    3. Circular mask is applied to isolate minimap region
    4. Filtering: HSV color filters + Canny edge detection (for ORB matching)
    5. Dynamic circles (enemies/allies) are masked out
    6. For phase correlation: North-alignment via rotation, then 225x225 crop
    
    Coordinate Systems:
    - Raw ROI: 400x400 screen-space (before stretch)
    - Stretched: 400x520 (after 30% vertical stretch) - this is what's saved
    - Circular Crop: 400x400 region extracted from stretched image (y: 60-460)
    - Phase Correlation Crop: 225x225 centered in circular crop
    
    Filtering Consistency:
    - _filter_blue_colors: Primary filter for ORB feature matching (includes edge detection)
    - _filter_colors_only: Used ONLY for HSV debug window (no edge detection)
    - Debug preview uses _filter_blue_colors to show exactly what ORB sees
    
    Configuration:
    All magic numbers and tunable parameters are defined in __init__ for easy adjustment.
    See the CONFIGURATION PARAMETERS section for details.
    """
    
    def __init__(self, vision_engine):
        self.vision = vision_engine
        
        # ============================================================================
        # CONFIGURATION PARAMETERS
        # All magic numbers and tunable parameters are defined here for easy adjustment
        # ============================================================================
        
        # --- Image Dimensions ---
        self.MINIMAP_ROI_WIDTH = 400
        self.MINIMAP_ROI_HEIGHT = 400
        self.MINIMAP_STRETCHED_WIDTH = 400
        self.MINIMAP_STRETCHED_HEIGHT = 520
        self.CIRCULAR_CROP_SIZE = 400  # For 400x400 circular region extraction
        self.PHASE_CORR_CROP_SIZE = 225  # For phase correlation calculation
        self.MASTER_MAP_CROP_SIZE = 250  # For master map generation
        
        # --- Coordinates (relative to dimensions above) ---
        self.MINIMAP_CENTER_X = 200  # Before stretch
        self.MINIMAP_CENTER_Y = 200  # Before stretch
        self.MINIMAP_CENTER_STRETCHED_X = 200
        self.MINIMAP_CENTER_STRETCHED_Y = 260
        self.MINIMAP_RADIUS = 150  # Aggressively reduced to eliminate border artifacts (was 190, then 170)
        self.CIRCULAR_CROP_Y1 = 60  # For extracting 400x400 from 400x520
        self.CIRCULAR_CROP_Y2 = 460
        
        # --- HSV Filter Ranges (for map feature detection) ---
        # Blue filter (normal minimap state)
        self.blue_lower = [84, 75, 100]
        self.blue_upper = [97, 245, 245]
        
        # Alert/Red filter (enemy nearby state - two ranges for HSV wheel wrap)
        self.alert_lower = [0, 40, 50]
        self.alert_upper = [13, 255, 255]
        self.red_lower = [170, 40, 50]
        self.red_upper = [180, 255, 255]
        
        # Gold filter (player arrow indicator)
        self.gold_lower = [15, 100, 150]
        self.gold_upper = [45, 255, 255]
        self.gold_arrow_lower = [15, 150, 150]  # Higher saturation for arrow detection
        self.gold_arrow_upper = [45, 255, 255]
        
        # --- Dynamic Circle Masking (for enemy dots and AI team circles) ---
        # NOTE: HSV ranges must match or be broader than main color filter ranges
        # to ensure red dots that pass color filter can be detected and masked
        self.CIRCLE_MIN_RADIUS = 7  # Reduced to catch smaller enemy dots
        self.CIRCLE_MAX_RADIUS = 13
        # Use same or broader ranges as main alert/red filter to catch all red dots
        self.CIRCLE_DETECTION_RED_LOWER1 = [0, 40, 50]  # Match alert_lower
        self.CIRCLE_DETECTION_RED_UPPER1 = [30, 255, 255]  # Match alert_upper
        self.CIRCLE_DETECTION_RED_LOWER2 = [170, 40, 50]  # Match red_lower
        self.CIRCLE_DETECTION_RED_UPPER2 = [180, 255, 255]  # Match red_upper
        # Expanded blue/cyan range for teammate circles (broader to catch variations)
        self.CIRCLE_DETECTION_BLUE_LOWER = [90, 80, 80]  # Expanded from [100, 100, 100]
        self.CIRCLE_DETECTION_BLUE_UPPER = [140, 255, 255]  # Expanded from [130, 255, 255]
        # Also detect cyan/light blue (common for teammate indicators)
        self.CIRCLE_DETECTION_CYAN_LOWER = [85, 60, 60]
        self.CIRCLE_DETECTION_CYAN_UPPER = [100, 255, 255]
        self.CIRCLE_DETECTION_WHITE_LOWER = [0, 0, 180]  # Lowered from 200 to catch more white/light centers
        self.CIRCLE_DETECTION_WHITE_UPPER = [180, 50, 255]  # Increased saturation tolerance from 30 to 50
        self.CIRCLE_MASK_PADDING = 4  # Pixels to add around detected circles
        
        # --- Edge Detection Parameters ---
        self.CANNY_OTSU_LOW_FACTOR = 0.5
        self.CANNY_OTSU_HIGH_FACTOR = 1.5
        self.CANNY_MIN_LOW = 30
        self.CANNY_MAX_HIGH = 200
        self.EDGE_DILATE_KERNEL = (2, 2)
        self.EDGE_DILATE_ITERATIONS = 1
        
        # --- Center Masking (to remove player arrow and static UI) ---
        self.CENTER_MASK_RADIUS_PHASE_CORR = 35
        self.CENTER_MASK_RADIUS_DEFAULT = 60
        self.CENTER_MASK_RADIUS_MASTER_MAP = 16  # Reduced by 60% (from 40 to 16) to shrink center void
        self.ARROW_CENTER_MASK_RADIUS = 80
        
        # --- Phase Correlation ---
        # Using shared constant from movement.constants
        # PHASE_CORR_CONFIDENCE_THRESHOLD imported from constants
        
        # --- Feature Matching Thresholds ---
        self.MIN_INLIERS_FOR_TRACKING = 8
        self.MIN_INLIERS_FOR_RECOVERY = 15
        # Using shared constants from movement.constants
        # ARRIVAL_DISTANCE_THRESHOLD, ARRIVAL_ANGLE_THRESHOLD, ARRIVAL_BUFFER_SIZE imported from constants
        
        # --- Debug Visualization Dimensions ---
        self.DEBUG_CTRL_HEIGHT = 150
        self.DEBUG_HIST_HEIGHT = 120
        self.DEBUG_CHART_HEIGHT = 120
        self.DEBUG_CROP_PREVIEW_HEIGHT = 250
        self.DEBUG_HIST_BINS = 30
        self.DEBUG_HIST_PADDING = 30
        self.DEBUG_HIST_BOTTOM_PADDING = 20
        self.DEBUG_CHART_PADDING = 10
        self.DEBUG_CHART_Y_AXIS_WIDTH = 30
        self.DEBUG_CROP_PREVIEW_TOP_PADDING = 30
        self.DEBUG_CROP_PREVIEW_BOTTOM_PADDING = 30
        self.DEBUG_CANVAS_SPACING = 20
        self.DEBUG_PADDING = 20
        
        # --- Histogram/Chart Ranges ---
        self.HIST_DX_MIN = -500
        self.HIST_DX_MAX = 500
        self.HIST_DY_MIN = -500
        self.HIST_DY_MAX = 500
        self.HIST_ANGLE_MIN = -180
        self.HIST_ANGLE_MAX = 180
        
        # --- Color Constants (BGR format) ---
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_ORANGE = (0, 165, 255)
        self.COLOR_CYAN = (0, 255, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GRAY_LIGHT = (200, 200, 200)
        self.COLOR_GRAY_MEDIUM = (100, 100, 100)
        self.COLOR_GRAY_DARK = (30, 30, 30)
        self.COLOR_BLACK = (0, 0, 0)
        
        # --- Calibration ---
        self.CALIB_TEMPLATE_THRESHOLD = 0.3
        
        # --- Arrow Detection ---
        self.ARROW_MIN_CONTOUR_AREA = 5
        
        # --- Master Map Generation ---
        self.MASTER_MAP_PADDING = 200
        self.MASTER_MAP_MIN_SIZE = 500
        self.MASTER_MAP_MAX_SIZE = 10000
        self.MASTER_MAP_DOWNSAMPLE_FACTOR = 0.5
        self.MASTER_MAP_CIRCLE_BORDER = 10
        
        # ============================================================================
        # RUNTIME STATE (Initialized from configuration)
        # ============================================================================
        
        # Default Minimap configuration (Approximate for 1080p, needs calibration)
        # Raw screen-space ROI (400x400)
        self.minimap_roi = (1530, 0, self.MINIMAP_ROI_WIDTH, self.MINIMAP_ROI_HEIGHT)  # x, y, w, h
        self.minimap_center = (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y)  # relative to 400x400 ROI (before stretch)
        self.minimap_center_stretched = (self.MINIMAP_CENTER_STRETCHED_X, self.MINIMAP_CENTER_STRETCHED_Y)  # relative to 400x520 (after stretch)
        self.minimap_radius = self.MINIMAP_RADIUS
        
        # Initialize mask
        self._update_mask()
        
        # Odometry State
        self.last_dx = 0
        self.last_dy = 0
        self.last_angle = 0
        self.last_matches = []
        self.last_kp1 = []
        self.last_kp2 = []
        self.last_filtered_target = None
        self.last_filtered_current = None
        
        # Cropped images for dx/dy calculation preview
        self.last_cropped_curr = None
        self.last_cropped_prev = None
        
        # Histogram tracking (rolling buffer for real-time histograms)
        self.hist_buffer_size = 200
        self.dx_history = []
        self.dy_history = []
        self.angle_history = []
        
        # Phase correlation history (for averaging when offsets are invalid)
        self.dx_pc_history = []
        self.dy_pc_history = []
        self.pc_history_size = 10  # Keep last 10 values for averaging
        
        # Decay tracking for averaged values
        self.consecutive_invalid_count = 0  # Track consecutive invalid readings
        self.average_decay_factor = 0.95  # Decay factor per invalid reading (5% reduction)
        
        # Angle unwrapping - track last angle to prevent flip-flopping
        self.last_unwrapped_angle = None  # Last unwrapped angle value
        # Track angle offset for calibration (persistent offset between gold arrow and homography)
        self.angle_offset_estimate = 0.0  # Estimated systematic offset
        self.angle_offset_samples = []  # Samples for offset estimation
        self.angle_offset_max_samples = 30  # Number of samples to use for offset estimation
        
        # Debugging
        self.debug_window_name = "Visual Odometry"
        self.calibrated = False
        
        # HSV Debug Mode
        self.hsv_debug_enabled = False
        self.hsv_debug_window_name = "HSV Filter Debug"

    def _update_mask(self):
        """Updates circular masks for both raw and stretched minimap images."""
        # Mask for raw 400x400 screen-space images
        self.mask = np.zeros((self.MINIMAP_ROI_HEIGHT, self.MINIMAP_ROI_WIDTH), dtype=np.uint8)
        cv2.circle(self.mask, self.minimap_center, self.minimap_radius, (255), -1)
        
        # Mask for stretched 400x520 images (circular region)
        self.mask_stretched = np.zeros((self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH), dtype=np.uint8)
        cv2.circle(self.mask_stretched, self.minimap_center_stretched, self.minimap_radius, (255), -1)

    def calibrate(self, image: np.ndarray) -> bool:
        """
        Attempts to auto-calibrate the minimap ROI using the minimap_outline template.
        """
        # Search in the top-right quadrant
        h, w = image.shape[:2]
        search_roi = (w // 2, 0, w // 2, h // 2)
        
        # Assuming 'minimap_outline' template is loaded in vision engine
        if "minimap_outline" not in self.vision.templates:
            print("[CALIB] 'minimap_outline' template not loaded.")
            return False

        match = self.vision.find_template("minimap_outline", image, threshold=self.CALIB_TEMPLATE_THRESHOLD, roi=search_roi)
        
        if match:
            mx, my, conf = match
            template = self.vision.templates["minimap_outline"]
            th, tw = template.shape[:2]
            
            # Center of the match
            center_x = mx + tw // 2
            center_y = my + th // 2
            
            # Define new ROI centered on this
            roi_w, roi_h = self.MINIMAP_ROI_WIDTH, self.MINIMAP_ROI_HEIGHT
            new_x = max(0, center_x - roi_w // 2)
            new_y = max(0, center_y - roi_h // 2)
            
            # Ensure within bounds
            if new_x + roi_w > w: new_x = w - roi_w
            if new_y + roi_h > h: new_y = h - roi_h
            
            self.minimap_roi = (new_x, new_y, roi_w, roi_h)
            self.minimap_center = (roi_w // 2, roi_h // 2)
            
            self._update_mask()
            self.calibrated = True
            print(f"[CALIB] Minimap ROI calibrated to: {self.minimap_roi}")
            return True
        else:
            print("[CALIB] Failed to find minimap_outline.")
            return False
            
    def get_minimap_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns the stretched 400x520 crop of the minimap ROI.
        Stretching happens FIRST to correct UI geometry before any rotation.
        """
        if not self.calibrated:
             self.calibrate(image)
             self.calibrated = True 

        roi_img = self.vision.get_roi_slice(image, self.minimap_roi)
        
        # Safety check if ROI is valid for image
        if roi_img.shape[0] != self.minimap_roi[3] or roi_img.shape[1] != self.minimap_roi[2]:
            return None
        
        # Stretch vertically by 30% FIRST (before rotation)
        # NOTE: Saved images are always stretched (400x520) - this is the only transformation applied during recording
        stretched = cv2.resize(roi_img, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT), interpolation=cv2.INTER_LINEAR)
        return stretched

    def extract_minimap(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts and masks the minimap from the full screen image.
        Returns stretched 400x520 image with circular mask applied.
        """
        roi_img = self.get_minimap_crop(image)
        if roi_img is None:
            return None
            
        # Apply circular mask (400x520 stretched)
        masked = cv2.bitwise_and(roi_img, roi_img, mask=self.mask_stretched)
        return masked

    def _create_hsv_color_mask(self, hsv: np.ndarray, include_gold: bool = True) -> np.ndarray:
        """
        Creates a combined HSV color mask for map features (blue, alert/red, optionally gold).
        This is the core color filtering logic used by both _filter_blue_colors and _filter_colors_only.
        
        Args:
            hsv: HSV image
            include_gold: Whether to include gold filter (for arrow detection)
        
        Returns:
            Combined color mask
        """
        # Primary: Blue/cyan range (normal minimap state)
        lower_blue = np.array(self.blue_lower)
        upper_blue = np.array(self.blue_upper)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Secondary: Red/orange range (alert/enemy nearby state)
        # Many games shift minimap to red/orange when enemies are detected
        lower_alert = np.array(self.alert_lower)  # Red (wraps around)
        upper_alert = np.array(self.alert_upper)  # Orange
        mask_alert = cv2.inRange(hsv, lower_alert, upper_alert)
        
        # Also catch red on the other side of HSV wheel
        lower_red = np.array(self.red_lower)
        upper_red = np.array(self.red_upper)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine all color ranges
        mask_colors = cv2.bitwise_or(mask_blue, mask_alert)
        mask_colors = cv2.bitwise_or(mask_colors, mask_red)

        if include_gold:
            # Gold filter for arrow
            lower_gold = np.array(self.gold_lower)
            upper_gold = np.array(self.gold_upper)
            mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
            mask_colors = cv2.bitwise_or(mask_colors, mask_gold)

        return mask_colors

    def _apply_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Applies adaptive Canny edge detection to the image.
        This supplements color filtering to preserve map structure during color shifts.
        
        Args:
            image: Input BGR image
        
        Returns:
            Edge-filtered BGR image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding to handle varying brightness
        # Use Otsu's method for automatic threshold selection
        otsu_val = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        
        # Canny edge detection with adaptive thresholds
        # Lower threshold is 50% of Otsu threshold, upper is 150%
        canny_low = max(self.CANNY_MIN_LOW, int(otsu_val * self.CANNY_OTSU_LOW_FACTOR))
        canny_high = min(self.CANNY_MAX_HIGH, int(otsu_val * self.CANNY_OTSU_HIGH_FACTOR))
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Dilate edges slightly to make them more visible
        edges = cv2.dilate(edges, np.ones(self.EDGE_DILATE_KERNEL, np.uint8), iterations=self.EDGE_DILATE_ITERATIONS)
        
        # Create edge mask and apply to original image
        edge_mask = edges.astype(np.uint8)
        edge_result = cv2.bitwise_and(image, image, mask=edge_mask)
        
        return edge_result

    def _filter_blue_colors(self, image: np.ndarray, include_gold: bool = True) -> np.ndarray:
        """
        Filters the image to isolate map features regardless of color shifts.
        Handles both normal blue minimap and alert-state (enemy nearby) color changes.
        This is the PRIMARY filter used for ORB feature matching - includes edge detection.
        
        Returns the filtered BGR image (mostly black with colored lines and edges).
        This is what ORB actually sees, so preview should use this method too.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create color mask using shared helper
        mask_colors = self._create_hsv_color_mask(hsv, include_gold=include_gold)

        # Apply color mask
        result = cv2.bitwise_and(image, image, mask=mask_colors)
        
        # Mask out dynamic circles (red enemy dots and blue AI team circles)
        circle_mask = self._mask_dynamic_circles(image)
        result = cv2.bitwise_and(result, result, mask=circle_mask)
        
        # Always supplement with edge detection for robustness
        # This ensures map structure is preserved even during severe color shifts
        edge_result = self._apply_edge_detection(image)
        
        # Combine color filter and edge detection (use max to preserve brightest features)
        result = cv2.max(result, edge_result)
        
        return result

    def _filter_colors_only(self, image: np.ndarray, include_gold: bool = True) -> np.ndarray:
        """
        Filters the image using only HSV color filters (no edge detection).
        This is used ONLY for the HSV debug window to show color filters separately.
        For actual feature matching, use _filter_blue_colors instead.
        
        Returns the filtered BGR image with only color-filtered pixels visible.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create color mask using shared helper
        mask_colors = self._create_hsv_color_mask(hsv, include_gold=include_gold)

        # Apply color mask only (no edge detection)
        result = cv2.bitwise_and(image, image, mask=mask_colors)
        
        # Mask out dynamic circles (red enemy dots and blue AI team circles)
        circle_mask = self._mask_dynamic_circles(image)
        result = cv2.bitwise_and(result, result, mask=circle_mask)
        
        return result

    def _mask_dynamic_circles(self, image: np.ndarray, min_radius: int = None, max_radius: int = None) -> np.ndarray:
        """
        Detects and masks out small circular features (red enemy dots and blue AI team circles).
        These are dynamic elements that shouldn't be part of feature matching.
        
        Args:
            image: Input BGR image
            min_radius: Minimum circle radius to detect (defaults to config value)
            max_radius: Maximum circle radius to detect (defaults to config value)
        
        Returns:
            A mask where detected circles are set to 0 (to be masked out), everything else is 255
        """
        if min_radius is None:
            min_radius = self.CIRCLE_MIN_RADIUS
        if max_radius is None:
            max_radius = self.CIRCLE_MAX_RADIUS
            
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect red dots (enemies)
        # Red can be on both sides of HSV wheel
        lower_red1 = np.array(self.CIRCLE_DETECTION_RED_LOWER1)
        upper_red1 = np.array(self.CIRCLE_DETECTION_RED_UPPER1)
        lower_red2 = np.array(self.CIRCLE_DETECTION_RED_LOWER2)
        upper_red2 = np.array(self.CIRCLE_DETECTION_RED_UPPER2)
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Detect blue circles (AI team members - blue with white centers)
        # Blue range (expanded)
        lower_blue = np.array(self.CIRCLE_DETECTION_BLUE_LOWER)
        upper_blue = np.array(self.CIRCLE_DETECTION_BLUE_UPPER)
        mask_blue_circles = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also detect cyan/light blue (common for teammate indicators)
        lower_cyan = np.array(self.CIRCLE_DETECTION_CYAN_LOWER)
        upper_cyan = np.array(self.CIRCLE_DETECTION_CYAN_UPPER)
        mask_cyan_circles = cv2.inRange(hsv, lower_cyan, upper_cyan)
        
        # Also detect white/light centers (high value, low saturation)
        lower_white = np.array(self.CIRCLE_DETECTION_WHITE_LOWER)
        upper_white = np.array(self.CIRCLE_DETECTION_WHITE_UPPER)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all teammate color masks
        mask_teammate_colors = cv2.bitwise_or(mask_blue_circles, mask_cyan_circles)
        mask_teammate_colors = cv2.bitwise_or(mask_teammate_colors, mask_white)
        
        # Combine red and teammate masks
        mask_circles = cv2.bitwise_or(mask_red, mask_teammate_colors)
        
        # Apply morphological operations to better connect circle pixels
        # This helps with teammate circles that might have gaps or be partially visible
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_circles = cv2.morphologyEx(mask_circles, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_circles = cv2.morphologyEx(mask_circles, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Use HoughCircles to detect circular shapes
        # Apply to the combined mask with more lenient parameters for teammate detection
        circles = cv2.HoughCircles(
            mask_circles,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_radius * 2,
            param1=50,
            param2=13,  # Lowered from 20 to be more sensitive (detect more circles)
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw filled circle on mask to exclude this area
                cv2.circle(mask, (x, y), r + self.CIRCLE_MASK_PADDING, 0, -1)  # Padding for mask out
        
        # Enhanced contour detection for more robust circle detection
        # This helps catch circles that HoughCircles might miss, especially teammate circles
        contours, _ = cv2.findContours(mask_circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (roughly circular area for our size range)
            min_area = np.pi * min_radius * min_radius * 0.5  # Lowered threshold to catch smaller circles
            max_area = np.pi * max_radius * max_radius * 1.5  # Increased to catch slightly larger circles
            
            if min_area <= area <= max_area:
                # Check circularity (lowered threshold to be more permissive)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Circularity of 1.0 is a perfect circle, lowered from 0.7 to 0.6 for teammate circles
                    if circularity > 0.6:
                        # Get bounding circle
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        # Expanded radius range to catch more circles
                        if min_radius * 0.8 <= radius <= max_radius * 1.2:
                            # Mask out this circle with extra padding for teammate circles
                            cv2.circle(mask, (int(x), int(y)), int(radius) + self.CIRCLE_MASK_PADDING + 1, 0, -1)
        
        # Additional pass: detect circles in grayscale for teammate dots that might not be strongly colored
        # This catches circles that are visible but might not match the color ranges perfectly
        # Use adaptive threshold to better detect circles with varying brightness
        thresh_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
        # Apply circular mask to focus on minimap region
        if hasattr(self, 'mask_stretched') and thresh_gray.shape == self.mask_stretched.shape:
            thresh_gray = cv2.bitwise_and(thresh_gray, self.mask_stretched)
        
        # Apply morphological operations to clean up and connect circle pixels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Find contours in grayscale that might be circles
        contours_gray, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_gray:
            area = cv2.contourArea(contour)
            min_area = np.pi * min_radius * min_radius * 0.5
            max_area = np.pi * max_radius * max_radius * 1.5
            
            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # More lenient for grayscale detection, but still require reasonable circularity
                    if circularity > 0.6:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if min_radius * 0.8 <= radius <= max_radius * 1.2:
                            # Check if this area is already masked (avoid double masking)
                            center_x, center_y = int(x), int(y)
                            if 0 <= center_x < w and 0 <= center_y < h:
                                if mask[center_y, center_x] == 255:  # Not yet masked
                                    # Additional validation: check if center region matches teammate colors
                                    # Sample a small region around the center
                                    y1, y2 = max(0, center_y - 2), min(h, center_y + 3)
                                    x1, x2 = max(0, center_x - 2), min(w, center_x + 3)
                                    center_region_hsv = hsv[y1:y2, x1:x2]
                                    if center_region_hsv.size > 0:
                                        # Check if any pixels in center region match teammate colors
                                        mask_blue_check = cv2.inRange(center_region_hsv, lower_blue, upper_blue)
                                        mask_cyan_check = cv2.inRange(center_region_hsv, lower_cyan, upper_cyan)
                                        mask_white_check = cv2.inRange(center_region_hsv, lower_white, upper_white)
                                        if (np.any(mask_blue_check > 0) or np.any(mask_cyan_check > 0) or 
                                            np.any(mask_white_check > 0)):
                                            # Likely a teammate circle - mask it out
                                            cv2.circle(mask, (center_x, center_y), int(radius) + self.CIRCLE_MASK_PADDING, 0, -1)
        
        return mask

    def _draw_histogram(self, canvas: np.ndarray, x: int, y: int, width: int, height: int, 
                       values: list, label: str, color: Tuple[int, int, int], 
                       min_val: float = None, max_val: float = None):
        """
        Draws a histogram on the canvas at the specified position.
        
        Args:
            canvas: The canvas to draw on
            x, y: Top-left corner position
            width, height: Dimensions of the histogram area
            values: List of values to histogram
            label: Label text for the histogram
            color: BGR color tuple for the bars
            min_val, max_val: Optional min/max for the x-axis range
        """
        if not values:
            return
        
        # Calculate histogram bins
        num_bins = self.DEBUG_HIST_BINS
        if min_val is None:
            min_val = min(values)
        if max_val is None:
            max_val = max(values)
        
        # Avoid division by zero
        if max_val == min_val:
            max_val = min_val + 1
        
        # Create histogram
        bins = np.linspace(min_val, max_val, num_bins + 1)
        hist, _ = np.histogram(values, bins=bins)
        
        # Normalize histogram to fit in height (leave some padding)
        max_count = max(hist) if max(hist) > 0 else 1
        bar_height_scale = (height - self.DEBUG_HIST_PADDING) / max_count
        
        # Draw histogram bars
        bin_width = width / num_bins
        bottom_y = y + height - self.DEBUG_HIST_BOTTOM_PADDING
        for i in range(num_bins):
            bar_height = int(hist[i] * bar_height_scale)
            if bar_height > 0:
                x1 = int(x + i * bin_width)
                y1 = bottom_y - bar_height
                x2 = int(x + (i + 1) * bin_width)
                y2 = bottom_y
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        
        # Draw axes
        cv2.line(canvas, (x, bottom_y), (x + width, bottom_y), self.COLOR_GRAY_LIGHT, 1)  # X-axis
        cv2.line(canvas, (x, y), (x, bottom_y), self.COLOR_GRAY_LIGHT, 1)  # Y-axis
        
        # Draw label
        cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        
        # Draw min/max values
        cv2.putText(canvas, f"{min_val:.1f}", (x, y + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_GRAY_LIGHT, 1)
        max_text = f"{max_val:.1f}"
        text_size = cv2.getTextSize(max_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(canvas, max_text, (x + width - text_size[0], y + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_GRAY_LIGHT, 1)
        
        # Draw current value as a vertical line
        if values:
            current_val = values[-1]
            if min_val <= current_val <= max_val:
                norm_x = int(x + ((current_val - min_val) / (max_val - min_val)) * width)
                cv2.line(canvas, (norm_x, y), (norm_x, bottom_y), self.COLOR_CYAN, 2)

    def _draw_line_chart(self, canvas: np.ndarray, x: int, y: int, width: int, height: int,
                        values: list, label: str, color: Tuple[int, int, int],
                        min_val: float = None, max_val: float = None, zero_line: bool = True):
        """
        Draws a time series line chart on the canvas at the specified position.
        
        Args:
            canvas: The canvas to draw on
            x, y: Top-left corner position
            width, height: Dimensions of the chart area
            values: List of values to plot (time series)
            label: Label text for the chart
            color: BGR color tuple for the line
            min_val, max_val: Optional min/max for the y-axis range
            zero_line: Whether to draw a horizontal line at y=0
        """
        if not values or len(values) < 2:
            return
        
        # Calculate y-axis range
        if min_val is None:
            min_val = min(values)
        if max_val is None:
            max_val = max(values)
        
        # Add some padding to the range
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1
            min_val -= 0.5
            max_val += 0.5
        else:
            padding = range_val * 0.1
            min_val -= padding
            max_val += padding
        
        # Draw axes
        chart_bottom = y + height - self.DEBUG_HIST_BOTTOM_PADDING
        chart_top = y + self.DEBUG_CHART_PADDING
        chart_left = x + self.DEBUG_CHART_Y_AXIS_WIDTH
        chart_right = x + width - self.DEBUG_CHART_PADDING
        
        cv2.line(canvas, (chart_left, chart_top), (chart_left, chart_bottom), self.COLOR_GRAY_LIGHT, 1)  # Y-axis
        cv2.line(canvas, (chart_left, chart_bottom), (chart_right, chart_bottom), self.COLOR_GRAY_LIGHT, 1)  # X-axis
        
        # Draw zero line if requested
        if zero_line and min_val <= 0 <= max_val:
            zero_y = int(chart_bottom - ((0 - min_val) / (max_val - min_val)) * (chart_bottom - chart_top))
            cv2.line(canvas, (chart_left, zero_y), (chart_right, zero_y), self.COLOR_GRAY_MEDIUM, 1)
        
        # Draw the line
        num_points = len(values)
        if num_points > 1:
            points = []
            for i, val in enumerate(values):
                # Normalize x position (time)
                norm_x = chart_left + int((i / (num_points - 1)) * (chart_right - chart_left))
                # Normalize y position (value)
                norm_y = int(chart_bottom - ((val - min_val) / (max_val - min_val)) * (chart_bottom - chart_top))
                # Clamp to chart bounds
                norm_y = max(chart_top, min(chart_bottom, norm_y))
                points.append((norm_x, norm_y))
            
            # Draw the line connecting all points
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], color, 2)
            
            # Draw the most recent point as a circle
            if points:
                cv2.circle(canvas, points[-1], 3, color, -1)
        
        # Draw label
        cv2.putText(canvas, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1)
        
        # Draw min/max values on y-axis
        cv2.putText(canvas, f"{max_val:.1f}", (x, chart_top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_GRAY_LIGHT, 1)
        min_text = f"{min_val:.1f}"
        text_size = cv2.getTextSize(min_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(canvas, min_text, (x, chart_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLOR_GRAY_LIGHT, 1)
        
        # Draw current value
        if values:
            current_val = values[-1]
            current_text = f"{current_val:.1f}"
            text_size = cv2.getTextSize(current_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(canvas, current_text, (chart_right - text_size[0] - 5, chart_top + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _unwrap_angle(self, angle: float, current_img: np.ndarray = None, 
                     target_arrow_angle: Optional[float] = None) -> float:
        """
        Unwrap angle to prevent flip-flopping between positive and negative values.
        
        When gold arrow angle is available, it's used as the primary source since it's
        more stable than homography decomposition. When far from alignment, the gold
        arrow is trusted more heavily to prevent wild oscillations.
        
        Args:
            angle: Raw angle from homography decomposition (in degrees, range -180 to 180).
            current_img: Optional current image to extract gold arrow angle for validation.
            target_arrow_angle: Optional target node arrow angle for validation.
            
        Returns:
            Angle in [-180, 180] range, adjusted to prevent flip-flopping.
        """
        # Try to get expected angle from gold arrow if available
        expected_angle = None
        if current_img is not None and target_arrow_angle is not None:
            try:
                current_minimap = self.extract_minimap(current_img)
                if current_minimap is not None:
                    curr_arrow_angle, _ = self.get_gold_arrow_angle(current_minimap)
                    # Expected angle is the difference between current and target arrow angles
                    # This represents the rotation needed to align current with target
                    expected_angle = curr_arrow_angle - target_arrow_angle
                    # Normalize to [-180, 180]
                    while expected_angle > 180:
                        expected_angle -= 360
                    while expected_angle < -180:
                        expected_angle += 360
            except:
                pass
        
        # If we have gold arrow angle, use it as primary source (it's more stable)
        if expected_angle is not None:
            # Estimate systematic offset between gold arrow and homography
            # This helps correct for calibration issues
            angle_diff_raw = angle - expected_angle
            # Normalize difference
            while angle_diff_raw > 180:
                angle_diff_raw -= 360
            while angle_diff_raw < -180:
                angle_diff_raw += 360
            
            # Track offset samples (only when both angles are reasonable)
            if abs(angle) < 90 and abs(expected_angle) < 90:
                self.angle_offset_samples.append(angle_diff_raw)
                if len(self.angle_offset_samples) > self.angle_offset_max_samples:
                    self.angle_offset_samples.pop(0)
                
                # Update offset estimate as median of recent samples (more robust than mean)
                if len(self.angle_offset_samples) >= 10:
                    sorted_samples = sorted(self.angle_offset_samples)
                    median_idx = len(sorted_samples) // 2
                    self.angle_offset_estimate = sorted_samples[median_idx]
            
            # Apply offset correction to expected angle
            corrected_expected = expected_angle - self.angle_offset_estimate
            # Normalize
            while corrected_expected > 180:
                corrected_expected -= 360
            while corrected_expected < -180:
                corrected_expected += 360
            
            # When gold arrow is available, trust it more, especially when far from alignment
            if self.last_unwrapped_angle is None:
                # First reading - use corrected gold arrow
                self.last_unwrapped_angle = corrected_expected
                return corrected_expected
            
            # Normalize last angle
            last_normalized = self.last_unwrapped_angle
            while last_normalized > 180:
                last_normalized -= 360
            while last_normalized < -180:
                last_normalized += 360
            
            # Calculate how far we are from alignment (using corrected expected)
            abs_expected = abs(corrected_expected)
            
            # When far from alignment (>30 degrees), trust gold arrow almost completely
            # When close to alignment (<30 degrees), blend with homography angle for precision
            if abs_expected > 30:
                # Far from alignment: trust corrected gold arrow 85%, homography 15%
                # Increased homography weight to help correct systematic offsets
                angle_diff = corrected_expected - last_normalized
                # Normalize difference
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                
                # Smooth large changes (limit rate of change)
                max_change = 15.0  # Max degrees per frame
                if abs(angle_diff) > max_change:
                    angle_diff = np.sign(angle_diff) * max_change
                
                result = last_normalized + angle_diff
                # Blend with homography angle (15% - increased from 10% to help with offset correction)
                result = 0.85 * result + 0.15 * angle
            else:
                # Close to alignment: blend corrected gold arrow (60%) with homography (40%)
                # Increased homography weight when close for better precision
                result = 0.7 * corrected_expected + 0.3 * angle
                # Smooth transition from last angle
                angle_diff = result - last_normalized
                while angle_diff > 180:
                    angle_diff -= 360
                while angle_diff < -180:
                    angle_diff += 360
                # Limit rate of change even when close
                max_change = 10.0
                if abs(angle_diff) > max_change:
                    angle_diff = np.sign(angle_diff) * max_change
                result = last_normalized + angle_diff
            
            # Normalize result
            while result > 180:
                result -= 360
            while result < -180:
                result += 360
            
            self.last_unwrapped_angle = result
            return result
        
        # No gold arrow available - fall back to homography angle with unwrapping
        if self.last_unwrapped_angle is None:
            self.last_unwrapped_angle = angle
            return angle
        
        # Normalize last angle
        last_normalized = self.last_unwrapped_angle
        while last_normalized > 180:
            last_normalized -= 360
        while last_normalized < -180:
            last_normalized += 360
        
        # Calculate difference from normalized last angle
        angle_diff = angle - last_normalized
        
        # Normalize difference to [-180, 180] range
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        # If the difference is large (>90 degrees), it's likely a wrap-around
        if abs(angle_diff) > 90:
            if angle_diff > 0:
                angle_diff -= 360
            else:
                angle_diff += 360
        
        # Apply smoothing to limit rate of change (especially important when far from alignment)
        max_change = 20.0  # Max degrees per frame when no gold arrow
        if abs(angle_diff) > max_change:
            angle_diff = np.sign(angle_diff) * max_change
        
        # Calculate result
        result = last_normalized + angle_diff
        
        # Normalize result
        while result > 180:
            result -= 360
        while result < -180:
            result += 360
        
        self.last_unwrapped_angle = result
        return result
    
    def get_gold_arrow_angle(self, image: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """
        Detects the orientation and centroid of the gold arrow indicator.
        Works on stretched 400x520 images. Arrow center is at (200, 260).
        Returns (angle_deg, centroid_xy). Angle 0 is Up, positive is CW.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # High-Saturation Gold Filter (uses higher saturation threshold for arrow detection)
        lower_gold = np.array(self.gold_arrow_lower)
        upper_gold = np.array(self.gold_arrow_upper)
        mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
        
        h, w = image.shape[:2]
        # Center mask around expected arrow position (200, 260 in stretched space)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, self.minimap_center_stretched, self.ARROW_CENTER_MASK_RADIUS, (255), -1)
        mask_gold = cv2.bitwise_and(mask_gold, center_mask)
        
        contours, _ = cv2.findContours(mask_gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, self.minimap_center_stretched
            
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < self.ARROW_MIN_CONTOUR_AREA:
            return 0.0, self.minimap_center_stretched
            
        # Find centroid
        M = cv2.moments(main_contour)
        if M["m00"] == 0: return 0.0, self.minimap_center_stretched
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)

        # PCA for orientation
        points = main_contour.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=1)
        
        vX, vY = eigenvectors[0][0], eigenvectors[0][1]
        
        # Find tip (furthest point from centroid)
        max_d = -1
        tip = centroid
        for pt in main_contour:
            px, py = pt[0]
            d = (px - cX)**2 + (py - cY)**2
            if d > max_d:
                max_d = d
                tip = (px, py)
        
        dot = (tip[0] - cX) * vX + (tip[1] - cY) * vY
        if dot < 0: vX, vY = -vX, -vY
            
        # Calculate angle from eigenvector to match homography convention
        # The eigenvector (vX, vY) points in the direction of the arrow tip
        # In image coordinates: X increases right, Y increases down
        # 
        # Convention: 0° = Up, positive = Clockwise (CW)
        # When arrow points up: vX ≈ 0, vY < 0 (negative Y = up in image coords)
        # When arrow points right: vX > 0, vY ≈ 0 (right is 90° CW from up, so should be -90°)
        #
        # To convert direction vector to rotation angle:
        # We want the angle that rotates (0, -1) [up vector] to (vX, vY)
        # This is: angle = arctan2(vX, -vY)
        #   Up (vX=0, vY<0): arctan2(0, -(-1)) = arctan2(0, 1) = 0° ✓
        #   Right (vX>0, vY=0): arctan2(1, 0) = 90° (but we want -90° for CW)
        #
        # So we need to negate: angle = -arctan2(vX, -vY)
        #   Up: -arctan2(0, 1) = 0° ✓
        #   Right: -arctan2(1, 0) = -90° ✓
        angle_rad = np.arctan2(vX, -vY)
        angle_deg = -np.degrees(angle_rad)  # Negate to get CW positive convention
        
        return angle_deg, centroid

    def mask_center(self, image: np.ndarray, radius: int = None) -> np.ndarray:
        """
        Masks out the center of the image (player arrow and static UI).
        
        Args:
            image: Input image to mask
            radius: Mask radius (defaults to CENTER_MASK_RADIUS_DEFAULT)
        """
        if radius is None:
            radius = self.CENTER_MASK_RADIUS_DEFAULT
        h, w = image.shape[:2]
        masked = image.copy()
        cv2.circle(masked, (w//2, h//2), radius, self.COLOR_BLACK, -1)
        return masked

    def compute_drift_pc(self, current_img: np.ndarray, prev_img: np.ndarray, 
                         curr_arrow_angle: float, curr_pivot: Tuple[int, int],
                         prev_arrow_angle: float, prev_pivot: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Computes World-Space drift (dX, dY) using Phase Correlation on North-Aligned images.
        Images are already stretched (400x520). Rotates around arrow pivots, then extracts
        400x400 circular region for matching.
        Returns (world_dx, world_dy, conf)
        """
        h, w = current_img.shape[:2]  # Should be 520, 400
        
        # 1. Normalize both images to North-Up using their true pivots (in stretched space)
        R_curr = cv2.getRotationMatrix2D(curr_pivot, curr_arrow_angle, 1.0)
        R_prev = cv2.getRotationMatrix2D(prev_pivot, prev_arrow_angle, 1.0)
        
        norm_curr = cv2.warpAffine(current_img, R_curr, (w, h))
        norm_prev = cv2.warpAffine(prev_img, R_prev, (w, h))
        
        # 2. Extract 400x400 circular region from stretched images (centered at 200, 260)
        # Crop region: x=0 to 400, y=60 to 460 (centered at y=260)
        crop_y1, crop_y2 = self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2
        crop_curr = norm_curr[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
        crop_prev = norm_prev[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
        
        # Apply circular mask to the cropped region
        crop_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
        cv2.circle(crop_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)  # Center at (200, 200) in crop space
        
        crop_curr = cv2.bitwise_and(crop_curr, crop_curr, mask=crop_mask)
        crop_prev = cv2.bitwise_and(crop_prev, crop_prev, mask=crop_mask)
        
        # 3. Pre-process for Matching
        # Use include_gold=False for actual calculation (gold arrow shouldn't affect phase correlation)
        blue_curr = self._filter_blue_colors(crop_curr, include_gold=False)
        blue_prev = self._filter_blue_colors(crop_prev, include_gold=False)
        
        gray_curr = cv2.cvtColor(blue_curr, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(blue_prev, cv2.COLOR_BGR2GRAY)
        
        # For preview: use same filters as angular check (include_gold=True)
        # Keep as BGR to preserve color information (blue, gold, etc.)
        blue_curr_preview = self._filter_blue_colors(crop_curr, include_gold=True)
        blue_prev_preview = self._filter_blue_colors(crop_prev, include_gold=True)
        
        # 4. Crop down to PHASE_CORR_CROP_SIZE to remove border for dx/dy calculation
        # This removes the outer border that may cause issues with phase correlation
        # Center the crop in the 400x400 image
        crop_size = self.PHASE_CORR_CROP_SIZE
        crop_offset = (self.CIRCULAR_CROP_SIZE - crop_size) // 2  # Centers the crop
        
        # Crop color-filtered images for preview (store unmasked BGR version to preserve colors)
        curr_cropped_preview = blue_curr_preview[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
        prev_cropped_preview = blue_prev_preview[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
        
        # Store cropped images for debug preview (unmasked, with gold filter to match angular check, in BGR color)
        self.last_cropped_curr = curr_cropped_preview.copy()
        self.last_cropped_prev = prev_cropped_preview.copy()
        
        # Now apply center mask for actual phase correlation calculation (use non-gold filtered version)
        gray_curr_cropped = gray_curr[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
        gray_prev_cropped = gray_prev[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
        gray_curr_cropped = self.mask_center(gray_curr_cropped, radius=self.CENTER_MASK_RADIUS_PHASE_CORR)
        gray_prev_cropped = self.mask_center(gray_prev_cropped, radius=self.CENTER_MASK_RADIUS_PHASE_CORR)
        
        # Apply Hanning window to eliminate edge artifacts
        win = cv2.createHanningWindow((crop_size, crop_size), cv2.CV_32F)
        
        f_curr = gray_curr_cropped.astype(np.float32) * win
        f_prev = gray_prev_cropped.astype(np.float32) * win
        
        # 5. Phase Correlation on cropped images
        shift, conf = cv2.phaseCorrelate(f_prev, f_curr)
        dx, dy = shift
        
        if conf < PHASE_CORR_CONFIDENCE_THRESHOLD:
            return 0.0, 0.0, conf
        
        # Validate dx, dy offsets are realistic
        # Uses PHASE_CORR_CROP_SIZE (225) crop, but user mentioned 200 for center
        # If offset is greater than 1/2 that distance (100), it's not realistic
        max_offset_center = 250  # half of 200 for center crop
        
        # If either dx or dy exceeds half the center image width, use average instead
        if abs(dx) > max_offset_center or abs(dy) > max_offset_center:
            # Store original invalid values for logging
            original_dx = max(-255, min(255, dx))
            original_dy = max(-255, min(255, dy))
            
            # Invalid transformation - use average of recent values instead
            if self.dx_pc_history and self.dy_pc_history:
                # Calculate average of recent history
                recent_count = min(self.pc_history_size, len(self.dx_pc_history))
                avg_dx = np.mean(self.dx_pc_history[-recent_count:])
                avg_dy = np.mean(self.dy_pc_history[-recent_count:])
                
                # Apply decay based on consecutive invalid readings
                # Use a separate counter for PC to track independently
                if not hasattr(self, 'consecutive_invalid_pc_count'):
                    self.consecutive_invalid_pc_count = 0
                self.consecutive_invalid_pc_count += 1
                decay = self.average_decay_factor ** self.consecutive_invalid_pc_count
                dx = original_dx #avg_dx * decay
                dy = original_dy #avg_dy * decay
                
                print(f"[DRIFT_PC] Invalid offset (dx={original_dx:.1f}, dy={original_dy:.1f}), using decayed average (dx={dx:.1f}, dy={dy:.1f}, decay={decay:.3f}, count={self.consecutive_invalid_pc_count})")
            else:
                # No history available - return zeros with low confidence
                if hasattr(self, 'consecutive_invalid_pc_count'):
                    self.consecutive_invalid_pc_count = 0  # Reset counter
                print(f"[DRIFT_PC] Invalid offset (dx={dx:.1f}, dy={dy:.1f}), no history available")
                return 0.0, 0.0, 0.0
        else:
            # Valid reading - reset consecutive invalid count
            if hasattr(self, 'consecutive_invalid_pc_count'):
                self.consecutive_invalid_pc_count = 0
        
        # Update history with valid values (or averaged values if they were invalid)
        self.dx_pc_history.append(dx)
        self.dy_pc_history.append(dy)
        if len(self.dx_pc_history) > self.pc_history_size:
            self.dx_pc_history.pop(0)
            self.dy_pc_history.pop(0)
            
        # Translation is already in world-space (stretched coordinates)
        return dx, dy, conf

    def compute_drift(self, current_img: np.ndarray, target_minimap: np.ndarray, 
                     target_arrow_angle: Optional[float] = None) -> Optional[Tuple[float, float, float, int, np.ndarray]]:
        """
        Legacy Feature-Matching drift (ORB). Used for navigation lookahead.
        
        Args:
            current_img: Current screen image.
            target_minimap: Target minimap image to match against.
            target_arrow_angle: Optional target node arrow angle for angle validation.
        """
        current_minimap = self.extract_minimap(current_img)
        if current_minimap is None or target_minimap is None:
            return None
        
        # Ensure target_minimap is masked and has same dimensions as current_minimap
        # Target images are already stretched (400x520) from recording via get_minimap_crop,
        # so we do NOT stretch them again - only apply mask if needed
        # NOTE: This confirms that saved images are always stretched (400x520)
        target_h, target_w = target_minimap.shape[:2]
        current_h, current_w = current_minimap.shape[:2]
        
        if (target_h, target_w) == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH) and (current_h, current_w) == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH):
            # Both are stretched 400x520, just apply mask to target to match current_minimap
            target_minimap = cv2.bitwise_and(target_minimap, target_minimap, mask=self.mask_stretched)
        elif (target_h, target_w) != (current_h, current_w):
            # Dimensions don't match - resize target to match current (shouldn't happen normally)
            # This preserves aspect ratio if possible, but forces match for feature detection
            target_minimap = cv2.resize(target_minimap, (current_w, current_h), interpolation=cv2.INTER_LINEAR)
            # Re-apply mask after resize if dimensions are now 400x520
            if target_minimap.shape[:2] == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH):
                target_minimap = cv2.bitwise_and(target_minimap, target_minimap, mask=self.mask_stretched)
            
        filt_current = self._filter_blue_colors(current_minimap)
        filt_target = self._filter_blue_colors(target_minimap)
        
        self.last_filtered_current = filt_current
        self.last_filtered_target = filt_target
            
        kp1, des1 = self.vision.feature_matcher.detect_and_compute(filt_target, self.mask)
        kp2, des2 = self.vision.feature_matcher.detect_and_compute(filt_current, self.mask)
        
        matches = self.vision.feature_matcher.match_features(des1, des2)
        
        self.last_kp1 = kp1
        self.last_kp2 = kp2
        self.last_matches = matches
        
        M, mask = self.vision.feature_matcher.compute_homography(kp1, kp2, matches)
        
        if M is None:
            return None
            
        inliers_count = int(np.sum(mask)) if mask is not None else 0
        dx, dy, angle = self.vision.feature_matcher.decompose_homography(M)
        
        # Unwrap angle to prevent flip-flopping between positive and negative
        # Use gold arrow angle and target arrow angle for validation if available
        angle = self._unwrap_angle(angle, current_img, target_arrow_angle)
        
        # Validate dx, dy offsets are realistic
        # Image width is 300 for large (circular mask diameter = 2 * MINIMAP_RADIUS = 300)
        # Image width is 200 for center (used in some contexts)
        # If offset is greater than 1/2 that distance, it's not realistic and should be treated as lost/junk
        # Since compute_drift uses the full circular masked region (radius 150, diameter 300),
        # use half of 300 = 150 as the threshold
        max_offset_threshold = self.MINIMAP_RADIUS  # 150 (half of 300 diameter for large)
        
        # If either dx or dy exceeds half the image width, use average instead
        if abs(dx) > max_offset_threshold or abs(dy) > max_offset_threshold:
            # Store original invalid values for logging
            original_dx, original_dy = dx, dy
            
            # Invalid transformation - use average of recent values instead
            if self.dx_history and self.dy_history:
                # Calculate average of recent history (use last 10 values if available)
                recent_count = min(10, len(self.dx_history))
                avg_dx = np.mean(self.dx_history[-recent_count:])
                avg_dy = np.mean(self.dy_history[-recent_count:])
                
                # Apply decay based on consecutive invalid readings
                self.consecutive_invalid_count += 1
                decay = self.average_decay_factor ** self.consecutive_invalid_count
                dx = avg_dx * decay
                dy = avg_dy * decay
                
                print(f"[DRIFT] Invalid offset (dx={original_dx:.1f}, dy={original_dy:.1f}), using decayed average (dx={dx:.1f}, dy={dy:.1f}, decay={decay:.3f}, count={self.consecutive_invalid_count})")
            else:
                # No history available - treat as lost state
                self.consecutive_invalid_count = 0  # Reset counter
                return None
        else:
            # Valid reading - reset consecutive invalid count
            self.consecutive_invalid_count = 0
        
        self.last_dx = dx
        self.last_dy = dy
        self.last_angle = angle
        self.last_inliers_count = inliers_count
        
        # Update histogram buffers
        self.dx_history.append(dx)
        self.dy_history.append(dy)
        self.angle_history.append(angle)
        
        # Keep buffer size limited
        if len(self.dx_history) > self.hist_buffer_size:
            self.dx_history.pop(0)
            self.dy_history.pop(0)
            self.angle_history.pop(0)
        
        return dx, dy, angle, inliers_count, M

    def _generate_master_map_advanced(self, nodes: list, landmark_dir: str) -> Optional[np.ndarray]:
        """
        Advanced master map generation using SuperPoint + LightGlue + Bundle Adjustment.
        
        Uses state-of-the-art stitching pipeline for maximum accuracy.
        """
        if not nodes:
            return None
        
        try:
            from .movement.stitching import (
                SuperPointExtractor, LightGlueMatcher, 
                MAGSACEstimator, BundleAdjustment, APAPWarper
            )
        except ImportError as e:
            print(f"[MASTER MAP] Failed to import advanced stitching components: {e}")
            print("[MASTER MAP] Falling back to legacy method.")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        # Check if imports actually succeeded (they might be None if dependencies are missing)
        if (SuperPointExtractor is None or LightGlueMatcher is None or 
            MAGSACEstimator is None or BundleAdjustment is None):
            print("[MASTER MAP] Advanced stitching components not available (missing dependencies).")
            print("[MASTER MAP] Falling back to legacy method.")
            print("[MASTER MAP] To use advanced stitching, install: torch, kornia, scipy")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        print("[MASTER MAP] Using advanced stitching pipeline...")
        
        # 1. Load and preprocess all node images
        # We need TWO versions:
        # - Full images for feature extraction (more features = better matching)
        # - Filtered images for final display (clean visual appearance)
        node_images_full = []  # Full minimap for feature extraction
        node_images_filtered = []  # Filtered for display
        node_angles = []
        node_pivots = []
        
        for node in nodes:
            path = os.path.join(landmark_dir, node['minimap_path'])
            if not os.path.exists(path):
                continue
            
            img = cv2.imread(path)
            if img is None:
                continue
            
            # Stretch if needed
            h, w = img.shape[:2]
            if h != self.MINIMAP_STRETCHED_HEIGHT or w != self.MINIMAP_STRETCHED_WIDTH:
                img = cv2.resize(img, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT), 
                               interpolation=cv2.INTER_LINEAR)
            
            # Rotate to North
            arrow_angle = node.get('arrow_angle', 0.0)
            default_pivot = (self.MINIMAP_CENTER_STRETCHED_X, self.MINIMAP_CENTER_STRETCHED_Y)
            pivot_stretched = node.get('arrow_pivot', default_pivot)
            
            R = cv2.getRotationMatrix2D(pivot_stretched, arrow_angle, 1.0)
            img_rotated = cv2.warpAffine(img, R, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT))
            
            # Extract circular region
            crop_y1, crop_y2 = self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2
            img_circle = img_rotated[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
            
            # Verify crop size is correct
            if img_circle.shape[:2] != (self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE):
                print(f"[MASTER MAP] WARNING: Circular crop size mismatch! Expected {self.CIRCULAR_CROP_SIZE}x{self.CIRCULAR_CROP_SIZE}, got {img_circle.shape[:2]}")
                img_circle = cv2.resize(img_circle, (self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # Create circular mask - center should be at (200, 200) in the 400x400 cropped image
            # The center in stretched space is (200, 260), after cropping y: 60-460, center becomes (200, 200)
            # CRITICAL: Use reduced radius to aggressively remove border artifacts
            circle_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
            cv2.circle(circle_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)
            
            # Apply mask strictly - zero everything outside circle
            # This is the FIRST and most important step to remove border artifacts
            img_north = np.zeros_like(img_circle, dtype=np.uint8)
            img_north[circle_mask > 0] = img_circle[circle_mask > 0]
            
            # Double-check: ensure no border pixels remain
            outside_mask = (circle_mask == 0)
            if np.any(outside_mask):
                num_border_pixels = np.sum(img_north[outside_mask] > 0)
                if num_border_pixels > 0:
                    print(f"[MASTER MAP] WARNING: Found {num_border_pixels} border pixels in initial crop for node {i}, zeroing them")
                    img_north[outside_mask] = 0
            
            # Store FULL image for feature extraction (more features available)
            # CRITICAL: Strictly enforce circular mask to prevent border artifacts from affecting feature matching
            # Only mask out center (player arrow) and dynamic circles, keep all other features
            img_full = img_north.copy()
            # Mask center to remove player arrow
            center_masked_full = self.mask_center(img_full, radius=self.CENTER_MASK_RADIUS_MASTER_MAP)
            # Mask dynamic circles (enemies/allies)
            circle_mask_dyn = self._mask_dynamic_circles(img_full)
            # Apply both masks
            img_full = cv2.bitwise_and(center_masked_full, center_masked_full, mask=circle_mask_dyn)
            # CRITICAL: Strictly enforce circular boundary - zero everything outside circle
            # This prevents border artifacts from being used in feature extraction
            img_full_strict = np.zeros_like(img_full, dtype=np.uint8)
            img_full_strict[circle_mask > 0] = img_full[circle_mask > 0]
            # One more pass to ensure no border pixels
            img_full_strict = cv2.bitwise_and(img_full_strict, img_full_strict, mask=circle_mask)
            node_images_full.append(img_full_strict)
            
            # Create filtered version for display (blue paths only)
            # CRITICAL: Apply circular mask at EVERY step to prevent border artifacts
            hsv = cv2.cvtColor(img_north, cv2.COLOR_BGR2HSV)
            lower_blue = np.array(self.blue_lower)
            upper_blue = np.array(self.blue_upper)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            circle_mask_dyn = self._mask_dynamic_circles(img_north)
            
            # Apply circular mask to color mask BEFORE combining
            mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=circle_mask)
            combined_mask = cv2.bitwise_and(mask_blue, circle_mask_dyn)
            combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=circle_mask)  # Enforce again
            color_filtered = cv2.bitwise_and(img_north, img_north, mask=combined_mask)
            
            # Edge detection - but mask the result immediately
            edge_result = self._apply_edge_detection(img_north)
            edge_result = cv2.bitwise_and(edge_result, edge_result, mask=circle_mask)  # Mask edges
            
            color_mask_gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
            color_mask_binary = (color_mask_gray > 0).astype(np.uint8)
            # Apply circular mask to binary mask BEFORE dilation
            color_mask_binary = cv2.bitwise_and(color_mask_binary, color_mask_binary, mask=circle_mask)
            
            kernel_dilate = np.ones((3, 3), np.uint8)
            color_mask_dilated = cv2.dilate(color_mask_binary, kernel_dilate, iterations=2)
            # CRITICAL: Re-apply circular mask after dilation to remove artifacts outside circle
            color_mask_dilated = cv2.bitwise_and(color_mask_dilated, color_mask_dilated, mask=circle_mask)
            
            edge_gray = cv2.cvtColor(edge_result, cv2.COLOR_BGR2GRAY)
            edge_masked = cv2.bitwise_and(edge_gray, edge_gray, mask=color_mask_dilated)
            edge_result_filtered = cv2.cvtColor(edge_masked, cv2.COLOR_GRAY2BGR)
            blue_only = cv2.max(color_filtered, edge_result_filtered)
            
            # FINAL: Strictly enforce circular boundary - zero everything outside
            blue_only_masked = np.zeros_like(blue_only, dtype=np.uint8)
            blue_only_masked[circle_mask > 0] = blue_only[circle_mask > 0]
            
            # One more pass: ensure no pixels outside circle
            blue_only_masked = cv2.bitwise_and(blue_only_masked, blue_only_masked, mask=circle_mask)
            
            node_images_filtered.append(blue_only_masked)
            
            node_angles.append(arrow_angle)
            node_pivots.append(pivot_stretched)
        
        if len(node_images_full) == 0:
            print("[MASTER MAP] No valid node images found.")
            return None
        
        n_nodes = len(node_images_full)
        print(f"[MASTER MAP] Processing {n_nodes} nodes with advanced stitching...")
        print(f"[MASTER MAP] Using FULL minimap images for feature extraction (better matching)")
        
        # Define fragment size early (used for validation)
        fragment_size = self.CIRCULAR_CROP_SIZE  # 400x400
        
        # 2. Extract features using SuperPoint on FULL images (not filtered)
        # Full images have more features = better matching accuracy
        print("[MASTER MAP] Extracting features with SuperPoint from full minimap images...")
        try:
            extractor = SuperPointExtractor(n_features=SUPERPOINT_N_FEATURES, device=SUPERPOINT_DEVICE)
        except (ImportError, RuntimeError, AttributeError) as e:
            print(f"[MASTER MAP] Failed to initialize SuperPoint: {e}")
            print("[MASTER MAP] Falling back to legacy method.")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        try:
            # Use FULL images for feature extraction (more features available)
            # Full images contain all minimap features, not just filtered blue paths
            # This gives SuperPoint/LightGlue much more information to work with
            all_features = [extractor.extract(img) for img in node_images_full]
            feature_counts = [len(f['keypoints']) for f in all_features]
            print(f"[MASTER MAP] Extracted features from FULL minimap images: {feature_counts} keypoints per node")
            print(f"[MASTER MAP] Total features: {sum(feature_counts)} (using full images for better matching)")
        except Exception as e:
            print(f"[MASTER MAP] Failed to extract features: {e}")
            print("[MASTER MAP] Falling back to legacy method.")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        # 3. Match features using LightGlue
        print("[MASTER MAP] Matching features with LightGlue...")
        try:
            matcher = LightGlueMatcher(filter_threshold=LIGHTGLUE_FILTER_THRESHOLD, device=LIGHTGLUE_DEVICE)
            all_matches = matcher.match_all_pairs(all_features)
        except Exception as e:
            print(f"[MASTER MAP] Failed to match features: {e}")
            print("[MASTER MAP] Falling back to legacy method.")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        # 4. Estimate pairwise transformations using MAGSAC
        print("[MASTER MAP] Estimating transformations with MAGSAC++...")
        try:
            estimator = MAGSACEstimator(
                method=MAGSAC_METHOD,
                threshold=MAGSAC_THRESHOLD,
                confidence=MAGSAC_CONFIDENCE,
                max_iters=MAGSAC_MAX_ITERS
            )
        except Exception as e:
            print(f"[MASTER MAP] Failed to initialize MAGSAC: {e}")
            print("[MASTER MAP] Falling back to legacy method.")
            return self._generate_master_map_legacy(nodes, landmark_dir)
        
        pairwise_transforms = {}
        match_counts = {}
        for (i, j), match_data in all_matches.items():
            matches = match_data['matches']
            
            # Require more matches for reliability (increased threshold)
            if len(matches) < 12:  # Increased from 8 to 12
                print(f"[MASTER MAP] Skipping ({i}->{j}): only {len(matches)} matches (need 12+)")
                continue
            
            # Additional filtering: check match quality
            # For adjacent nodes, matches should be reasonably distributed
            kpts_i = all_features[i]['keypoints']
            kpts_j = all_features[j]['keypoints']
            
            # Check if matches are well-distributed (not all clustered)
            match_kpts_i = kpts_i[matches[:, 0]]
            match_kpts_j = kpts_j[matches[:, 1]]
            
            # Compute spread of matches
            spread_i = np.std(match_kpts_i, axis=0)
            spread_j = np.std(match_kpts_j, axis=0)
            min_spread = min(np.min(spread_i), np.min(spread_j))
            
            if min_spread < 20:  # Matches too clustered
                print(f"[MASTER MAP] Skipping ({i}->{j}): matches too clustered (spread={min_spread:.1f}px)")
                continue
            
            M, inlier_mask, conf = estimator.estimate_from_features(
                all_features[i], all_features[j], matches
            )
            
            if M is not None and conf > 0.5:
                n_inliers = np.sum(inlier_mask) if inlier_mask is not None else len(matches)
                
                # Validate homography quality
                kpts_i = all_features[i]['keypoints']
                kpts_j = all_features[j]['keypoints']
                inlier_matches = matches[inlier_mask] if inlier_mask is not None else matches
                
                # Check reprojection error
                if len(inlier_matches) > 0:
                    src_pts = kpts_i[inlier_matches[:, 0]]
                    dst_pts = kpts_j[inlier_matches[:, 1]]
                    
                    # Transform source points
                    src_pts_hom = np.column_stack([src_pts, np.ones(len(src_pts))])
                    transformed = (M @ src_pts_hom.T).T
                    transformed = transformed[:, :2] / transformed[:, 2:3]
                    
                    # Compute reprojection errors
                    errors = np.linalg.norm(transformed - dst_pts, axis=1)
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    median_error = np.median(errors)
                    
                    # Reject if reprojection error is too high
                    if mean_error > 10.0 or median_error > 8.0:
                        print(f"[MASTER MAP] Rejecting ({i}->{j}): high reprojection error "
                              f"(mean={mean_error:.1f}, median={median_error:.1f}, max={max_error:.1f})")
                        continue
                else:
                    mean_error = 0.0
                    median_error = 0.0
                
                # Validate transform parameters (images are North-aligned)
                # For affine transforms, we can directly read rotation and scale
                rotation_angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                scale_x = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                scale_y = np.sqrt(M[0, 1]**2 + M[1, 1]**2)
                
                # For North-aligned images, rotation should be very small (< 2°) and scale very close to 1
                # Since images are pre-rotated to North, any rotation in the transform is error
                if abs(rotation_angle) > 2.0:
                    print(f"[MASTER MAP] Rejecting ({i}->{j}): excessive rotation "
                          f"(rot={rotation_angle:.1f}°, should be < 2° for North-aligned images)")
                    continue
                
                # Scale should be very close to 1.0 (within 5% tolerance)
                if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
                    print(f"[MASTER MAP] Rejecting ({i}->{j}): invalid scale "
                          f"(scale=({scale_x:.3f}, {scale_y:.3f}), should be ~1.0)")
                    continue
                
                # Additional validation: check if translation is reasonable
                # For affine, translation is directly in [0,2] and [1,2]
                dx_test = M[0, 2]
                dy_test = M[1, 2]
                translation_mag = np.sqrt(dx_test**2 + dy_test**2)
                
                # Translation should be reasonable (not more than 2 fragment widths)
                # For adjacent nodes, translation should typically be < 1 fragment width
                max_reasonable_translation = fragment_size * 2.0
                if translation_mag > max_reasonable_translation:
                    print(f"[MASTER MAP] Rejecting ({i}->{j}): translation too large "
                          f"({translation_mag:.1f}px, max={max_reasonable_translation:.1f}px)")
                    continue
                
                # For adjacent nodes (i+1 == j), translation should be relatively small
                if abs(i - j) == 1 and translation_mag > fragment_size * 1.5:
                    print(f"[MASTER MAP] WARNING: Large translation for adjacent nodes ({i}->{j}): "
                          f"{translation_mag:.1f}px (expected < {fragment_size * 1.5:.1f}px)")
                
                pairwise_transforms[(i, j)] = M
                match_counts[(i, j)] = n_inliers
                # Update inliers in match data
                all_matches[(i, j)]['inliers'] = inlier_mask
                print(f"[MASTER MAP] Transform ({i}->{j}): {n_inliers} inliers, conf={conf:.3f}, "
                      f"reproj_error={mean_error:.2f}px, translation=({dx_test:.1f}, {dy_test:.1f}), rot={rotation_angle:.1f}°")
        
        if len(pairwise_transforms) == 0:
            print("[MASTER MAP] ERROR: No valid transformations found from feature matching.")
            print("[MASTER MAP] Cannot create map without feature matches. Check that nodes have overlapping features.")
            return None
        
        # Check if we have enough transforms to position all nodes
        # Need at least n-1 transforms for n nodes (spanning tree)
        min_required = n_nodes - 1
        if len(pairwise_transforms) < min_required:
            print(f"[MASTER MAP] WARNING: Only {len(pairwise_transforms)} transforms for {n_nodes} nodes (need {min_required} minimum).")
            print("[MASTER MAP] Some nodes may not be positioned correctly, but continuing anyway.")
        
        print(f"[MASTER MAP] Found {len(pairwise_transforms)} valid pairwise transformations")
        
        # 5. Initialize positions using ONLY feature matching transforms
        # Use least-squares to solve for all positions simultaneously
        # This minimizes accumulated errors compared to sequential BFS positioning
        initial_positions = np.zeros((n_nodes, 3))  # [x, y, theta]
        
        # Helper function to extract translation from transform matrix
        def extract_translation_robust(H, src_idx=None, dst_idx=None):
            """Extract translation from transform matrix (affine or homography)."""
            # For affine transforms, translation is directly in the matrix
            # For homography, we need to transform a point
            # Since we're using affine now, we can read translation directly
            if H.shape == (3, 3):
                # Check if it's affine (bottom row is [0, 0, 1])
                is_affine = np.allclose(H[2, :], [0, 0, 1])
                
                if is_affine:
                    # Affine transform: translation is directly in [0,2] and [1,2]
                    dx_hom = H[0, 2]
                    dy_hom = H[1, 2]
                    # For affine, we can also check consistency by transforming origin
                    origin_transformed = H @ np.array([[0.0], [0.0], [1.0]])
                    dx_check = origin_transformed[0, 0] / origin_transformed[2, 0]
                    dy_check = origin_transformed[1, 0] / origin_transformed[2, 0]
                    # Should match (within small tolerance)
                    if abs(dx_hom - dx_check) > 0.1 or abs(dy_hom - dy_check) > 0.1:
                        print(f"[MASTER MAP] WARNING: Affine translation mismatch: direct=({dx_hom:.1f}, {dy_hom:.1f}), "
                              f"transformed=({dx_check:.1f}, {dy_check:.1f})")
                    dx_std = 0.0  # Affine translation is exact
                    dy_std = 0.0
                else:
                    # Full homography: transform reference points
                    ref_points = np.array([
                        [self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y],  # Center
                        [self.MINIMAP_CENTER_X + 50, self.MINIMAP_CENTER_Y],  # Right
                        [self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y + 50],  # Down
                        [self.MINIMAP_CENTER_X - 50, self.MINIMAP_CENTER_Y],  # Left
                        [self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y - 50],  # Up
                    ], dtype=np.float32)
                    
                    translations = []
                    for pt in ref_points:
                        pt_homogeneous = np.array([[pt[0], pt[1], 1.0]]).T
                        transformed = H @ pt_homogeneous
                        transformed = transformed[:2] / transformed[2]
                        translations.append(transformed.flatten() - pt)
                    
                    translations = np.array(translations)
                    dx_hom = np.median(translations[:, 0])
                    dy_hom = np.median(translations[:, 1])
                    dx_std = np.std(translations[:, 0])
                    dy_std = np.std(translations[:, 1])
            else:
                # Fallback: assume translation is in last column
                dx_hom = H[0, -1] if H.shape[0] > 0 and H.shape[1] > 0 else 0.0
                dy_hom = H[1, -1] if H.shape[0] > 1 and H.shape[1] > 0 else 0.0
                dx_std = 0.0
                dy_std = 0.0
            
            # Method 2: Direct feature match translation (if available)
            dx_match = None
            dy_match = None
            if src_idx is not None and dst_idx is not None and (src_idx, dst_idx) in all_matches:
                match_data = all_matches[(src_idx, dst_idx)]
                matches = match_data['matches']
                inliers = match_data.get('inliers', np.ones(len(matches), dtype=bool))
                valid_matches = matches[inliers] if inliers is not None else matches
                
                if len(valid_matches) > 0:
                    kpts_src = all_features[src_idx]['keypoints']
                    kpts_dst = all_features[dst_idx]['keypoints']
                    
                    # Compute translation for each inlier match
                    # Note: These are raw pixel translations in image space
                    # They may not match homography translation if there's rotation/scale
                    match_translations = []
                    for match in valid_matches[:min(50, len(valid_matches))]:  # Use up to 50 matches
                        idx_src, idx_dst = match
                        pt_src = kpts_src[idx_src]
                        pt_dst = kpts_dst[idx_dst]
                        match_translations.append(pt_dst - pt_src)
                    
                    if match_translations:
                        match_translations = np.array(match_translations)
                        dx_match = np.median(match_translations[:, 0])
                        dy_match = np.median(match_translations[:, 1])
                        match_dy_std = np.std(match_translations[:, 1])
                        
                        # Only use match-based if it's significantly different from zero
                        # and agrees reasonably with homography
                        match_mag = np.sqrt(dx_match**2 + dy_match**2)
                        if match_mag < 1.0:  # Too close to zero, likely wrong
                            dx_match = None
                            dy_match = None
            
            # Method 3: Phase correlation (if images are available)
            dx_pc = None
            dy_pc = None
            pc_conf = 0.0
            if src_idx is not None and dst_idx is not None:
                try:
                    img_src = node_images_full[src_idx]
                    img_dst = node_images_full[dst_idx]
                    angle_src = node_angles[src_idx]
                    angle_dst = node_angles[dst_idx]
                    pivot_src = node_pivots[src_idx]
                    pivot_dst = node_pivots[dst_idx]
                    
                    # Use phase correlation as cross-validation
                    dx_pc, dy_pc, pc_conf = self.compute_drift_pc(
                        img_dst, img_src, angle_dst, pivot_dst, angle_src, pivot_src
                    )
                except Exception as e:
                    pass  # Phase correlation failed, skip it
            
            # CRITICAL: Always use transform matrix (affine/homography) as base - it's the most reliable
            # Other methods are for cross-validation only
            # Never default to zero - if transform gives zero, that's the actual translation
            
            # Choose best translation based on consistency
            candidates = []
            # Transform matrix is always available and is our primary method
            candidates.append(('transform', dx_hom, dy_hom, dy_std))
            
            if dx_match is not None:
                # Only use match-based if it's significantly different and more consistent
                if abs(dy_match - dy_hom) < 20 and match_dy_std < dy_std * 0.8:
                    candidates.append(('matches', dx_match, dy_match, match_dy_std if 'match_dy_std' in locals() else dy_std))
            
            if dx_pc is not None and pc_conf > 0.5:
                # Only use phase correlation if it agrees with homography
                if abs(dy_pc - dy_hom) < 25:  # Must be within 25px of homography
                    candidates.append(('phase_corr', dx_pc, dy_pc, 0))
            
            # Always prefer homography unless another method is clearly better
            if len(candidates) > 1:
                # Prefer phase correlation if confidence is very high and it agrees
                pc_candidate = [c for c in candidates if c[0] == 'phase_corr']
                if pc_candidate and pc_conf > 0.8 and abs(dy_pc - dy_hom) < 15:
                    _, dx, dy, _ = pc_candidate[0]
                    print(f"[MASTER MAP] Using phase correlation for ({src_idx}->{dst_idx}): "
                          f"dx={dx:.1f}, dy={dy:.1f} (conf={pc_conf:.3f}, agrees with transform)")
                    return dx, dy
                
                # Otherwise, use transform matrix (most reliable)
                return dx_hom, dy_hom
            
            # Default to transform matrix (always available)
            return dx_hom, dy_hom
        
        # Extract all translations first
        transform_translations = {}
        y_translations = []  # Track y-translations for validation
        zero_translation_count = 0
        for (src, dst), H in pairwise_transforms.items():
            dx, dy = extract_translation_robust(H, src, dst)
            translation_mag = np.sqrt(dx**2 + dy**2)
            
            # Validate translation is reasonable
            if translation_mag > 800:  # More than 2 fragment widths seems wrong
                print(f"[MASTER MAP] WARNING: Large translation ({translation_mag:.1f}px) for ({src}->{dst})")
                # Don't add to translations if it's suspiciously large
                continue
            
            # Check for suspicious zero translations
            # Adjacent nodes should have SOME translation (unless they're truly overlapping)
            if translation_mag < 0.5 and abs(src - dst) == 1:  # Adjacent nodes with zero translation
                print(f"[MASTER MAP] WARNING: Suspicious zero translation for adjacent nodes ({src}->{dst}), skipping")
                zero_translation_count += 1
                continue
            
            transform_translations[(src, dst)] = (dx, dy)
            y_translations.append(dy)
            print(f"[MASTER MAP] Transform ({src}->{dst}): dx={dx:.1f}, dy={dy:.1f}")
        
        if zero_translation_count > 0:
            print(f"[MASTER MAP] Filtered {zero_translation_count} suspicious zero translations")
        
        # Validate y-translations for consistency
        if y_translations:
            y_median = np.median(y_translations)
            y_std = np.std(y_translations)
            y_mean = np.mean(y_translations)
            print(f"[MASTER MAP] Y-translation stats: mean={y_mean:.1f}, median={y_median:.1f}, std={y_std:.1f}")
            if y_std > 100:  # High variance suggests inconsistent transforms
                print(f"[MASTER MAP] WARNING: High variance in y-translations ({y_std:.1f}), may indicate alignment issues")
            
            # Check for systematic bias
            if abs(y_mean) > 50 and y_std < 50:
                print(f"[MASTER MAP] WARNING: Systematic y-bias detected (mean={y_mean:.1f}), transforms may have consistent error")
        
        # Validate transform consistency (check triangle closure)
        # Only check transforms that we're actually using for positioning
        consistency_errors = []
        transforms_to_check = set(transform_translations.keys())  # Only check transforms we're using
        for (i, j) in transforms_to_check:
            for (j2, k) in transforms_to_check:
                if j == j2 and (i, k) in transforms_to_check:
                    # We have i->j, j->k, and i->k
                    H_ij = pairwise_transforms[(i, j)]
                    H_jk = pairwise_transforms[(j, k)]
                    H_ik = pairwise_transforms[(i, k)]
                    
                    # Compose: H_ik_composed = H_jk @ H_ij
                    H_ik_composed = H_jk @ H_ij
                    
                    # Check translation difference
                    dx_ij, dy_ij = extract_translation_robust(H_ij)
                    dx_jk, dy_jk = extract_translation_robust(H_jk)
                    dx_ik, dy_ik = extract_translation_robust(H_ik)
                    dx_composed, dy_composed = extract_translation_robust(H_ik_composed)
                    
                    error_x = abs(dx_ik - dx_composed)
                    error_y = abs(dy_ik - dy_composed)
                    
                    if error_y > 20:  # Significant y-error
                        consistency_errors.append((i, j, k, error_x, error_y))
                        print(f"[MASTER MAP] Consistency check ({i}->{j}->{k}): y-error={error_y:.1f}px")
        
        if consistency_errors:
            print(f"[MASTER MAP] WARNING: Found {len(consistency_errors)} transform consistency issues (may cause vertical misalignment)")
            
            # Filter out transforms that are part of inconsistent chains
            # Count how many errors each transform is involved in
            transform_error_count = {}
            for (i, j, k, err_x, err_y) in consistency_errors:
                transform_error_count[(i, j)] = transform_error_count.get((i, j), 0) + 1
                transform_error_count[(j, k)] = transform_error_count.get((j, k), 0) + 1
                transform_error_count[(i, k)] = transform_error_count.get((i, k), 0) + 1
            
            # Remove transforms that are involved in many errors
            # Keep only transforms with 0-1 errors (most are good)
            max_errors_per_transform = 2
            filtered_transforms = {}
            filtered_translations = {}
            for (src, dst) in transform_translations.keys():
                error_count = transform_error_count.get((src, dst), 0)
                if error_count <= max_errors_per_transform:
                    filtered_transforms[(src, dst)] = pairwise_transforms[(src, dst)]
                    filtered_translations[(src, dst)] = transform_translations[(src, dst)]
                else:
                    print(f"[MASTER MAP] Filtering out inconsistent transform ({src}->{dst}) with {error_count} consistency errors")
            
            # Update dictionaries if we filtered anything
            original_count = len(transform_translations)
            if len(filtered_transforms) < len(pairwise_transforms):
                filtered_count = len(filtered_transforms)
                print(f"[MASTER MAP] Filtered {len(pairwise_transforms) - filtered_count} inconsistent transforms")
                pairwise_transforms = filtered_transforms
                transform_translations = filtered_translations
                
                # Also update match_counts to match
                filtered_match_counts = {k: match_counts[k] for k in filtered_transforms.keys() if k in match_counts}
                match_counts = filtered_match_counts
                
                # Check if we still have enough transforms after filtering
                if len(filtered_transforms) < n_nodes - 1:
                    print(f"[MASTER MAP] WARNING: After filtering, only {len(filtered_transforms)} transforms remain (need {n_nodes-1} minimum)")
                    print("[MASTER MAP] Some nodes may not be positioned correctly")
        
        # Solve for all positions simultaneously using least-squares
        # Build system: for each transform (i->j), pos_j = pos_i + translation_ij
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        
        n_constraints = len(transform_translations)
        if n_constraints > 0:
            # Build weighted constraint matrix: A * positions = b
            # Weight each constraint by match quality (inlier count, confidence)
            A = np.zeros((n_constraints * 2 + 2, n_nodes * 2))  # +2 for anchor
            b = np.zeros(n_constraints * 2 + 2)
            weights = np.ones(n_constraints * 2 + 2)  # Weight vector
            
            constraint_idx = 0
            for (src, dst), (dx, dy) in transform_translations.items():
                # Get match quality metrics for weighting
                n_inliers = match_counts.get((src, dst), 0)
                # Weight by inlier count (more inliers = more reliable)
                weight = min(1.0, n_inliers / 20.0)  # Normalize to 0-1, cap at 20 inliers
                
                # X constraint: pos_dst_x - pos_src_x = dx
                A[constraint_idx * 2, dst * 2] = 1.0
                A[constraint_idx * 2, src * 2] = -1.0
                b[constraint_idx * 2] = dx
                weights[constraint_idx * 2] = weight
                
                # Y constraint: pos_dst_y - pos_src_y = dy
                A[constraint_idx * 2 + 1, dst * 2 + 1] = 1.0
                A[constraint_idx * 2 + 1, src * 2 + 1] = -1.0
                b[constraint_idx * 2 + 1] = dy
                weights[constraint_idx * 2 + 1] = weight
                
                constraint_idx += 1
            
            # Anchor: node 0 at origin (high weight)
            A[n_constraints * 2, 0] = 1.0  # x0 = 0
            A[n_constraints * 2 + 1, 1] = 1.0  # y0 = 0
            b[n_constraints * 2] = 0.0
            b[n_constraints * 2 + 1] = 0.0
            weights[n_constraints * 2] = 10.0  # High weight for anchor
            weights[n_constraints * 2 + 1] = 10.0
            
            # Apply weights: W * A * x = W * b
            W = np.diag(weights)
            A_weighted = W @ A
            b_weighted = W @ b
            
            # Solve weighted least-squares
            try:
                result = lsqr(A_weighted, b_weighted, atol=1e-6, btol=1e-6)
                positions_xy = result[0].reshape(n_nodes, 2)
                
                # Combine with angles
                initial_positions = np.column_stack([
                    positions_xy,
                    np.array(node_angles)
                ])
                
                # Compute position statistics
                pos_range_x = positions_xy[:, 0].max() - positions_xy[:, 0].min()
                pos_range_y = positions_xy[:, 1].max() - positions_xy[:, 1].min()
                pos_std_x = np.std(positions_xy[:, 0])
                pos_std_y = np.std(positions_xy[:, 1])
                
                print(f"[MASTER MAP] Solved {n_nodes} positions using least-squares from {n_constraints} transforms")
                print(f"[MASTER MAP] Position range: x=[{positions_xy[:, 0].min():.1f}, {positions_xy[:, 0].max():.1f}] "
                      f"(span={pos_range_x:.1f}px, std={pos_std_x:.1f}), "
                      f"y=[{positions_xy[:, 1].min():.1f}, {positions_xy[:, 1].max():.1f}] "
                      f"(span={pos_range_y:.1f}px, std={pos_std_y:.1f})")
                
                # Check if positions seem reasonable
                if pos_range_x > fragment_size * 10 or pos_range_y > fragment_size * 10:
                    print(f"[MASTER MAP] WARNING: Very large position range detected! "
                          f"This may indicate transform errors or misalignment.")
                if pos_std_x > fragment_size * 3 or pos_std_y > fragment_size * 3:
                    print(f"[MASTER MAP] WARNING: High position variance detected! "
                          f"This may indicate inconsistent transforms.")
            except Exception as e:
                print(f"[MASTER MAP] Least-squares solve failed: {e}, using BFS fallback")
                # Fallback to BFS
                initial_positions[0] = [0.0, 0.0, node_angles[0]]
                positioned = {0}
                queue = [0]
                while queue:
                    i = queue.pop(0)
                    for (src, dst) in transform_translations.keys():
                        if src == i and dst not in positioned:
                            dx, dy = transform_translations[(src, dst)]
                            initial_positions[dst] = [
                                initial_positions[src][0] + dx,
                                initial_positions[src][1] + dy,
                                node_angles[dst]
                            ]
                            positioned.add(dst)
                            queue.append(dst)
                
                for i in range(n_nodes):
                    if i not in positioned:
                        initial_positions[i] = [0.0, 0.0, node_angles[i]]
                        print(f"[MASTER MAP] WARNING: Node {i} not reachable, positioned at origin")
        else:
            print("[MASTER MAP] ERROR: No transforms available for positioning")
            return None
        
        print(f"[MASTER MAP] Initial positions range: x=[{initial_positions[:, 0].min():.1f}, {initial_positions[:, 0].max():.1f}], "
              f"y=[{initial_positions[:, 1].min():.1f}, {initial_positions[:, 1].max():.1f}]")
        
        # Summary statistics
        print(f"[MASTER MAP] Matching summary:")
        print(f"  - Total node pairs: {n_nodes * (n_nodes - 1) // 2}")
        print(f"  - Valid transforms: {len(pairwise_transforms)}")
        print(f"  - Average inliers per transform: {np.mean(list(match_counts.values())):.1f}")
        print(f"  - Transform coverage: {len(pairwise_transforms) / max(1, n_nodes - 1) * 100:.1f}% of minimum required")
        
        # 6. Bundle Adjustment - use simpler transform-based optimization
        # Feature-based BA is too complex and error-prone for this use case
        if n_nodes < 5:
            print(f"[MASTER MAP] Skipping bundle adjustment for {n_nodes} nodes (too few for reliable optimization)")
            optimized_positions = initial_positions
        else:
            print("[MASTER MAP] Optimizing with Bundle Adjustment (transform-based)...")
            ba = BundleAdjustment(
                max_iterations=BUNDLE_ADJUSTMENT_MAX_ITERS,
                ftol=BUNDLE_ADJUSTMENT_FTOL,
                xtol=BUNDLE_ADJUSTMENT_XTOL,
                gtol=BUNDLE_ADJUSTMENT_GTOL
            )
            
            # Use simpler transform-based optimization (more stable)
            try:
                optimized_positions = ba.optimize_simple(initial_positions, pairwise_transforms)
                
                # Validate optimization didn't make things worse
                initial_span = np.max(initial_positions[:, :2], axis=0) - np.min(initial_positions[:, :2], axis=0)
                optimized_span = np.max(optimized_positions[:, :2], axis=0) - np.min(optimized_positions[:, :2], axis=0)
                
                # Check if optimization created extreme values, NaN, or excessive expansion
                if (np.any(np.isnan(optimized_positions)) or 
                    np.any(np.abs(optimized_positions[:, :2]) > 10000) or
                    np.any(optimized_span > initial_span * 5) or
                    np.any(optimized_span < initial_span * 0.1)):  # Also check for collapse
                    print("[MASTER MAP] Bundle adjustment produced invalid results, using initial positions")
                    optimized_positions = initial_positions
                else:
                    # Check if optimization actually improved (reduced spread or kept it similar)
                    span_ratio = optimized_span / (initial_span + 1e-6)
                    if np.all(span_ratio < 3.0) and np.all(span_ratio > 0.3):  # Reasonable bounds
                        print(f"[MASTER MAP] Optimized positions range: x=[{optimized_positions[:, 0].min():.1f}, {optimized_positions[:, 0].max():.1f}], "
                              f"y=[{optimized_positions[:, 1].min():.1f}, {optimized_positions[:, 1].max():.1f}]")
                    else:
                        print("[MASTER MAP] Bundle adjustment produced unreasonable spread, using initial positions")
                        optimized_positions = initial_positions
            except Exception as e:
                print(f"[MASTER MAP] Bundle adjustment failed: {e}")
                print("[MASTER MAP] Using initial positions (no optimization)")
                optimized_positions = initial_positions
        
        # 7. Generate master map using optimized positions
        print("[MASTER MAP] Stitching master map...")
        # Use full-sized minimap fragments (400x400) instead of cropped
        # fragment_size already defined earlier
        fragment_mask = np.zeros((fragment_size, fragment_size), dtype=np.uint8)
        cv2.circle(fragment_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), 
                  self.MINIMAP_RADIUS, (255), -1)
        
        # Determine canvas size based on full fragments
        pts = optimized_positions[:, :2]  # x, y only
        min_x, min_y = np.min(pts, axis=0) - fragment_size//2
        max_x, max_y = np.max(pts, axis=0) + fragment_size//2
        
        padding = self.MASTER_MAP_PADDING
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)
        # Don't constrain maximum size - use full canvas
        canvas_w = max(canvas_w, self.MASTER_MAP_MIN_SIZE)
        canvas_h = max(canvas_h, self.MASTER_MAP_MIN_SIZE)
        
        print(f"[MASTER MAP] Canvas size: {canvas_w}x{canvas_h} (fragments: {fragment_size}x{fragment_size})")
        
        master_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        coverage_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # Stitch nodes using optimized positions
        # Use FILTERED images for display (clean visual appearance)
        centers = []
        fragment_masks_used = []  # Store masks used for each fragment
        for i, (img_filtered, node) in enumerate(zip(node_images_filtered, nodes)):
            # Use full-sized minimap (already North-aligned and filtered)
            # img_filtered is already 400x400 circular region with blue paths filtered
            
            # Verify image size matches expected fragment size
            if img_filtered.shape[:2] != (fragment_size, fragment_size):
                print(f"[MASTER MAP] WARNING: Fragment {i} size mismatch! Expected {fragment_size}x{fragment_size}, got {img_filtered.shape[:2]}")
                # Resize to expected size
                img_filtered = cv2.resize(img_filtered, (fragment_size, fragment_size), interpolation=cv2.INTER_LINEAR)
            
            # CRITICAL: Ensure fragment is properly masked to remove border artifacts
            # First, mask center artifacts (player arrow area)
            center_masked = self.mask_center(img_filtered, radius=self.CENTER_MASK_RADIUS_MASTER_MAP)
            
            # Verify fragment_mask matches the image size
            if center_masked.shape[:2] != fragment_mask.shape[:2]:
                print(f"[MASTER MAP] ERROR: Fragment {i} mask size mismatch! Image: {center_masked.shape[:2]}, Mask: {fragment_mask.shape[:2]}")
                # This shouldn't happen, but resize mask if needed
                mask_to_use = cv2.resize(fragment_mask, (center_masked.shape[1], center_masked.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_to_use = fragment_mask.copy()
            
            # Store mask for later use in blending
            fragment_masks_used.append(mask_to_use)
            
            # Apply circular mask STRICTLY - this removes ALL pixels outside the circle
            # Create fragment with explicit masking: anything outside circle = black
            final_fragment = np.zeros_like(center_masked, dtype=np.uint8)
            # Only copy pixels that are inside the circular mask
            final_fragment[mask_to_use > 0] = center_masked[mask_to_use > 0]
            
            # Final verification: ensure no non-zero pixels outside circle
            outside_mask = (mask_to_use == 0)
            pixels_outside = np.any(final_fragment[outside_mask] > 0) if np.any(outside_mask) else False
            if pixels_outside:
                num_bad_pixels = np.sum(final_fragment[outside_mask] > 0) if np.any(outside_mask) else 0
                print(f"[MASTER MAP] WARNING: Found {num_bad_pixels} pixels outside circle in fragment {i}, zeroing them")
                final_fragment[outside_mask] = 0
            
            # Placement using optimized coordinates
            wx, wy = optimized_positions[i, :2]
            tx = int(wx - min_x - fragment_size//2)
            ty = int(wy - min_y - fragment_size//2)
            
            # Place fragment with blending (same as legacy method)
            tx_clipped = max(0, min(tx, canvas_w - 1))
            ty_clipped = max(0, min(ty, canvas_h - 1))
            tx_end = min(tx + fragment_size, canvas_w)
            ty_end = min(ty + fragment_size, canvas_h)
            
            frag_h = ty_end - ty_clipped
            frag_w = tx_end - tx_clipped
            
            if frag_h > 0 and frag_w > 0:
                frag_y1 = ty_clipped - ty
                frag_y2 = frag_y1 + frag_h
                frag_x1 = tx_clipped - tx
                frag_x2 = frag_x1 + frag_w
                
                visible_fragment = final_fragment[frag_y1:frag_y2, frag_x1:frag_x2]

                # Extract the corresponding portion of the circular mask
                # Use the mask that was stored for this fragment
                mask_for_fragment = fragment_masks_used[i]
                frag_mask_portion = mask_for_fragment[frag_y1:frag_y2, frag_x1:frag_x2]
                within_circle = (frag_mask_portion > 0).astype(np.float32)
                
                # CRITICAL: Strictly enforce mask on visible fragment - zero any pixels outside circle
                # This prevents border artifacts from being blended into the final map
                outside_circle_mask = (frag_mask_portion == 0)
                if np.any(outside_circle_mask):
                    visible_fragment[outside_circle_mask] = 0
                
                # Distance-based weights (for blending multiple overlapping fragments)
                # Use actual circular center relative to fragment origin
                actual_center_x = self.MINIMAP_CENTER_X - frag_x1
                actual_center_y = self.MINIMAP_CENTER_Y - frag_y1
                y_coords, x_coords = np.ogrid[0:frag_h, 0:frag_w]
                center_dist = np.sqrt((x_coords - actual_center_x)**2 + (y_coords - actual_center_y)**2)
                max_dist = self.MINIMAP_RADIUS
                if max_dist > 0:
                    weights = np.maximum(0, 1.0 - center_dist / max_dist)
                else:
                    weights = np.ones((frag_h, frag_w), dtype=np.float32)
                
                # CRITICAL: Weights must be zero outside the circle
                weights = weights * within_circle
                
                region = master_canvas[ty_clipped:ty_end, tx_clipped:tx_end]
                region_coverage = coverage_map[ty_clipped:ty_end, tx_clipped:tx_end]
                
                # Create mask: fragment has content AND is within circular mask
                has_content = (visible_fragment.sum(axis=2) > 0).astype(np.float32)
                mask = has_content * within_circle  # Both conditions must be true
                
                # Final weights: distance-based blending, but only within circle
                new_weights = weights * mask
                
                total_weights = region_coverage + new_weights
                total_weights = np.maximum(total_weights, 1e-6)
                
                for c in range(3):
                    region_channel = region[:, :, c].astype(np.float32)
                    frag_channel = visible_fragment[:, :, c].astype(np.float32)
                    blended_channel = (region_channel * region_coverage + frag_channel * new_weights) / total_weights
                    region[:, :, c] = blended_channel.astype(np.uint8)
                
                coverage_map[ty_clipped:ty_end, tx_clipped:tx_end] = total_weights
            
            centers.append((int(wx - min_x), int(wy - min_y)))
        
        # Draw breadcrumbs
        if len(centers) > 1:
            for i in range(len(centers) - 1):
                cv2.line(master_canvas, centers[i], centers[i+1], self.COLOR_ORANGE, 2)
            cv2.circle(master_canvas, centers[0], 8, self.COLOR_GREEN, -1)
            cv2.circle(master_canvas, centers[-1], 8, self.COLOR_RED, -1)
        
        # Find bounding box of actual content (non-black regions)
        # Crop to actual content to remove large black borders
        gray = cv2.cvtColor(master_canvas, cv2.COLOR_BGR2GRAY)
        non_zero = np.where(gray > 0)
        if len(non_zero[0]) > 0:
            min_y, max_y = non_zero[0].min(), non_zero[0].max()
            min_x, max_x = non_zero[1].min(), non_zero[1].max()
            content_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            print(f"[MASTER MAP] Content bounding box: x={min_x}, y={min_y}, w={max_x-min_x}, h={max_y-min_y}")
            print(f"[MASTER MAP] Canvas size: {canvas_w}x{canvas_h}")
            
            # Crop to content with small padding to avoid edge artifacts
            crop_padding = 10
            crop_x1 = max(0, min_x - crop_padding)
            crop_y1 = max(0, min_y - crop_padding)
            crop_x2 = min(canvas_w, max_x + crop_padding)
            crop_y2 = min(canvas_h, max_y + crop_padding)
            
            master_canvas = master_canvas[crop_y1:crop_y2, crop_x1:crop_x2]
            print(f"[MASTER MAP] Cropped to content: {crop_x2-crop_x1}x{crop_y2-crop_y1} (removed {canvas_w-(crop_x2-crop_x1)}x{canvas_h-(crop_y2-crop_y1)} black borders)")
        else:
            print("[MASTER MAP] WARNING: No content found in canvas!")
        
        print("[MASTER MAP] Advanced stitching complete.")
        return master_canvas
    
    def _generate_master_map_legacy(self, nodes: list, landmark_dir: str) -> Optional[np.ndarray]:
        """
        Legacy master map generation using cumulative coordinates.
        
        This is the original implementation kept for backward compatibility.
        """
        if not nodes:
            return None

        # 1. Calculate Cumulative World Coordinates (North-Up Frame) with missing offset handling
        world_coords = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0
        missing_count = 0
        
        for i in range(1, len(nodes)):
            offset = nodes[i].get('relative_offset')
            if offset and offset.get('dx') is not None and offset.get('dy') is not None:
                dx, dy = offset['dx'], offset['dy']
                curr_x += dx
                curr_y += dy
                missing_count = 0  # Reset counter on valid offset
            else:
                # Missing offset - use zero (position doesn't advance)
                missing_count += 1
                if missing_count > 3:
                    print(f"[MASTER MAP] Warning: Multiple consecutive missing offsets at node {i}")
            world_coords.append((curr_x, curr_y))

        # 2. Validate coordinate system
        pts = np.array(world_coords)
        if len(pts) > 0:
            print(f"[MASTER MAP] Coordinate range: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], "
                  f"y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
            
            # Check for NaN or extreme values
            if np.any(np.isnan(pts)) or np.any(np.abs(pts) > 100000):
                print("[MASTER MAP] Warning: Invalid coordinates detected!")

        # 3. Determine Canvas Size
        crop_size = self.MASTER_MAP_CROP_SIZE
        
        crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        cv2.circle(crop_mask, (crop_size//2, crop_size//2), crop_size//2 - self.MASTER_MAP_CIRCLE_BORDER, (255), -1)

        min_x, min_y = np.min(pts, axis=0) - crop_size//2
        max_x, max_y = np.max(pts, axis=0) + crop_size//2

        padding = self.MASTER_MAP_PADDING
        min_x -= padding; min_y -= padding
        max_x += padding; max_y += padding
        
        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)
        
        canvas_w = min(max(canvas_w, self.MASTER_MAP_MIN_SIZE), self.MASTER_MAP_MAX_SIZE)
        canvas_h = min(max(canvas_h, self.MASTER_MAP_MIN_SIZE), self.MASTER_MAP_MAX_SIZE)

        master_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Create coverage tracking for alpha blending
        coverage_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # 4. Stitch North-Aligned Nodes
        centers = []
        for i, node in enumerate(nodes):
            path = os.path.join(landmark_dir, node['minimap_path'])
            if not os.path.exists(path): continue
                
            img = cv2.imread(path)
            if img is None: continue
            
            # Step 1: Stretch (if not already stretched)
            # NOTE: Images should already be stretched from recording, but handle legacy unstretched images
            h, w = img.shape[:2]
            if h != self.MINIMAP_STRETCHED_HEIGHT or w != self.MINIMAP_STRETCHED_WIDTH:
                # Assume it's 400x400 raw, stretch to 400x520
                img = cv2.resize(img, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # Step 2: Rotate stretched image to North BEFORE cropping (same as compute_drift_pc)
            # This ensures proper alignment - rotation must happen on full stretched image
            arrow_angle = node.get('arrow_angle', 0.0)
            # Get pivot in stretched space (arrow_pivot is stored in stretched coordinates)
            default_pivot = (self.MINIMAP_CENTER_STRETCHED_X, self.MINIMAP_CENTER_STRETCHED_Y)  # (200, 260)
            pivot_stretched = node.get('arrow_pivot', default_pivot)
            
            # Rotate to align to North (same approach as compute_drift_pc)
            # In compute_drift_pc, rotation is applied by arrow_angle directly
            # arrow_angle is the angle the arrow is pointing (0=Up/North, positive=CW)
            # OpenCV's getRotationMatrix2D rotates counter-clockwise for positive angles
            # To align to North, we rotate CCW by arrow_angle (same as compute_drift_pc)
            R = cv2.getRotationMatrix2D(pivot_stretched, arrow_angle, 1.0)
            img_rotated = cv2.warpAffine(img, R, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT))
            
            # Step 3: Crop 400x400 circle from rotated stretched image
            # Extract region centered at (200, 260): x=0-400, y=60-460
            crop_y1, crop_y2 = self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2
            img_circle = img_rotated[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
            
            # Apply circular mask to the 400x400 region
            circle_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
            cv2.circle(circle_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)
            img_north = cv2.bitwise_and(img_circle, img_circle, mask=circle_mask)
            
            # Step 4: Apply filtering for master map - only blue/cyan paths, exclude gold and red/alert
            # Gold is the player arrow (transient), red/alert is enemy state (transient)
            # For master map, we only want the permanent blue/cyan path features
            hsv = cv2.cvtColor(img_north, cv2.COLOR_BGR2HSV)
            
            # Only use blue/cyan range (exclude gold and red/alert)
            lower_blue = np.array(self.blue_lower)
            upper_blue = np.array(self.blue_upper)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Apply dynamic circle masking to remove enemy dots and AI team circles
            circle_mask = self._mask_dynamic_circles(img_north)
            combined_mask = cv2.bitwise_and(mask_blue, circle_mask)
            color_filtered = cv2.bitwise_and(img_north, img_north, mask=combined_mask)
            
            # Get edge detection result
            edge_result = self._apply_edge_detection(img_north)
            
            # Filter edges to only keep those near color-filtered regions (removes gridlines)
            # This ensures we only keep edges that correspond to actual map features, not UI gridlines
            color_mask_gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
            color_mask_binary = (color_mask_gray > 0).astype(np.uint8)
            # Dilate the color mask slightly to include edges near color features
            kernel_dilate = np.ones((3, 3), np.uint8)
            color_mask_dilated = cv2.dilate(color_mask_binary, kernel_dilate, iterations=2)
            # Only keep edges that are near color-filtered regions
            edge_gray = cv2.cvtColor(edge_result, cv2.COLOR_BGR2GRAY)
            edge_masked = cv2.bitwise_and(edge_gray, edge_gray, mask=color_mask_dilated)
            edge_result_filtered = cv2.cvtColor(edge_masked, cv2.COLOR_GRAY2BGR)
            
            # Combine color filter and filtered edge detection (same as route following)
            blue_only = cv2.max(color_filtered, edge_result_filtered)
            
            # Apply morphological operations to clean up noise while preserving colors
            # Create a binary mask from non-black pixels (where we have filtered content)
            gray_mask = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)
            binary_mask = (gray_mask > 0).astype(np.uint8) * 255
            
            # Clean up the mask: remove small noise, fill small gaps
            kernel_small = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            kernel_close = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
            # Apply cleaned mask to preserve colors while removing noise
            blue_only = cv2.bitwise_and(blue_only, blue_only, mask=binary_mask)
            
            # Step 5: Crop MASTER_MAP_CROP_SIZE subsection around center
            crop_x1, crop_y1 = self.MINIMAP_CENTER_X - crop_size//2, self.MINIMAP_CENTER_Y - crop_size//2
            crop_x2, crop_y2 = self.MINIMAP_CENTER_X + crop_size//2, self.MINIMAP_CENTER_Y + crop_size//2
            cropped = blue_only[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Mask out center artifacts
            center_masked_crop = self.mask_center(cropped, radius=self.CENTER_MASK_RADIUS_MASTER_MAP)
            final_fragment = cv2.bitwise_and(center_masked_crop, center_masked_crop, mask=crop_mask)
            
            # Placement in Canvas
            wx, wy = world_coords[i]
            tx = int(wx - min_x - crop_size//2)
            ty = int(wy - min_y - crop_size//2)
            
            # Allow partial placement near boundaries
            tx_clipped = max(0, min(tx, canvas_w - 1))
            ty_clipped = max(0, min(ty, canvas_h - 1))
            tx_end = min(tx + crop_size, canvas_w)
            ty_end = min(ty + crop_size, canvas_h)
            
            frag_h = ty_end - ty_clipped
            frag_w = tx_end - tx_clipped
            
            if frag_h > 0 and frag_w > 0:
                # Extract visible portion of fragment
                frag_y1 = ty_clipped - ty
                frag_y2 = frag_y1 + frag_h
                frag_x1 = tx_clipped - tx
                frag_x2 = frag_x1 + frag_w
                
                visible_fragment = final_fragment[frag_y1:frag_y2, frag_x1:frag_x2]
                
                # Create distance-based weight (higher weight near center)
                y_coords, x_coords = np.ogrid[0:frag_h, 0:frag_w]
                # Calculate distance from fragment center (accounting for clipping)
                frag_center_x = frag_w // 2
                frag_center_y = frag_h // 2
                center_dist = np.sqrt((x_coords - frag_center_x)**2 + (y_coords - frag_center_y)**2)
                max_dist = min(frag_w, frag_h) // 2
                if max_dist > 0:
                    weights = np.maximum(0, 1.0 - center_dist / max_dist)
                else:
                    weights = np.ones((frag_h, frag_w), dtype=np.float32)
                
                # Extract region
                region = master_canvas[ty_clipped:ty_end, tx_clipped:tx_end]
                region_coverage = coverage_map[ty_clipped:ty_end, tx_clipped:tx_end]
                
                # Create mask for non-zero pixels in fragment
                mask = (visible_fragment.sum(axis=2) > 0).astype(np.float32)
                new_weights = weights * mask
                
                # Blend: weighted average where both exist, direct placement where empty
                total_weights = region_coverage + new_weights
                total_weights = np.maximum(total_weights, 1e-6)  # Avoid division by zero
                
                # Weighted blend for each channel
                for c in range(3):
                    region_channel = region[:, :, c].astype(np.float32)
                    frag_channel = visible_fragment[:, :, c].astype(np.float32)
                    blended_channel = (region_channel * region_coverage + frag_channel * new_weights) / total_weights
                    region[:, :, c] = blended_channel.astype(np.uint8)
                
                # Update coverage
                coverage_map[ty_clipped:ty_end, tx_clipped:tx_end] = total_weights
            
            centers.append((int(wx - min_x), int(wy - min_y)))

        # 5. Draw Breadcrumbs
        if len(centers) > 1:
            for i in range(len(centers) - 1):
                cv2.line(master_canvas, centers[i], centers[i+1], self.COLOR_ORANGE, 2)
            cv2.circle(master_canvas, centers[0], 8, self.COLOR_GREEN, -1)  # Start
            cv2.circle(master_canvas, centers[-1], 8, self.COLOR_RED, -1)  # End
        
        # 6. Downsample
        final_map = cv2.resize(master_canvas, (0, 0), fx=self.MASTER_MAP_DOWNSAMPLE_FACTOR, fy=self.MASTER_MAP_DOWNSAMPLE_FACTOR, interpolation=cv2.INTER_AREA)
        return final_map

    def generate_master_map(self, nodes: list, landmark_dir: str) -> Optional[np.ndarray]:
        """
        Generates a composite master map from a list of hybrid nodes using North-Aligned SLAM.
        Stitches minimap images together using recorded world-space translations.
        
        Uses advanced stitching pipeline (SuperPoint + LightGlue + Bundle Adjustment) if enabled,
        otherwise falls back to legacy cumulative coordinate method.
        """
        if not nodes:
            return None
        
        if ENABLE_ADVANCED_STITCHING:
            return self._generate_master_map_advanced(nodes, landmark_dir)
        else:
            return self._generate_master_map_legacy(nodes, landmark_dir)

        # 1. Calculate Cumulative World Coordinates (North-Up Frame) with missing offset handling
        world_coords = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0
        missing_count = 0
        
        for i in range(1, len(nodes)):
            offset = nodes[i].get('relative_offset')
            if offset and offset.get('dx') is not None and offset.get('dy') is not None:
                dx, dy = offset['dx'], offset['dy']
                curr_x += dx
                curr_y += dy
                missing_count = 0  # Reset counter on valid offset
            else:
                # Missing offset - use zero (position doesn't advance)
                missing_count += 1
                if missing_count > 3:
                    print(f"[MASTER MAP] Warning: Multiple consecutive missing offsets at node {i}")
            world_coords.append((curr_x, curr_y))

        # 2. Validate coordinate system
        pts = np.array(world_coords)
        if len(pts) > 0:
            print(f"[MASTER MAP] Coordinate range: x=[{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], "
                  f"y=[{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
            
            # Check for NaN or extreme values
            if np.any(np.isnan(pts)) or np.any(np.abs(pts) > 100000):
                print("[MASTER MAP] Warning: Invalid coordinates detected!")

        # 3. Determine Canvas Size
        crop_size = self.MASTER_MAP_CROP_SIZE
        
        crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        cv2.circle(crop_mask, (crop_size//2, crop_size//2), crop_size//2 - self.MASTER_MAP_CIRCLE_BORDER, (255), -1)

        min_x, min_y = np.min(pts, axis=0) - crop_size//2
        max_x, max_y = np.max(pts, axis=0) + crop_size//2

        padding = self.MASTER_MAP_PADDING
        min_x -= padding; min_y -= padding
        max_x += padding; max_y += padding
        
        canvas_w = int(max_x - min_x)
        canvas_h = int(max_y - min_y)
        
        canvas_w = min(max(canvas_w, self.MASTER_MAP_MIN_SIZE), self.MASTER_MAP_MAX_SIZE)
        canvas_h = min(max(canvas_h, self.MASTER_MAP_MIN_SIZE), self.MASTER_MAP_MAX_SIZE)

        master_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Create coverage tracking for alpha blending
        coverage_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        
        # 4. Stitch North-Aligned Nodes
        centers = []
        for i, node in enumerate(nodes):
            path = os.path.join(landmark_dir, node['minimap_path'])
            if not os.path.exists(path): continue
                
            img = cv2.imread(path)
            if img is None: continue
            
            # Step 1: Stretch (if not already stretched)
            # NOTE: Images should already be stretched from recording, but handle legacy unstretched images
            h, w = img.shape[:2]
            if h != self.MINIMAP_STRETCHED_HEIGHT or w != self.MINIMAP_STRETCHED_WIDTH:
                # Assume it's 400x400 raw, stretch to 400x520
                img = cv2.resize(img, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # Step 2: Rotate stretched image to North BEFORE cropping (same as compute_drift_pc)
            # This ensures proper alignment - rotation must happen on full stretched image
            arrow_angle = node.get('arrow_angle', 0.0)
            # Get pivot in stretched space (arrow_pivot is stored in stretched coordinates)
            default_pivot = (self.MINIMAP_CENTER_STRETCHED_X, self.MINIMAP_CENTER_STRETCHED_Y)  # (200, 260)
            pivot_stretched = node.get('arrow_pivot', default_pivot)
            
            # Rotate to align to North (same approach as compute_drift_pc)
            # In compute_drift_pc, rotation is applied by arrow_angle directly
            # arrow_angle is the angle the arrow is pointing (0=Up/North, positive=CW)
            # OpenCV's getRotationMatrix2D rotates counter-clockwise for positive angles
            # To align to North, we rotate CCW by arrow_angle (same as compute_drift_pc)
            R = cv2.getRotationMatrix2D(pivot_stretched, arrow_angle, 1.0)
            img_rotated = cv2.warpAffine(img, R, (self.MINIMAP_STRETCHED_WIDTH, self.MINIMAP_STRETCHED_HEIGHT))
            
            # Step 3: Crop 400x400 circle from rotated stretched image
            # Extract region centered at (200, 260): x=0-400, y=60-460
            crop_y1, crop_y2 = self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2
            img_circle = img_rotated[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
            
            # Apply circular mask to the 400x400 region
            circle_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
            cv2.circle(circle_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)
            img_north = cv2.bitwise_and(img_circle, img_circle, mask=circle_mask)
            
            # Step 4: Apply filtering for master map - only blue/cyan paths, exclude gold and red/alert
            # Gold is the player arrow (transient), red/alert is enemy state (transient)
            # For master map, we only want the permanent blue/cyan path features
            hsv = cv2.cvtColor(img_north, cv2.COLOR_BGR2HSV)
            
            # Only use blue/cyan range (exclude gold and red/alert)
            lower_blue = np.array(self.blue_lower)
            upper_blue = np.array(self.blue_upper)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Apply dynamic circle masking to remove enemy dots and AI team circles
            circle_mask = self._mask_dynamic_circles(img_north)
            combined_mask = cv2.bitwise_and(mask_blue, circle_mask)
            color_filtered = cv2.bitwise_and(img_north, img_north, mask=combined_mask)
            
            # Get edge detection result
            edge_result = self._apply_edge_detection(img_north)
            
            # Filter edges to only keep those near color-filtered regions (removes gridlines)
            # This ensures we only keep edges that correspond to actual map features, not UI gridlines
            color_mask_gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
            color_mask_binary = (color_mask_gray > 0).astype(np.uint8)
            # Dilate the color mask slightly to include edges near color features
            kernel_dilate = np.ones((3, 3), np.uint8)
            color_mask_dilated = cv2.dilate(color_mask_binary, kernel_dilate, iterations=2)
            # Only keep edges that are near color-filtered regions
            edge_gray = cv2.cvtColor(edge_result, cv2.COLOR_BGR2GRAY)
            edge_masked = cv2.bitwise_and(edge_gray, edge_gray, mask=color_mask_dilated)
            edge_result_filtered = cv2.cvtColor(edge_masked, cv2.COLOR_GRAY2BGR)
            
            # Combine color filter and filtered edge detection (same as route following)
            blue_only = cv2.max(color_filtered, edge_result_filtered)
            
            # Apply morphological operations to clean up noise while preserving colors
            # Create a binary mask from non-black pixels (where we have filtered content)
            gray_mask = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)
            binary_mask = (gray_mask > 0).astype(np.uint8) * 255
            
            # Clean up the mask: remove small noise, fill small gaps
            kernel_small = np.ones((2, 2), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            kernel_close = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            
            # Apply cleaned mask to preserve colors while removing noise
            blue_only = cv2.bitwise_and(blue_only, blue_only, mask=binary_mask)
            
            # Step 5: Crop MASTER_MAP_CROP_SIZE subsection around center
            crop_x1, crop_y1 = self.MINIMAP_CENTER_X - crop_size//2, self.MINIMAP_CENTER_Y - crop_size//2
            crop_x2, crop_y2 = self.MINIMAP_CENTER_X + crop_size//2, self.MINIMAP_CENTER_Y + crop_size//2
            cropped = blue_only[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Mask out center artifacts
            center_masked_crop = self.mask_center(cropped, radius=self.CENTER_MASK_RADIUS_MASTER_MAP)
            final_fragment = cv2.bitwise_and(center_masked_crop, center_masked_crop, mask=crop_mask)
            
            # Placement in Canvas
            wx, wy = world_coords[i]
            tx = int(wx - min_x - crop_size//2)
            ty = int(wy - min_y - crop_size//2)
            
            # Allow partial placement near boundaries
            tx_clipped = max(0, min(tx, canvas_w - 1))
            ty_clipped = max(0, min(ty, canvas_h - 1))
            tx_end = min(tx + crop_size, canvas_w)
            ty_end = min(ty + crop_size, canvas_h)
            
            frag_h = ty_end - ty_clipped
            frag_w = tx_end - tx_clipped
            
            if frag_h > 0 and frag_w > 0:
                # Extract visible portion of fragment
                frag_y1 = ty_clipped - ty
                frag_y2 = frag_y1 + frag_h
                frag_x1 = tx_clipped - tx
                frag_x2 = frag_x1 + frag_w
                
                visible_fragment = final_fragment[frag_y1:frag_y2, frag_x1:frag_x2]
                
                # Create distance-based weight (higher weight near center)
                y_coords, x_coords = np.ogrid[0:frag_h, 0:frag_w]
                # Calculate distance from fragment center (accounting for clipping)
                frag_center_x = frag_w // 2
                frag_center_y = frag_h // 2
                center_dist = np.sqrt((x_coords - frag_center_x)**2 + (y_coords - frag_center_y)**2)
                max_dist = min(frag_w, frag_h) // 2
                if max_dist > 0:
                    weights = np.maximum(0, 1.0 - center_dist / max_dist)
                else:
                    weights = np.ones((frag_h, frag_w), dtype=np.float32)
                
                # Extract region
                region = master_canvas[ty_clipped:ty_end, tx_clipped:tx_end]
                region_coverage = coverage_map[ty_clipped:ty_end, tx_clipped:tx_end]
                
                # Create mask for non-zero pixels in fragment
                mask = (visible_fragment.sum(axis=2) > 0).astype(np.float32)
                new_weights = weights * mask
                
                # Blend: weighted average where both exist, direct placement where empty
                total_weights = region_coverage + new_weights
                total_weights = np.maximum(total_weights, 1e-6)  # Avoid division by zero
                
                # Weighted blend for each channel
                for c in range(3):
                    region_channel = region[:, :, c].astype(np.float32)
                    frag_channel = visible_fragment[:, :, c].astype(np.float32)
                    blended_channel = (region_channel * region_coverage + frag_channel * new_weights) / total_weights
                    region[:, :, c] = blended_channel.astype(np.uint8)
                
                # Update coverage
                coverage_map[ty_clipped:ty_end, tx_clipped:tx_end] = total_weights
            
            centers.append((int(wx - min_x), int(wy - min_y)))

        # 5. Draw Breadcrumbs
        if len(centers) > 1:
            for i in range(len(centers) - 1):
                cv2.line(master_canvas, centers[i], centers[i+1], self.COLOR_ORANGE, 2)
            cv2.circle(master_canvas, centers[0], 8, self.COLOR_GREEN, -1)  # Start
            cv2.circle(master_canvas, centers[-1], 8, self.COLOR_RED, -1)  # End
        
        # 6. Downsample
        final_map = cv2.resize(master_canvas, (0, 0), fx=self.MASTER_MAP_DOWNSAMPLE_FACTOR, fy=self.MASTER_MAP_DOWNSAMPLE_FACTOR, interpolation=cv2.INTER_AREA)
        return final_map

    def _prepare_debug_canvas(self, current_minimap: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
        """
        Prepares the debug canvas with proper dimensions.
        
        Returns:
            (canvas, canvas_h, canvas_w, minimap_h, minimap_w)
        """
        h, w = current_minimap.shape[:2]
        
        ctrl_h = self.DEBUG_CTRL_HEIGHT
        hist_h = self.DEBUG_HIST_HEIGHT
        chart_h = self.DEBUG_CHART_HEIGHT
        crop_preview_h = self.DEBUG_CROP_PREVIEW_HEIGHT
        canvas_h = h + crop_preview_h + ctrl_h + hist_h + chart_h
        canvas_w = w * 2 + self.DEBUG_CANVAS_SPACING
        
        debug_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        return debug_canvas, canvas_h, canvas_w, h, w

    def _prepare_target_minimap(self, target_minimap: np.ndarray, current_minimap: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepares target minimap for display by ensuring it matches current minimap dimensions and is masked.
        NOTE: Target images are already stretched (400x520) from recording.
        
        Returns:
            Prepared target minimap or None
        """
        if target_minimap is None:
            return None
            
        target_h, target_w = target_minimap.shape[:2]
        current_h, current_w = current_minimap.shape[:2]
        
        if (target_h, target_w) == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH) and (current_h, current_w) == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH):
            # Both are stretched 400x520, just apply mask to target to match current_minimap
            target_minimap = cv2.bitwise_and(target_minimap, target_minimap, mask=self.mask_stretched)
        elif (target_h, target_w) != (current_h, current_w):
            # Dimensions don't match - resize target to match current
            target_minimap = cv2.resize(target_minimap, (current_w, current_h), interpolation=cv2.INTER_LINEAR)
            # Re-apply mask after resize if dimensions are now 400x520
            if target_minimap.shape[:2] == (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH):
                target_minimap = cv2.bitwise_and(target_minimap, target_minimap, mask=self.mask_stretched)
        
        return target_minimap

    def _draw_minimap_comparison(self, canvas: np.ndarray, target_minimap: np.ndarray, 
                                  current_minimap: np.ndarray, h: int, w: int, 
                                  tracking_active: bool, status_msg: str):
        """
        Draws the target and current minimap comparison in the top section of the debug canvas.
        Uses _filter_colors_only to match what the HSV debug "Combined" view shows.
        This shows only the color-filtered pixels without edge detection artifacts.
        """
        # Use _filter_colors_only to match the HSV debug "Combined" view
        # This shows only the color-filtered features without edge detection grid lines
        prepared_target = self._prepare_target_minimap(target_minimap, current_minimap)
        
        if prepared_target is not None:
            # Use _filter_colors_only to match HSV debug panel (no edge detection)
            # Use include_gold=True for visualization (gold arrow helps with visual reference)
            display_target = self._filter_colors_only(prepared_target, include_gold=True)
            target_h, target_w = display_target.shape[:2]
            # Resize to match current minimap dimensions if needed
            if (target_h, target_w) != (h, w):
                display_target = cv2.resize(display_target, (w, h))
            canvas[0:h, 0:w] = display_target
        else:
            display_target = None
        
        # Use _filter_colors_only to match HSV debug panel
        display_current = self._filter_colors_only(current_minimap, include_gold=True)
        canvas[0:h, w+self.DEBUG_CANVAS_SPACING:w*2+self.DEBUG_CANVAS_SPACING] = display_current
        
        # Overlay Info (Drift)
        info_text = f"dx: {self.last_dx:.2f}, dy: {self.last_dy:.2f}, ang: {self.last_angle:.2f}"
        if hasattr(self, 'last_inliers_count'):
             info_text += f", N: {self.last_inliers_count}"
        cv2.putText(canvas, info_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GREEN, 1)
        
        # Draw flow arrow on Current
        cx, cy = w + self.DEBUG_CANVAS_SPACING + w//2, h//2
        end_x = int(cx + self.last_dx)
        end_y = int(cy + self.last_dy)
        cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), self.COLOR_RED, 2)

        # Status Overlay
        if not tracking_active:
             msg = status_msg if status_msg else "LOST TRACKING"
             text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
             tx = (canvas.shape[1] - text_size[0]) // 2
             ty = h // 2
             cv2.putText(canvas, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.COLOR_RED, 2)

    def _calculate_circular_crop_bounds(self) -> Tuple[int, int]:
        """
        Calculates the Y bounds for extracting the 400x400 circular region from stretched 400x520 image.
        
        Returns:
            (crop_y1, crop_y2) - Y coordinates for the circular crop region
        """
        return self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2

    def _process_minimap_for_cropped_preview(self, minimap: np.ndarray) -> Optional[np.ndarray]:
        """
        Processes a minimap image to create a 225x225 cropped preview for comparison.
        This extracts the circular region, applies filtering, and crops to PHASE_CORR_CROP_SIZE.
        
        Args:
            minimap: Input minimap image (should be 400x520 stretched)
        
        Returns:
            225x225 cropped and filtered image, or None if processing fails
        """
        if minimap is None or minimap.shape[:2] != (self.MINIMAP_STRETCHED_HEIGHT, self.MINIMAP_STRETCHED_WIDTH):
            return None
        
        # Extract 400x400 circular region
        crop_y1, crop_y2 = self._calculate_circular_crop_bounds()
        crop_400 = minimap[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
        
        # Apply circular mask
        crop_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
        cv2.circle(crop_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)
        crop_400 = cv2.bitwise_and(crop_400, crop_400, mask=crop_mask)
        
        # Filter and process - use _filter_colors_only to match HSV debug panel (no edge detection)
        # Keep as BGR to preserve color information
        blue_crop = self._filter_colors_only(crop_400, include_gold=True)
        
        # Crop to PHASE_CORR_CROP_SIZE - don't apply center mask for preview (user wants to see full region)
        crop_size = self.PHASE_CORR_CROP_SIZE
        crop_offset = (self.CIRCULAR_CROP_SIZE - crop_size) // 2
        cropped = blue_crop[crop_offset:crop_offset+crop_size, crop_offset:crop_offset+crop_size]
        
        return cropped

    def _draw_cropped_preview(self, canvas: np.ndarray, current_minimap: np.ndarray, 
                              target_minimap: Optional[np.ndarray], h: int, w: int, 
                              canvas_w: int, crop_preview_h: int):
        """
        Draws the cropped preview section showing the 225x225 images used for comparison.
        Shows target node (Previous) vs current minimap (Current) for drift visualization.
        
        Args:
            canvas: Canvas to draw on
            current_minimap: Current real-time minimap (400x520 stretched)
            target_minimap: Target node's minimap (400x520 stretched) - this is what we're comparing against
            h, w: Minimap dimensions
            canvas_w: Canvas width
            crop_preview_h: Height allocated for cropped preview section
        """
        crop_preview_y_start = h
        crop_preview_y_end = h + crop_preview_h
        cv2.rectangle(canvas, (0, crop_preview_y_start), (canvas_w, crop_preview_y_end), (25, 25, 25), -1)
        
        # Process target minimap for "Previous" (target node we're trying to reach)
        cropped_prev = None
        if target_minimap is not None:
            # Prepare target minimap same way as in _draw_minimap_comparison
            prepared_target = self._prepare_target_minimap(target_minimap, current_minimap)
            if prepared_target is not None:
                cropped_prev = self._process_minimap_for_cropped_preview(prepared_target)
        
        # If target minimap not available, try using stored cropped preview from phase correlation
        if cropped_prev is None:
            cropped_prev = self.last_cropped_prev
        
        # Process current minimap for "Current" (real-time position)
        cropped_curr = self._process_minimap_for_cropped_preview(current_minimap)
        
        # If current processing failed, try using stored cropped preview
        if cropped_curr is None:
            cropped_curr = self.last_cropped_curr
        
        # If still no current, we can't display anything
        if cropped_curr is None:
            return
        
        # Draw cropped images - show current even if previous (target) is not available
        # Use cropped_curr for dimensions since we know it's not None
        crop_h, crop_w = cropped_curr.shape[:2]  # Should be PHASE_CORR_CROP_SIZE x PHASE_CORR_CROP_SIZE
        
        # Calculate available space for each preview
        left_half_w = w
        right_half_w = canvas_w - (w + self.DEBUG_CANVAS_SPACING)
        
        # Calculate available vertical space
        available_h = crop_preview_h - (self.DEBUG_CROP_PREVIEW_TOP_PADDING + self.DEBUG_CROP_PREVIEW_BOTTOM_PADDING)
        
        # Scale to fit within each half with padding, maintaining aspect ratio
        padding = self.DEBUG_PADDING
        max_display_w_left = left_half_w - padding * 2
        max_display_w_right = right_half_w - padding * 2
        max_display_w = min(max_display_w_left, max_display_w_right)
        
        # Calculate scale based on both width and height constraints
        scale_w = max_display_w / crop_w
        scale_h = available_h / crop_h
        scale = min(scale_w, scale_h)  # Use the smaller scale to fit both dimensions
        
        display_crop_w = int(crop_w * scale)
        display_crop_h = int(crop_h * scale)
        
        # Center previews in their respective halves (both horizontally and vertically)
        preview_x1 = padding + (left_half_w - padding * 2 - display_crop_w) // 2
        preview_x2 = w + self.DEBUG_CANVAS_SPACING + padding + (right_half_w - padding * 2 - display_crop_w) // 2
        preview_y = crop_preview_y_start + self.DEBUG_CROP_PREVIEW_TOP_PADDING + (available_h - display_crop_h) // 2
        
        # Final bounds check to ensure everything fits
        preview_x1 = max(0, min(preview_x1, w - display_crop_w))
        preview_x2 = max(w + self.DEBUG_CANVAS_SPACING, min(preview_x2, canvas_w - display_crop_w))
        preview_y = max(crop_preview_y_start + self.DEBUG_CROP_PREVIEW_TOP_PADDING, min(preview_y, crop_preview_y_end - display_crop_h - 10))
        
        # Calculate actual height for both previews
        end_y = min(preview_y + display_crop_h, crop_preview_y_end)
        actual_h = end_y - preview_y
        
        # Resize and display previous cropped image (left) - Target node
        if cropped_prev is not None:
                prev_display = cv2.resize(cropped_prev, (display_crop_w, display_crop_h), interpolation=cv2.INTER_LINEAR)
                # Copy with bounds checking
                end_x1 = min(preview_x1 + display_crop_w, w)
                actual_w1 = end_x1 - preview_x1
                if actual_h > 0 and actual_w1 > 0 and preview_y >= 0 and preview_x1 >= 0:
                    src_h = min(actual_h, prev_display.shape[0])
                    src_w = min(actual_w1, prev_display.shape[1])
                    canvas[preview_y:preview_y+src_h, preview_x1:preview_x1+src_w] = prev_display[:src_h, :src_w]
                
                # Add label for previous (target node)
                label_text = f"Target ({self.PHASE_CORR_CROP_SIZE}x{self.PHASE_CORR_CROP_SIZE})"
                cv2.putText(canvas, label_text, (preview_x1, preview_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
                
                # Draw border around previous preview
                cv2.rectangle(canvas, (preview_x1, preview_y), 
                             (preview_x1+display_crop_w, preview_y+display_crop_h), self.COLOR_GREEN, 2)
        else:
            # Show placeholder text if target not available
            label_text = f"Target (N/A)"
            cv2.putText(canvas, label_text, (preview_x1, preview_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GRAY_LIGHT, 1)
        
        # Resize and display current cropped image (right) - Current real-time position
        curr_display = cv2.resize(cropped_curr, (display_crop_w, display_crop_h), interpolation=cv2.INTER_LINEAR)
        end_x2 = min(preview_x2 + display_crop_w, canvas_w)
        actual_w2 = end_x2 - preview_x2
        if actual_h > 0 and actual_w2 > 0 and preview_y >= 0 and preview_x2 >= 0:
            src_h = min(actual_h, curr_display.shape[0])
            src_w = min(actual_w2, curr_display.shape[1])
            canvas[preview_y:preview_y+src_h, preview_x2:preview_x2+src_w] = curr_display[:src_h, :src_w]
        
        # Add label for current
        label_text = f"Current ({self.PHASE_CORR_CROP_SIZE}x{self.PHASE_CORR_CROP_SIZE})"
        cv2.putText(canvas, label_text, (preview_x2, preview_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        # Draw border around current preview
        cv2.rectangle(canvas, (preview_x2, preview_y), 
                     (preview_x2+display_crop_w, preview_y+display_crop_h), self.COLOR_GREEN, 2)

    def _draw_controller_visualization(self, canvas: np.ndarray, controller_state: Dict, 
                                        ctrl_y_start: int, canvas_w: int, canvas_h: int):
        """
        Draws controller state visualization (sticks and buttons).
        """
        cv2.rectangle(canvas, (0, ctrl_y_start), (canvas_w, canvas_h), self.COLOR_GRAY_DARK, -1)
        
        if controller_state:
            ls_center = (100, ctrl_y_start + 75)
            rs_center = (canvas_w - 100, ctrl_y_start + 75)
            btn_center = (canvas_w // 2, ctrl_y_start + 75)
            radius = 40
            
            # Left Stick (Movement)
            cv2.circle(canvas, ls_center, radius, self.COLOR_GRAY_MEDIUM, 2)
            lx = controller_state.get('lx', 0.0)
            ly = controller_state.get('ly', 0.0)
            ls_dot_x = int(ls_center[0] + lx * radius)
            ls_dot_y = int(ls_center[1] + ly * radius)
            cv2.circle(canvas, (ls_dot_x, ls_dot_y), 8, self.COLOR_CYAN, -1)
            cv2.putText(canvas, "L-Stick (Move)", (ls_center[0]-50, ls_center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_GRAY_LIGHT, 1)
            
            # Right Stick (Camera)
            cv2.circle(canvas, rs_center, radius, self.COLOR_GRAY_MEDIUM, 2)
            rx = controller_state.get('rx', 0.0)
            ry = controller_state.get('ry', 0.0)
            rs_dot_x = int(rs_center[0] + rx * radius)
            rs_dot_y = int(rs_center[1] + ry * radius)
            cv2.circle(canvas, (rs_dot_x, rs_dot_y), 8, self.COLOR_CYAN, -1)
            cv2.putText(canvas, "R-Stick (Cam)", (rs_center[0]-50, rs_center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_GRAY_LIGHT, 1)
            
            btns = controller_state.get('buttons', set())
            b_radius = 12
            off = 35
            pos_y = (btn_center[0], btn_center[1] - off)
            pos_b = (btn_center[0] + off, btn_center[1])
            pos_a = (btn_center[0], btn_center[1] + off)
            pos_x = (btn_center[0] - off, btn_center[1])
            
            cv2.circle(canvas, pos_y, b_radius, self.COLOR_CYAN if 'y' in btns else (50, 50, 0), -1)
            cv2.putText(canvas, "Y", (pos_y[0]-5, pos_y[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
            cv2.circle(canvas, pos_b, b_radius, self.COLOR_RED if 'b' in btns else (50, 0, 0), -1)
            cv2.putText(canvas, "B", (pos_b[0]-5, pos_b[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
            cv2.circle(canvas, pos_a, b_radius, self.COLOR_GREEN if 'a' in btns else (0, 50, 0), -1)
            cv2.putText(canvas, "A", (pos_a[0]-5, pos_a[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)
            cv2.circle(canvas, pos_x, b_radius, self.COLOR_BLUE if 'x' in btns else (50, 0, 0), -1)
            cv2.putText(canvas, "X", (pos_x[0]-5, pos_x[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_BLACK, 1)

    def _draw_histograms_and_charts(self, canvas: np.ndarray, hist_y_start: int, chart_y_start: int, 
                                     canvas_w: int, hist_h: int, chart_h: int):
        """
        Draws histogram and line chart visualizations for dx, dy, and angle.
        """
        # Histogram section background
        cv2.rectangle(canvas, (0, hist_y_start), (canvas_w, hist_y_start + hist_h), (20, 20, 20), -1)
        
        # Calculate histogram dimensions (3 histograms side by side)
        hist_width = (canvas_w - 40) // 3  # Leave some padding
        hist_height = hist_h - self.DEBUG_HIST_BOTTOM_PADDING
        hist_spacing = 10
        
        # Draw dx histogram
        if self.dx_history:
            self._draw_histogram(
                canvas,
                x=10,
                y=hist_y_start + 10,
                width=hist_width,
                height=hist_height,
                values=self.dx_history,
                label="dx",
                color=self.COLOR_GREEN,
                min_val=self.HIST_DX_MIN,
                max_val=self.HIST_DX_MAX
            )
        
        # Draw dy histogram
        if self.dy_history:
            self._draw_histogram(
                canvas,
                x=20 + hist_width,
                y=hist_y_start + 10,
                width=hist_width,
                height=hist_height,
                values=self.dy_history,
                label="dy",
                color=self.COLOR_BLUE,
                min_val=self.HIST_DY_MIN,
                max_val=self.HIST_DY_MAX
            )
        
        # Draw angle histogram
        if self.angle_history:
            self._draw_histogram(
                canvas,
                x=30 + hist_width * 2,
                y=hist_y_start + 10,
                width=hist_width,
                height=hist_height,
                values=self.angle_history,
                label="angle",
                color=self.COLOR_ORANGE,
                min_val=self.HIST_ANGLE_MIN,
                max_val=self.HIST_ANGLE_MAX
            )
        
        # Chart section background
        cv2.rectangle(canvas, (0, chart_y_start), (canvas_w, canvas.shape[0]), (15, 15, 15), -1)
        
        # Draw dx line chart
        if self.dx_history:
            self._draw_line_chart(
                canvas,
                x=10,
                y=chart_y_start + 10,
                width=hist_width,
                height=chart_h - 20,
                values=self.dx_history,
                label="dx (time)",
                color=self.COLOR_GREEN,
                min_val=self.HIST_DX_MIN,
                max_val=self.HIST_DX_MAX,
                zero_line=True
            )
        
        # Draw dy line chart
        if self.dy_history:
            self._draw_line_chart(
                canvas,
                x=20 + hist_width,
                y=chart_y_start + 10,
                width=hist_width,
                height=chart_h - 20,
                values=self.dy_history,
                label="dy (time)",
                color=self.COLOR_BLUE,
                min_val=self.HIST_DY_MIN,
                max_val=self.HIST_DY_MAX,
                zero_line=True
            )
        
        # Draw angle line chart
        if self.angle_history:
            self._draw_line_chart(
                canvas,
                x=30 + hist_width * 2,
                y=chart_y_start + 10,
                width=hist_width,
                height=chart_h - 20,
                values=self.angle_history,
                label="angle (time)",
                color=self.COLOR_ORANGE,
                min_val=self.HIST_ANGLE_MIN,
                max_val=self.HIST_ANGLE_MAX,
                zero_line=True
            )

    def show_debug_view(self, current_img: np.ndarray, target_minimap: np.ndarray, controller_state: Dict = None, tracking_active: bool = True, status_msg: str = ""):
        """
        Displays a debug window showing the visual odometry process and controller input.
        Shows the color-filtered minimaps matching the HSV debug "Combined" view.
        
        NOTE: The preview uses _filter_colors_only (no edge detection) to match the HSV debug panel.
        This shows only the color-filtered features without edge detection grid lines.
        ORB feature matching still uses _filter_blue_colors (with edges) internally.
        """
        current_minimap = self.extract_minimap(current_img)
        if current_minimap is None: 
            return

        # Prepare canvas
        debug_canvas, canvas_h, canvas_w, h, w = self._prepare_debug_canvas(current_minimap)
        
        # Draw minimap comparison (target vs current)
        self._draw_minimap_comparison(debug_canvas, target_minimap, current_minimap, h, w, tracking_active, status_msg)
        
        # Draw cropped preview section
        crop_preview_h = self.DEBUG_CROP_PREVIEW_HEIGHT
        self._draw_cropped_preview(debug_canvas, current_minimap, target_minimap, h, w, canvas_w, crop_preview_h)
        
        # Draw controller visualization
        ctrl_y_start = h + crop_preview_h
        hist_h = self.DEBUG_HIST_HEIGHT
        chart_h = self.DEBUG_CHART_HEIGHT
        self._draw_controller_visualization(debug_canvas, controller_state, ctrl_y_start, canvas_w, canvas_h)
        
        # Draw histograms and charts
        hist_y_start = canvas_h - chart_h - hist_h
        chart_y_start = canvas_h - chart_h
        self._draw_histograms_and_charts(debug_canvas, hist_y_start, chart_y_start, canvas_w, hist_h, chart_h)

        cv2.imshow(self.debug_window_name, debug_canvas)
        if not hasattr(self, '_debug_win_pos_set'):
            win_x, win_y = self.vision.window_offset if hasattr(self.vision, 'window_offset') else (0, 0)
            win_w, win_h = self.vision.resolution if hasattr(self.vision, 'resolution') else RESOLUTION
            cv2.namedWindow(self.debug_window_name)
            cv2.moveWindow(self.debug_window_name, win_x + win_w + 10, win_y + 100)
            self._debug_win_pos_set = True
        cv2.waitKey(1)

    def _on_trackbar_blue_lower_h(self, val):
        """Callback for blue lower H trackbar."""
        self.blue_lower[0] = val

    def _on_trackbar_blue_lower_s(self, val):
        """Callback for blue lower S trackbar."""
        self.blue_lower[1] = val

    def _on_trackbar_blue_lower_v(self, val):
        """Callback for blue lower V trackbar."""
        self.blue_lower[2] = val

    def _on_trackbar_blue_upper_h(self, val):
        """Callback for blue upper H trackbar."""
        self.blue_upper[0] = val

    def _on_trackbar_blue_upper_s(self, val):
        """Callback for blue upper S trackbar."""
        self.blue_upper[1] = val

    def _on_trackbar_blue_upper_v(self, val):
        """Callback for blue upper V trackbar."""
        self.blue_upper[2] = val

    def _on_trackbar_alert_lower_h(self, val):
        """Callback for alert lower H trackbar."""
        self.alert_lower[0] = val

    def _on_trackbar_alert_lower_s(self, val):
        """Callback for alert lower S trackbar."""
        self.alert_lower[1] = val

    def _on_trackbar_alert_lower_v(self, val):
        """Callback for alert lower V trackbar."""
        self.alert_lower[2] = val

    def _on_trackbar_alert_upper_h(self, val):
        """Callback for alert upper H trackbar."""
        self.alert_upper[0] = val

    def _on_trackbar_alert_upper_s(self, val):
        """Callback for alert upper S trackbar."""
        self.alert_upper[1] = val

    def _on_trackbar_alert_upper_v(self, val):
        """Callback for alert upper V trackbar."""
        self.alert_upper[2] = val

    def _on_trackbar_red_lower_h(self, val):
        """Callback for red lower H trackbar."""
        self.red_lower[0] = val

    def _on_trackbar_red_lower_s(self, val):
        """Callback for red lower S trackbar."""
        self.red_lower[1] = val

    def _on_trackbar_red_lower_v(self, val):
        """Callback for red lower V trackbar."""
        self.red_lower[2] = val

    def _on_trackbar_red_upper_h(self, val):
        """Callback for red upper H trackbar."""
        self.red_upper[0] = val

    def _on_trackbar_red_upper_s(self, val):
        """Callback for red upper S trackbar."""
        self.red_upper[1] = val

    def _on_trackbar_red_upper_v(self, val):
        """Callback for red upper V trackbar."""
        self.red_upper[2] = val

    def _on_trackbar_gold_lower_h(self, val):
        """Callback for gold lower H trackbar."""
        self.gold_lower[0] = val

    def _on_trackbar_gold_lower_s(self, val):
        """Callback for gold lower S trackbar."""
        self.gold_lower[1] = val

    def _on_trackbar_gold_lower_v(self, val):
        """Callback for gold lower V trackbar."""
        self.gold_lower[2] = val

    def _on_trackbar_gold_upper_h(self, val):
        """Callback for gold upper H trackbar."""
        self.gold_upper[0] = val

    def _on_trackbar_gold_upper_s(self, val):
        """Callback for gold upper S trackbar."""
        self.gold_upper[1] = val

    def _on_trackbar_gold_upper_v(self, val):
        """Callback for gold upper V trackbar."""
        self.gold_upper[2] = val

    def enable_hsv_debug(self):
        """
        Enables HSV filter debug mode with trackbars for real-time adjustment.
        Creates a debug window with trackbars for all 3 filters (blue, alert/red, gold).
        """
        if self.hsv_debug_enabled:
            return
        
        self.hsv_debug_enabled = True
        cv2.namedWindow(self.hsv_debug_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.hsv_debug_window_name, 800, 1000)
        
        # Position window
        if hasattr(self.vision, 'window_offset') and hasattr(self.vision, 'resolution'):
            win_x, win_y = self.vision.window_offset
            win_w, win_h = self.vision.resolution
            cv2.moveWindow(self.hsv_debug_window_name, win_x + win_w + 20, win_y + 50)
        
        # Create trackbars for Blue filter
        cv2.createTrackbar('Blue L H', self.hsv_debug_window_name, self.blue_lower[0], 179, self._on_trackbar_blue_lower_h)
        cv2.createTrackbar('Blue L S', self.hsv_debug_window_name, self.blue_lower[1], 255, self._on_trackbar_blue_lower_s)
        cv2.createTrackbar('Blue L V', self.hsv_debug_window_name, self.blue_lower[2], 255, self._on_trackbar_blue_lower_v)
        cv2.createTrackbar('Blue U H', self.hsv_debug_window_name, self.blue_upper[0], 179, self._on_trackbar_blue_upper_h)
        cv2.createTrackbar('Blue U S', self.hsv_debug_window_name, self.blue_upper[1], 255, self._on_trackbar_blue_upper_s)
        cv2.createTrackbar('Blue U V', self.hsv_debug_window_name, self.blue_upper[2], 255, self._on_trackbar_blue_upper_v)
        
        # Create trackbars for Alert/Red filter (first range)
        cv2.createTrackbar('Alert L H', self.hsv_debug_window_name, self.alert_lower[0], 179, self._on_trackbar_alert_lower_h)
        cv2.createTrackbar('Alert L S', self.hsv_debug_window_name, self.alert_lower[1], 255, self._on_trackbar_alert_lower_s)
        cv2.createTrackbar('Alert L V', self.hsv_debug_window_name, self.alert_lower[2], 255, self._on_trackbar_alert_lower_v)
        cv2.createTrackbar('Alert U H', self.hsv_debug_window_name, self.alert_upper[0], 179, self._on_trackbar_alert_upper_h)
        cv2.createTrackbar('Alert U S', self.hsv_debug_window_name, self.alert_upper[1], 255, self._on_trackbar_alert_upper_s)
        cv2.createTrackbar('Alert U V', self.hsv_debug_window_name, self.alert_upper[2], 255, self._on_trackbar_alert_upper_v)
        
        # Create trackbars for Red filter (second range)
        cv2.createTrackbar('Red L H', self.hsv_debug_window_name, self.red_lower[0], 179, self._on_trackbar_red_lower_h)
        cv2.createTrackbar('Red L S', self.hsv_debug_window_name, self.red_lower[1], 255, self._on_trackbar_red_lower_s)
        cv2.createTrackbar('Red L V', self.hsv_debug_window_name, self.red_lower[2], 255, self._on_trackbar_red_lower_v)
        cv2.createTrackbar('Red U H', self.hsv_debug_window_name, self.red_upper[0], 179, self._on_trackbar_red_upper_h)
        cv2.createTrackbar('Red U S', self.hsv_debug_window_name, self.red_upper[1], 255, self._on_trackbar_red_upper_s)
        cv2.createTrackbar('Red U V', self.hsv_debug_window_name, self.red_upper[2], 255, self._on_trackbar_red_upper_v)
        
        # Create trackbars for Gold filter
        cv2.createTrackbar('Gold L H', self.hsv_debug_window_name, self.gold_lower[0], 179, self._on_trackbar_gold_lower_h)
        cv2.createTrackbar('Gold L S', self.hsv_debug_window_name, self.gold_lower[1], 255, self._on_trackbar_gold_lower_s)
        cv2.createTrackbar('Gold L V', self.hsv_debug_window_name, self.gold_lower[2], 255, self._on_trackbar_gold_lower_v)
        cv2.createTrackbar('Gold U H', self.hsv_debug_window_name, self.gold_upper[0], 179, self._on_trackbar_gold_upper_h)
        cv2.createTrackbar('Gold U S', self.hsv_debug_window_name, self.gold_upper[1], 255, self._on_trackbar_gold_upper_s)
        cv2.createTrackbar('Gold U V', self.hsv_debug_window_name, self.gold_upper[2], 255, self._on_trackbar_gold_upper_v)
        
        print("[HSV DEBUG] Debug mode enabled. Adjust trackbars to fine-tune HSV filters.")
        print("[HSV DEBUG] Press 'q' in the debug window to disable debug mode.")

    def disable_hsv_debug(self):
        """
        Disables HSV filter debug mode and closes the debug window.
        """
        if not self.hsv_debug_enabled:
            return
        
        self.hsv_debug_enabled = False
        try:
            cv2.destroyWindow(self.hsv_debug_window_name)
        except cv2.error:
            # Window may not exist if it was never created or already destroyed
            pass
        print("[HSV DEBUG] Debug mode disabled.")

    def show_hsv_debug(self, image: np.ndarray):
        """
        Shows the HSV filter debug window with real-time filtered results.
        Displays original minimap, filtered result, and individual filter masks.
        
        Args:
            image: Full screen image to extract minimap from
        """
        if not self.hsv_debug_enabled:
            return
        
        # Extract minimap
        minimap = self.extract_minimap(image)
        if minimap is None:
            return
        
        # Create HSV masks for individual filters
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        # Blue mask
        mask_blue = cv2.inRange(hsv, np.array(self.blue_lower), np.array(self.blue_upper))
        blue_only = cv2.bitwise_and(minimap, minimap, mask=mask_blue)
        
        # Alert mask
        mask_alert = cv2.inRange(hsv, np.array(self.alert_lower), np.array(self.alert_upper))
        alert_only = cv2.bitwise_and(minimap, minimap, mask=mask_alert)
        
        # Red mask
        mask_red = cv2.inRange(hsv, np.array(self.red_lower), np.array(self.red_upper))
        red_only = cv2.bitwise_and(minimap, minimap, mask=mask_red)
        
        # Gold mask
        mask_gold = cv2.inRange(hsv, np.array(self.gold_lower), np.array(self.gold_upper))
        gold_only = cv2.bitwise_and(minimap, minimap, mask=mask_gold)
        
        # Combine all color filters (without edge detection) for the "Filtered (Combined)" view
        mask_colors = cv2.bitwise_or(mask_blue, mask_alert)
        mask_colors = cv2.bitwise_or(mask_colors, mask_red)
        mask_colors = cv2.bitwise_or(mask_colors, mask_gold)
        filtered = cv2.bitwise_and(minimap, minimap, mask=mask_colors)
        
        # Create debug canvas
        h, w = minimap.shape[:2]
        canvas_h = h * 3
        canvas_w = w * 2 + 20
        
        debug_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Row 1: Original and Filtered
        debug_canvas[0:h, 0:w] = minimap
        debug_canvas[0:h, w+20:w*2+20] = filtered
        
        # Row 2: Blue and Alert filters
        debug_canvas[h:h*2, 0:w] = blue_only
        debug_canvas[h:h*2, w+20:w*2+20] = alert_only
        
        # Row 3: Red and Gold filters
        debug_canvas[h*2:h*3, 0:w] = red_only
        debug_canvas[h*2:h*3, w+20:w*2+20] = gold_only
        
        # Add labels
        cv2.putText(debug_canvas, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_canvas, "Filtered (Combined)", (w+30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_canvas, "Blue Filter", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(debug_canvas, "Alert Filter", (w+30, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_canvas, "Red Filter", (10, h*2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(debug_canvas, "Gold Filter", (w+30, h*2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Display HSV values
        info_y = h*3 - 10
        info_text = f"Blue: [{self.blue_lower}] - [{self.blue_upper}]"
        cv2.putText(debug_canvas, info_text, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow(self.hsv_debug_window_name, debug_canvas)
        
        # Check for 'q' key to disable debug mode
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.disable_hsv_debug()
