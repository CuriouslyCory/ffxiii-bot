import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple, Dict

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
        self.MINIMAP_RADIUS = 190
        self.CIRCULAR_CROP_Y1 = 60  # For extracting 400x400 from 400x520
        self.CIRCULAR_CROP_Y2 = 460
        
        # --- HSV Filter Ranges (for map feature detection) ---
        # Blue filter (normal minimap state)
        self.blue_lower = [84, 75, 100]
        self.blue_upper = [97, 245, 245]
        
        # Alert/Red filter (enemy nearby state - two ranges for HSV wheel wrap)
        self.alert_lower = [0, 40, 50]
        self.alert_upper = [30, 255, 255]
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
        self.CIRCLE_MIN_RADIUS = 3  # Reduced to catch smaller enemy dots
        self.CIRCLE_MAX_RADIUS = 13
        # Use same or broader ranges as main alert/red filter to catch all red dots
        self.CIRCLE_DETECTION_RED_LOWER1 = [0, 40, 50]  # Match alert_lower
        self.CIRCLE_DETECTION_RED_UPPER1 = [30, 255, 255]  # Match alert_upper
        self.CIRCLE_DETECTION_RED_LOWER2 = [170, 40, 50]  # Match red_lower
        self.CIRCLE_DETECTION_RED_UPPER2 = [180, 255, 255]  # Match red_upper
        self.CIRCLE_DETECTION_BLUE_LOWER = [100, 100, 100]
        self.CIRCLE_DETECTION_BLUE_UPPER = [130, 255, 255]
        self.CIRCLE_DETECTION_WHITE_LOWER = [0, 0, 200]
        self.CIRCLE_DETECTION_WHITE_UPPER = [180, 30, 255]
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
        self.CENTER_MASK_RADIUS_MASTER_MAP = 40
        self.ARROW_CENTER_MASK_RADIUS = 80
        
        # --- Phase Correlation ---
        self.PHASE_CORR_CONFIDENCE_THRESHOLD = 0.05
        
        # --- Feature Matching Thresholds ---
        self.MIN_INLIERS_FOR_TRACKING = 8
        self.MIN_INLIERS_FOR_RECOVERY = 15
        self.ARRIVAL_DISTANCE_THRESHOLD = 20
        self.ARRIVAL_ANGLE_THRESHOLD = 15
        self.ARRIVAL_BUFFER_SIZE = 5
        
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
        # Blue range
        lower_blue = np.array(self.CIRCLE_DETECTION_BLUE_LOWER)
        upper_blue = np.array(self.CIRCLE_DETECTION_BLUE_UPPER)
        mask_blue_circles = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also detect white centers (high value, low saturation)
        lower_white = np.array(self.CIRCLE_DETECTION_WHITE_LOWER)
        upper_white = np.array(self.CIRCLE_DETECTION_WHITE_UPPER)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine red and blue/white masks
        mask_circles = cv2.bitwise_or(mask_red, mask_blue_circles)
        mask_circles = cv2.bitwise_or(mask_circles, mask_white)
        
        # Use HoughCircles to detect circular shapes
        # Apply to the combined mask
        circles = cv2.HoughCircles(
            mask_circles,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_radius * 2,
            param1=50,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw filled circle on mask to exclude this area
                cv2.circle(mask, (x, y), r + self.CIRCLE_MASK_PADDING, 0, -1)  # Padding for mask out
        
        # Alternative: Use contour detection for more robust circle detection
        # This helps catch circles that HoughCircles might miss
        contours, _ = cv2.findContours(mask_circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area (roughly circular area for our size range)
            min_area = np.pi * min_radius * min_radius
            max_area = np.pi * max_radius * max_radius
            
            if min_area <= area <= max_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Circularity of 1.0 is a perfect circle, we accept > 0.7
                    if circularity > 0.7:
                        # Get bounding circle
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if min_radius <= radius <= max_radius:
                            # Mask out this circle
                            cv2.circle(mask, (int(x), int(y)), int(radius) + self.CIRCLE_MASK_PADDING, 0, -1)
        
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
            
        angle_rad = np.arctan2(vY, vX)
        return np.degrees(angle_rad) + 90, centroid

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
        
        if conf < self.PHASE_CORR_CONFIDENCE_THRESHOLD:
            return 0.0, 0.0, conf
            
        # Translation is already in world-space (stretched coordinates)
        return dx, dy, conf

    def compute_drift(self, current_img: np.ndarray, target_minimap: np.ndarray) -> Optional[Tuple[float, float, float, int, np.ndarray]]:
        """
        Legacy Feature-Matching drift (ORB). Used for navigation lookahead.
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

    def generate_master_map(self, nodes: list, landmark_dir: str) -> Optional[np.ndarray]:
        """
        Generates a composite master map from a list of hybrid nodes using North-Aligned SLAM.
        Stitches minimap images together using recorded world-space translations.
        """
        if not nodes:
            return None

        # 1. Calculate Cumulative World Coordinates (North-Up Frame)
        world_coords = [(0.0, 0.0)]
        curr_x, curr_y = 0.0, 0.0
        
        for i in range(1, len(nodes)):
            offset = nodes[i].get('relative_offset')
            if offset:
                dx, dy = offset['dx'], offset['dy']
                curr_x += dx
                curr_y += dy
            world_coords.append((curr_x, curr_y))

        # 2. Determine Canvas Size
        crop_size = self.MASTER_MAP_CROP_SIZE
        
        crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        cv2.circle(crop_mask, (crop_size//2, crop_size//2), crop_size//2 - self.MASTER_MAP_CIRCLE_BORDER, (255), -1)

        pts = np.array(world_coords)
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
        
        # 3. Stitch North-Aligned Nodes
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
            
            # Step 2: Crop 400x400 circle from stretched image (including border)
            # Extract region centered at (200, 260): x=0-400, y=60-460
            crop_y1, crop_y2 = self.CIRCULAR_CROP_Y1, self.CIRCULAR_CROP_Y2
            img_circle = img[crop_y1:crop_y2, 0:self.MINIMAP_STRETCHED_WIDTH]
            
            # Apply circular mask to the 400x400 region
            circle_mask = np.zeros((self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE), dtype=np.uint8)
            cv2.circle(circle_mask, (self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), self.MINIMAP_RADIUS, (255), -1)
            img_circle = cv2.bitwise_and(img_circle, img_circle, mask=circle_mask)
            
            # Step 3: Rotate the 400x400 circle to North
            arrow_angle = node.get('arrow_angle', 0.0)
            pivot = node.get('arrow_pivot', self.minimap_center_stretched)
            # Pivot in 400x400 crop space: (200, 200) - the center of the circle
            R = cv2.getRotationMatrix2D((self.MINIMAP_CENTER_X, self.MINIMAP_CENTER_Y), arrow_angle, 1.0)
            img_north = cv2.warpAffine(img_circle, R, (self.CIRCULAR_CROP_SIZE, self.CIRCULAR_CROP_SIZE))
            
            # Step 4: Filter blue outlines
            blue_only = self._filter_blue_colors(img_north, include_gold=False)
            
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
            
            # Composite
            if 0 <= tx < canvas_w - crop_size and 0 <= ty < canvas_h - crop_size:
                region = master_canvas[ty:ty+crop_size, tx:tx+crop_size]
                master_canvas[ty:ty+crop_size, tx:tx+crop_size] = cv2.max(region, final_fragment)
            
            centers.append((int(wx - min_x), int(wy - min_y)))

        # 4. Draw Breadcrumbs
        if len(centers) > 1:
            for i in range(len(centers) - 1):
                cv2.line(master_canvas, centers[i], centers[i+1], self.COLOR_ORANGE, 2)
            cv2.circle(master_canvas, centers[0], 8, self.COLOR_GREEN, -1)  # Start
            cv2.circle(master_canvas, centers[-1], 8, self.COLOR_RED, -1)  # End
        
        # 5. Downsample
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
        if cropped_curr is not None:
            crop_h, crop_w = cropped_prev.shape[:2]  # Should be PHASE_CORR_CROP_SIZE x PHASE_CORR_CROP_SIZE
            
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
            win_w, win_h = self.vision.resolution if hasattr(self.vision, 'resolution') else (1920, 1080)
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
