#!/usr/bin/env python3
"""
Debug utility for testing sensors.

Allows testing individual sensors with specific images or live screen capture.
Shows sensor input (ROI) and results/telemetry data.

Enhanced with filter parameter controls and debug output display.
"""
import cv2
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import numpy as np

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vision import VisionEngine
from src.core.roi_cache import ROICache
from src.sensors.base import Sensor
from src.sensors.health import HealthSensor
from src.sensors.minimap import MinimapSensor
from src.sensors.minimap_state import MinimapStateSensor
from src.sensors.compass import CompassSensor
from src.sensors.enemy_position import EnemyPositionSensor
from src.sensors.player_direction import PlayerDirectionSensor
from src.states.movement.constants import WINDOW_OFFSET, RESOLUTION
from src.filters.base import Filter
from src.filters.color import HSVFilter, BlueFilter, AlertFilter
from src.filters.mask import EllipseMaskFilter, RectangleMaskFilter, CircleMaskFilter, ShapeFilter
from src.filters.composite import CompositeFilter


class FilterParameterController:
    """
    Manages filter parameter controls (trackbars, etc.) for debugging.
    """
    
    def __init__(self):
        """Initialize filter parameter controller."""
        self.base_window_name = "Filter Parameters"
        self.window_name = self.base_window_name
        self.trackbar_values: Dict[str, Dict[str, int]] = {}
        self.filter_references: Dict[str, Filter] = {}
        self._trackbar_created = False
        self._current_sensor_name: Optional[str] = None
    
    def setup_filter_controls(self, filters: Dict[str, Filter], roi_image: Optional[np.ndarray], sensor_name: Optional[str] = None):
        """
        Set up filter parameter controls for registered filters.
        
        Args:
            filters: Dictionary of filter names to Filter instances
            roi_image: ROI image for determining mask parameter bounds (optional)
            sensor_name: Name of the sensor (for unique window naming)
        """
        # Clear existing trackbars and window
        if self._trackbar_created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass  # Window might not exist
            self._trackbar_created = False
        
        # Use unique window name per sensor to avoid trackbar conflicts
        if sensor_name:
            self.window_name = f"{self.base_window_name} - {sensor_name}"
            self._current_sensor_name = sensor_name
        else:
            self.window_name = self.base_window_name
            self._current_sensor_name = None
        
        self.filter_references = filters
        self.trackbar_values = {}
        
        if not filters:
            return
        
        # Create window for trackbars with unique name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 600)
        
        y_pos = 10
        for filter_name, filter_obj in filters.items():
            y_pos = self._create_filter_controls(filter_name, filter_obj, y_pos, roi_image)
            y_pos += 20  # Spacing between filters
        
        self._trackbar_created = True
    
    def _create_filter_controls(self, filter_name: str, filter_obj: Filter, y_pos: int, roi_image: Optional[np.ndarray]) -> int:
        """
        Create controls for a specific filter.
        
        Returns:
            Next y position for controls
        """
        # Handle CompositeFilter - extract individual filters
        if isinstance(filter_obj, CompositeFilter):
            # For composite filters, we'll handle the individual filters
            # For now, skip composite filters in parameter controls
            # (they'll be handled by their component filters)
            return y_pos
        
        # Handle HSV filters (HSVFilter, BlueFilter, AlertFilter)
        if isinstance(filter_obj, (HSVFilter, BlueFilter)):
            return self._create_hsv_controls(filter_name, filter_obj, y_pos)
        elif isinstance(filter_obj, AlertFilter):
            return self._create_alert_controls(filter_name, filter_obj, y_pos)
        
        # Handle mask filters
        if isinstance(filter_obj, EllipseMaskFilter):
            return self._create_ellipse_controls(filter_name, filter_obj, y_pos, roi_image)
        elif isinstance(filter_obj, RectangleMaskFilter):
            return self._create_rectangle_controls(filter_name, filter_obj, y_pos, roi_image)
        elif isinstance(filter_obj, CircleMaskFilter):
            return self._create_circle_controls(filter_name, filter_obj, y_pos, roi_image)
        
        return y_pos
    
    def _create_hsv_controls(self, filter_name: str, filter_obj: Filter, y_pos: int) -> int:
        """Create HSV trackbar controls."""
        prefix = f"{filter_name}_"
        self.trackbar_values[filter_name] = {}
        
        # Get current values
        lower = filter_obj.lower
        upper = filter_obj.upper
        
        # Create trackbars for lower bounds (H, S, V)
        cv2.createTrackbar(f"{prefix}H_lower", self.window_name, int(lower[0]), 179, lambda x: None)
        cv2.createTrackbar(f"{prefix}S_lower", self.window_name, int(lower[1]), 255, lambda x: None)
        cv2.createTrackbar(f"{prefix}V_lower", self.window_name, int(lower[2]), 255, lambda x: None)
        
        # Create trackbars for upper bounds (H, S, V)
        cv2.createTrackbar(f"{prefix}H_upper", self.window_name, int(upper[0]), 179, lambda x: None)
        cv2.createTrackbar(f"{prefix}S_upper", self.window_name, int(upper[1]), 255, lambda x: None)
        cv2.createTrackbar(f"{prefix}V_upper", self.window_name, int(upper[2]), 255, lambda x: None)
        
        # Store initial values
        self.trackbar_values[filter_name] = {
            "H_lower": int(lower[0]),
            "S_lower": int(lower[1]),
            "V_lower": int(lower[2]),
            "H_upper": int(upper[0]),
            "S_upper": int(upper[1]),
            "V_upper": int(upper[2]),
        }
        
        return y_pos + 200  # Approximate height for 6 trackbars
    
    def _create_alert_controls(self, filter_name: str, filter_obj: AlertFilter, y_pos: int) -> int:
        """Create AlertFilter trackbar controls (two HSV ranges)."""
        prefix = f"{filter_name}_"
        
        # Range 1
        cv2.createTrackbar(f"{prefix}R1_H_lower", self.window_name, int(filter_obj.lower1[0]), 179, lambda x: None)
        cv2.createTrackbar(f"{prefix}R1_S_lower", self.window_name, int(filter_obj.lower1[1]), 255, lambda x: None)
        cv2.createTrackbar(f"{prefix}R1_V_lower", self.window_name, int(filter_obj.lower1[2]), 255, lambda x: None)
        cv2.createTrackbar(f"{prefix}R1_H_upper", self.window_name, int(filter_obj.upper1[0]), 179, lambda x: None)
        cv2.createTrackbar(f"{prefix}R1_S_upper", self.window_name, int(filter_obj.upper1[1]), 255, lambda x: None)
        cv2.createTrackbar(f"{prefix}R1_V_upper", self.window_name, int(filter_obj.upper1[2]), 255, lambda x: None)
        
        # Initialize trackbar values
        self.trackbar_values[filter_name] = {
            "R1_H_lower": int(filter_obj.lower1[0]), "R1_S_lower": int(filter_obj.lower1[1]), "R1_V_lower": int(filter_obj.lower1[2]),
            "R1_H_upper": int(filter_obj.upper1[0]), "R1_S_upper": int(filter_obj.upper1[1]), "R1_V_upper": int(filter_obj.upper1[2]),
        }
        
        return y_pos + 400  # Approximate height for 12 trackbars
    
    def _create_ellipse_controls(self, filter_name: str, filter_obj: EllipseMaskFilter, y_pos: int, roi_image: Optional[np.ndarray]) -> int:
        """Create ellipse mask controls."""
        prefix = f"{filter_name}_"
        max_dim = 500
        
        if roi_image is not None:
            h, w = roi_image.shape[:2]
            max_dim = max(w, h)
        
        cv2.createTrackbar(f"{prefix}center_x", self.window_name, filter_obj.center[0], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}center_y", self.window_name, filter_obj.center[1], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}radius_x", self.window_name, filter_obj.axes[0], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}radius_y", self.window_name, filter_obj.axes[1], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}mask_inside", self.window_name, 1 if filter_obj.mask_inside else 0, 1, lambda x: None)
        
        # Initialize trackbar values
        self.trackbar_values[filter_name] = {
            "center_x": filter_obj.center[0],
            "center_y": filter_obj.center[1],
            "radius_x": filter_obj.axes[0],
            "radius_y": filter_obj.axes[1],
            "mask_inside": 1 if filter_obj.mask_inside else 0,
        }
        
        return y_pos + 150
    
    def _create_rectangle_controls(self, filter_name: str, filter_obj: RectangleMaskFilter, y_pos: int, roi_image: Optional[np.ndarray]) -> int:
        """Create rectangle mask controls."""
        prefix = f"{filter_name}_"
        max_dim = 500
        
        if roi_image is not None:
            h, w = roi_image.shape[:2]
            max_dim = max(w, h)
        
        width = filter_obj.bottom_right[0] - filter_obj.top_left[0]
        height = filter_obj.bottom_right[1] - filter_obj.top_left[1]
        cv2.createTrackbar(f"{prefix}x", self.window_name, filter_obj.top_left[0], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}y", self.window_name, filter_obj.top_left[1], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}width", self.window_name, width, max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}height", self.window_name, height, max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}mask_inside", self.window_name, 1 if filter_obj.mask_inside else 0, 1, lambda x: None)
        
        # Initialize trackbar values
        self.trackbar_values[filter_name] = {
            "x": filter_obj.top_left[0],
            "y": filter_obj.top_left[1],
            "width": width,
            "height": height,
            "mask_inside": 1 if filter_obj.mask_inside else 0,
        }
        
        return y_pos + 150
    
    def _create_circle_controls(self, filter_name: str, filter_obj: CircleMaskFilter, y_pos: int, roi_image: Optional[np.ndarray]) -> int:
        """Create circle mask controls."""
        prefix = f"{filter_name}_"
        max_dim = 500
        
        if roi_image is not None:
            h, w = roi_image.shape[:2]
            max_dim = max(w, h)
        
        cv2.createTrackbar(f"{prefix}center_x", self.window_name, filter_obj.center[0], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}center_y", self.window_name, filter_obj.center[1], max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}radius", self.window_name, filter_obj.radius, max_dim, lambda x: None)
        cv2.createTrackbar(f"{prefix}mask_inside", self.window_name, 1 if filter_obj.mask_inside else 0, 1, lambda x: None)
        
        # Initialize trackbar values
        self.trackbar_values[filter_name] = {
            "center_x": filter_obj.center[0],
            "center_y": filter_obj.center[1],
            "radius": filter_obj.radius,
            "mask_inside": 1 if filter_obj.mask_inside else 0,
        }
        
        return y_pos + 120
    
    def update_filter_parameters(self) -> bool:
        """
        Update filter parameters from trackbar values.
        
        Returns:
            True if any parameters changed, False otherwise
        """
        if not self._trackbar_created:
            return False
        
        changed = False
        
        for filter_name, filter_obj in self.filter_references.items():
            if filter_name not in self.trackbar_values:
                continue
            
            # Skip CompositeFilter
            if isinstance(filter_obj, CompositeFilter):
                continue
            
            values = self.trackbar_values[filter_name]
            prefix = f"{filter_name}_"
            
            # Update HSV filters
            if isinstance(filter_obj, (HSVFilter, BlueFilter)):
                h_lower = cv2.getTrackbarPos(f"{prefix}H_lower", self.window_name)
                s_lower = cv2.getTrackbarPos(f"{prefix}S_lower", self.window_name)
                v_lower = cv2.getTrackbarPos(f"{prefix}V_lower", self.window_name)
                h_upper = cv2.getTrackbarPos(f"{prefix}H_upper", self.window_name)
                s_upper = cv2.getTrackbarPos(f"{prefix}S_upper", self.window_name)
                v_upper = cv2.getTrackbarPos(f"{prefix}V_upper", self.window_name)
                
                if (values.get("H_lower") != h_lower or values.get("S_lower") != s_lower or
                    values.get("V_lower") != v_lower or values.get("H_upper") != h_upper or
                    values.get("S_upper") != s_upper or values.get("V_upper") != v_upper):
                    filter_obj.lower = np.array([h_lower, s_lower, v_lower])
                    filter_obj.upper = np.array([h_upper, s_upper, v_upper])
                    values.update({"H_lower": h_lower, "S_lower": s_lower, "V_lower": v_lower,
                                 "H_upper": h_upper, "S_upper": s_upper, "V_upper": v_upper})
                    changed = True
            
            # Update AlertFilter
            elif isinstance(filter_obj, AlertFilter):
                r1_h_lower = cv2.getTrackbarPos(f"{prefix}R1_H_lower", self.window_name)
                r1_s_lower = cv2.getTrackbarPos(f"{prefix}R1_S_lower", self.window_name)
                r1_v_lower = cv2.getTrackbarPos(f"{prefix}R1_V_lower", self.window_name)
                r1_h_upper = cv2.getTrackbarPos(f"{prefix}R1_H_upper", self.window_name)
                r1_s_upper = cv2.getTrackbarPos(f"{prefix}R1_S_upper", self.window_name)
                r1_v_upper = cv2.getTrackbarPos(f"{prefix}R1_V_upper", self.window_name)
                
                # Check if values changed
                if (values.get("R1_H_lower") != r1_h_lower or values.get("R1_S_lower") != r1_s_lower or
                    values.get("R1_V_lower") != r1_v_lower or values.get("R1_H_upper") != r1_h_upper or
                    values.get("R1_S_upper") != r1_s_upper or values.get("R1_V_upper") != r1_v_upper):
                    filter_obj.lower1 = np.array([r1_h_lower, r1_s_lower, r1_v_lower])
                    filter_obj.upper1 = np.array([r1_h_upper, r1_s_upper, r1_v_upper])
                    values.update({
                        "R1_H_lower": r1_h_lower, "R1_S_lower": r1_s_lower, "R1_V_lower": r1_v_lower,
                        "R1_H_upper": r1_h_upper, "R1_S_upper": r1_s_upper, "R1_V_upper": r1_v_upper,
                    })
                    changed = True
            
            # Update EllipseMaskFilter
            elif isinstance(filter_obj, EllipseMaskFilter):
                center_x = cv2.getTrackbarPos(f"{prefix}center_x", self.window_name)
                center_y = cv2.getTrackbarPos(f"{prefix}center_y", self.window_name)
                radius_x = cv2.getTrackbarPos(f"{prefix}radius_x", self.window_name)
                radius_y = cv2.getTrackbarPos(f"{prefix}radius_y", self.window_name)
                mask_inside_val = cv2.getTrackbarPos(f"{prefix}mask_inside", self.window_name)
                mask_inside = mask_inside_val > 0
                
                # Check if values changed
                if (values.get("center_x") != center_x or values.get("center_y") != center_y or
                    values.get("radius_x") != radius_x or values.get("radius_y") != radius_y or
                    values.get("mask_inside") != mask_inside_val):
                    filter_obj.center = (center_x, center_y)
                    filter_obj.axes = (radius_x, radius_y)
                    filter_obj.mask_inside = mask_inside
                    values.update({
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius_x": radius_x,
                        "radius_y": radius_y,
                        "mask_inside": mask_inside_val,
                    })
                    changed = True
            
            # Update RectangleMaskFilter
            elif isinstance(filter_obj, RectangleMaskFilter):
                x = cv2.getTrackbarPos(f"{prefix}x", self.window_name)
                y = cv2.getTrackbarPos(f"{prefix}y", self.window_name)
                width = cv2.getTrackbarPos(f"{prefix}width", self.window_name)
                height = cv2.getTrackbarPos(f"{prefix}height", self.window_name)
                mask_inside_val = cv2.getTrackbarPos(f"{prefix}mask_inside", self.window_name)
                mask_inside = mask_inside_val > 0
                
                # Check if values changed
                if (values.get("x") != x or values.get("y") != y or
                    values.get("width") != width or values.get("height") != height or
                    values.get("mask_inside") != mask_inside_val):
                    filter_obj.top_left = (x, y)
                    filter_obj.bottom_right = (x + width, y + height)
                    filter_obj.mask_inside = mask_inside
                    values.update({
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "mask_inside": mask_inside_val,
                    })
                    changed = True
            
            # Update CircleMaskFilter
            elif isinstance(filter_obj, CircleMaskFilter):
                center_x = cv2.getTrackbarPos(f"{prefix}center_x", self.window_name)
                center_y = cv2.getTrackbarPos(f"{prefix}center_y", self.window_name)
                radius = cv2.getTrackbarPos(f"{prefix}radius", self.window_name)
                mask_inside_val = cv2.getTrackbarPos(f"{prefix}mask_inside", self.window_name)
                mask_inside = mask_inside_val > 0
                
                # Check if values changed
                if (values.get("center_x") != center_x or values.get("center_y") != center_y or
                    values.get("radius") != radius or values.get("mask_inside") != mask_inside_val):
                    filter_obj.center = (center_x, center_y)
                    filter_obj.radius = radius
                    filter_obj.mask_inside = mask_inside
                    values.update({
                        "center_x": center_x,
                        "center_y": center_y,
                        "radius": radius,
                        "mask_inside": mask_inside_val,
                    })
                    changed = True
        
        return changed
    
    def cleanup(self):
        """Clean up filter parameter controls."""
        if self._trackbar_created:
            cv2.destroyWindow(self.window_name)
            self._trackbar_created = False


def add_enemy_position_markers(debug_outputs: Dict[str, np.ndarray], result: Any) -> Dict[str, np.ndarray]:
    """
    Add visual markers to EnemyPositionSensor debug outputs.
    
    Adds a green dot at the center and a blue dot at the calculated enemy position.
    
    Args:
        debug_outputs: Dictionary mapping labels to debug images
        result: Sensor result (dx, dy) offset tuple or None
        
    Returns:
        Modified debug outputs dictionary with markers added
    """
    if "filtered_enemies" not in debug_outputs:
        return debug_outputs
    
    # Get the filtered enemies image
    filtered_img = debug_outputs["filtered_enemies"].copy()
    h, w = filtered_img.shape[:2]
    
    # Calculate center
    center_x = w // 2
    center_y = h // 2
    
    # Draw green dot at center
    cv2.circle(filtered_img, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Draw blue dot at enemy position if result is available
    if result is not None and isinstance(result, tuple) and len(result) == 2:
        dx, dy = result
        enemy_x = center_x + dx
        enemy_y = center_y + dy
        
        # Only draw if within image bounds
        if 0 <= enemy_x < w and 0 <= enemy_y < h:
            cv2.circle(filtered_img, (enemy_x, enemy_y), 5, (255, 0, 0), -1)
    
    # Update the debug output
    debug_outputs = debug_outputs.copy()
    debug_outputs["filtered_enemies"] = filtered_img
    
    return debug_outputs


def create_debug_outputs_display(debug_outputs: Dict[str, np.ndarray], max_width: int = 400) -> Optional[np.ndarray]:
    """
    Create a grid display of debug output images.
    
    Args:
        debug_outputs: Dictionary mapping labels to debug images
        max_width: Maximum width for each image in the grid
        
    Returns:
        Composite image showing all debug outputs, or None if no outputs
    """
    if not debug_outputs:
        return None
    
    # Determine grid layout first to calculate size constraints
    num_images = len(debug_outputs)
    cols = 2
    rows = (num_images + cols - 1) // cols
    
    # Limit maximum canvas height to prevent overflow (~1200px leaves room for monitor)
    max_canvas_height = 1200
    max_image_height = (max_canvas_height // rows) - 30  # -30 for label text
    
    # Resize all images to consistent size, respecting height constraint
    resized_images = []
    labels = list(debug_outputs.keys())
    
    for label in labels:
        img = debug_outputs[label]
        h, w = img.shape[:2]
        # Scale based on both width and height constraints
        scale = min(max_width / w, max_image_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        resized_images.append((label, resized))
    
    # Find maximum dimensions after resizing
    max_h = max(img.shape[0] for _, img in resized_images)
    max_w = max(img.shape[1] for _, img in resized_images)
    
    # Create canvas
    canvas_h = rows * (max_h + 30)  # +30 for label text
    canvas_w = cols * (max_w + 20)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place images in grid
    for idx, (label, img) in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        
        y_offset = row * (max_h + 30)
        x_offset = col * (max_w + 20)
        
        # Place image
        img_h, img_w = img.shape[:2]
        y_start = y_offset + 25  # Leave space for label
        x_start = x_offset + (max_w - img_w) // 2  # Center horizontally
        
        canvas[y_start:y_start + img_h, x_start:x_start + img_w] = img
        
        # Add label
        cv2.putText(canvas, label, (x_offset + 5, y_offset + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return canvas


class SensorTester:
    """
    Utility for testing sensors with images.
    
    Provides interactive selection of sensors and displays ROI input and results.
    Enhanced with filter parameter controls and debug output display.
    """
    
    def __init__(self, vision: VisionEngine, roi_cache: ROICache):
        """
        Initialize sensor tester.
        
        Args:
            vision: VisionEngine instance
            roi_cache: ROICache instance
        """
        self.vision = vision
        self.roi_cache = roi_cache
        self.sensors: Dict[str, Sensor] = {}
        self.current_sensor: Optional[Sensor] = None
        self.filter_controller = FilterParameterController()
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize all available sensors with default configurations."""
        # Minimap State Sensor (uses ROI cache)
        self.sensors["MinimapStateSensor"] = MinimapStateSensor(self.roi_cache)
        
        # Health Sensor (needs vision engine and HP bar ROIs)
        hp_bar_rois = [
            (1190, 845, 360, 10),  # Char 1
            (1160, 900, 360, 10),  # Char 2
            (1110, 950, 360, 10),  # Char 3
        ]
        self.sensors["HealthSensor"] = HealthSensor(self.vision, hp_bar_rois)
        
        # Minimap Sensor (needs vision engine and ROI)
        # self.sensors["MinimapSensor"] = MinimapSensor(self.vision, roi=(1375, 57, 425, 320))
        
        # Compass Sensor (needs vision engine)
        self.sensors["CompassSensor"] = CompassSensor(self.vision)
        
        # Enemy Position Sensor (uses ROI cache)
        self.sensors["EnemyPositionSensor"] = EnemyPositionSensor(self.roi_cache)
        
        # Player Direction Sensor (uses ROI cache)
        self.sensors["PlayerDirectionSensor"] = PlayerDirectionSensor(self.roi_cache)
        
        # Enable all sensors by default for testing
        for sensor in self.sensors.values():
            sensor.enable()
    
    def list_sensors(self) -> list:
        """Get list of available sensor names."""
        return list(self.sensors.keys())
    
    def select_sensor(self, sensor_name: str, roi_image: Optional[np.ndarray] = None) -> bool:
        """
        Select a sensor by name.
        
        Args:
            sensor_name: Name of the sensor to select
            roi_image: Optional ROI image for setting up filter controls
            
        Returns:
            True if sensor was found and selected, False otherwise
        """
        if sensor_name in self.sensors:
            self.current_sensor = self.sensors[sensor_name]
            
            # Set up filter controls for this sensor with sensor name for unique window
            registered_filters = self.current_sensor.get_registered_filters()
            self.filter_controller.setup_filter_controls(registered_filters, roi_image, sensor_name)
            
            return True
        return False
    
    def get_sensor_input_roi(self, image: np.ndarray, sensor_name: str) -> Optional[Tuple[np.ndarray, str]]:
        """
        Get the ROI input image for a sensor.
        
        Args:
            image: Full screen image
            sensor_name: Name of the sensor
            
        Returns:
            Tuple of (ROI image, label) or None if ROI cannot be determined
        """
        sensor = self.sensors.get(sensor_name)
        if not sensor:
            return None
        
        # Handle sensors that use ROI cache
        if isinstance(sensor, MinimapStateSensor):
            roi_image = self.roi_cache.get_roi("minimap", image)
            if roi_image is not None:
                return (roi_image, "minimap ROI")
        
        if isinstance(sensor, EnemyPositionSensor):
            roi_image = self.roi_cache.get_roi("minimap", image)
            if roi_image is not None:
                return (roi_image, "minimap ROI")
        
        if isinstance(sensor, PlayerDirectionSensor):
            roi_image = self.roi_cache.get_roi("minimap_center_arrow", image)
            if roi_image is not None:
                return (roi_image, "minimap center arrow ROI")
        
        # Handle sensors with explicit ROI
        if isinstance(sensor, HealthSensor):
            # Return first HP bar ROI as example
            if sensor.hp_bar_rois:
                roi = sensor.hp_bar_rois[0]
                roi_image = self.vision.get_roi_slice(image, roi)
                return (roi_image, f"HP Bar ROI 1")
        
        if isinstance(sensor, MinimapSensor):
            roi_image = self.vision.get_roi_slice(image, sensor.roi)
            return (roi_image, "minimap ROI")
        
        # For sensors without explicit ROI, return full image
        return (image, "full image")
    
    def test_sensor(self, image: np.ndarray, sensor_name: str) -> Tuple[Any, Optional[np.ndarray]]:
        """
        Test a sensor with an image.
        
        Args:
            image: Input image to test with
            sensor_name: Name of the sensor to test
            
        Returns:
            Tuple of (sensor_result, roi_image)
        """
        sensor = self.sensors.get(sensor_name)
        if not sensor:
            return None, None
        
        # Update filter parameters from trackbars
        self.filter_controller.update_filter_parameters()
        
        # Get ROI input
        roi_info = self.get_sensor_input_roi(image, sensor_name)
        roi_image = roi_info[0] if roi_info else None
        
        # Run sensor (sensors read from full image, not ROI)
        result = sensor.read(image)
        # print(f"Sensor: {sensor_name}, Result: {result}, ROI: {roi_image.shape if roi_image is not None else 'None'}")
        
        return result, roi_image


def format_sensor_result(sensor_name: str, result: Any) -> str:
    """
    Format sensor result for display.
    
    Args:
        sensor_name: Name of the sensor
        result: Sensor result value
        
    Returns:
        Formatted string representation of the result
    """
    if result is None:
        return "None (sensor unavailable or disabled)"
    
    if isinstance(result, str):
        return result
    
    if isinstance(result, (int, float)):
        return f"{result:.2f}" if isinstance(result, float) else str(result)
    
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return "[] (empty)"
        # Format list nicely
        formatted = [format_sensor_result(sensor_name, item) for item in result]
        return f"[{', '.join(formatted)}]"
    
    if isinstance(result, dict):
        formatted = {k: format_sensor_result(sensor_name, v) for k, v in result.items()}
        return str(formatted)
    
    return str(result)


def create_display_image(
    full_image: np.ndarray,
    roi_image: Optional[np.ndarray],
    roi_label: str,
    sensor_name: str,
    result: Any,
    roi_coords: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Create a display image showing full image, ROI, and sensor results.
    
    Args:
        full_image: Full screen image
        roi_image: Extracted ROI image
        roi_label: Label for the ROI
        sensor_name: Name of the sensor
        result: Sensor result
        
    Returns:
        Display image with annotations
    """
    # Create a composite display
    h, w = full_image.shape[:2]
    
    # Resize full image for display if needed
    max_display_width = 960
    max_display_height = 540
    
    if w > max_display_width or h > max_display_height:
        scale = min(max_display_width / w, max_display_height / h)
        display_w = int(w * scale)
        display_h = int(h * scale)
        display_full = cv2.resize(full_image, (display_w, display_h))
    else:
        display_full = full_image.copy()
        display_w, display_h = w, h
    
    # Draw ROI rectangle on full image
    if roi_image is not None and roi_coords is not None:
        # Draw ROI rectangle using provided coordinates
        x, y, roi_w, roi_h = roi_coords
        # Scale coordinates if image was resized
        if w != display_w:
            scale_x = display_w / w
            scale_y = display_h / h
            x, y = int(x * scale_x), int(y * scale_y)
            roi_w, roi_h = int(roi_w * scale_x), int(roi_h * scale_y)
        cv2.rectangle(display_full, (x, y), (x + roi_w, y + roi_h), (0, 255, 0), 2)
        cv2.putText(display_full, roi_label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Create result panel
    result_text = format_sensor_result(sensor_name, result)
    
    # Create a larger display with ROI and result side by side
    if roi_image is not None:
        roi_h, roi_w = roi_image.shape[:2]
        # Resize ROI if too large
        max_roi_size = 400
        if roi_w > max_roi_size or roi_h > max_roi_size:
            roi_scale = min(max_roi_size / roi_w, max_roi_size / roi_h)
            roi_display_w = int(roi_w * roi_scale)
            roi_display_h = int(roi_h * roi_scale)
            roi_display = cv2.resize(roi_image, (roi_display_w, roi_display_h))
        else:
            roi_display = roi_image.copy()
            roi_display_w, roi_display_h = roi_w, roi_h
        
        # Create composite image: full image on left, ROI on right
        panel_height = max(display_h, roi_display_h + 100)
        panel_width = display_w + roi_display_w + 20
        composite = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Place full image on left
        composite[:display_h, :display_w] = display_full
        
        # Place ROI on right
        composite[:roi_display_h, display_w + 20:display_w + 20 + roi_display_w] = roi_display
        
        # Add text labels and results
        cv2.putText(composite, f"Sensor: {sensor_name}", (10, panel_height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, f"ROI: {roi_label}", (display_w + 30, roi_display_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add result text (wrap if needed)
        result_y = panel_height - 50
        result_lines = []
        words = result_text.split()
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (text_w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            if text_w > panel_width - 20:
                if current_line:
                    result_lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            result_lines.append(current_line)
        
        for i, line in enumerate(result_lines[:3]):  # Limit to 3 lines
            cv2.putText(composite, f"Result: {line}", (10, result_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # No ROI, just show full image with result
        composite = display_full.copy()
        cv2.putText(composite, f"Sensor: {sensor_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        result_text_short = result_text[:50] + "..." if len(result_text) > 50 else result_text
        cv2.putText(composite, f"Result: {result_text_short}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return composite


def print_menu(sensor_tester: SensorTester, current_image_path: Optional[str]):
    """Print the sensor testing menu."""
    print("\n" + "="*60)
    print("Sensor Testing Utility")
    print("="*60)
    if current_image_path:
        print(f"Current Image: {current_image_path}")
    else:
        print("Current Image: Live screen capture")
    print("\nAvailable Sensors:")
    sensors = sensor_tester.list_sensors()
    current_sensor_name = None
    if sensor_tester.current_sensor:
        # Find the key for the current sensor
        for name, sensor in sensor_tester.sensors.items():
            if sensor == sensor_tester.current_sensor:
                current_sensor_name = name
                break
    for i, sensor_name in enumerate(sensors, 1):
        marker = "->" if sensor_name == current_sensor_name else "  "
        print(f"  {marker} {i}. {sensor_name}")
    print("\nCommands:")
    print("  [1-9] - Select sensor by number")
    print("  't'   - Test selected sensor")
    print("  'l'   - Load image from templates/ directory")
    print("  'c'   - Switch to live capture mode")
    print("  'q'   - Quit")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Debug utility for testing sensors")
    parser.add_argument("image_path", nargs="?", help="Path to image file (relative to templates/)")
    args = parser.parse_args()
    
    # Configuration: Uses constants from src/states/movement/constants.py
    vision = VisionEngine(window_offset=WINDOW_OFFSET, resolution=RESOLUTION)
    roi_cache = ROICache(vision)
    roi_cache.register_roi("minimap", (1375, 57, 425, 320))
    roi_cache.register_roi("minimap_center_arrow", (1575, 202, 30, 30))
    
    # Load templates that sensors might need
    template_dir = "templates"
    templates_to_load = ["minimap_outline", "hp_container"]
    for template_name in templates_to_load:
        path = os.path.join(template_dir, f"{template_name}.png")
        if os.path.exists(path):
            vision.load_template(template_name, path)
    
    sensor_tester = SensorTester(vision, roi_cache)
    
    # Load initial image
    current_image: Optional[np.ndarray] = None
    current_image_path: Optional[str] = None
    use_live_capture = args.image_path is None
    
    if args.image_path:
        image_path = os.path.join(template_dir, args.image_path)
        if os.path.exists(image_path):
            current_image = cv2.imread(image_path)
            current_image_path = args.image_path
            if current_image is None:
                print(f"Error: Could not load image from {image_path}")
                return
            print(f"Loaded image: {image_path}")
        else:
            print(f"Warning: Image not found at {image_path}, using live capture")
            use_live_capture = True
    
    print_menu(sensor_tester, current_image_path)
    
    # Track last displayed state for auto-refresh
    last_display_time = 0
    auto_refresh_delay = 100  # milliseconds between auto-refreshes
    
    try:
        while True:
            if use_live_capture:
                current_image = vision.capture_screen()
            
            # Wait for user input (non-blocking for live capture)
            if use_live_capture:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF  # Use non-blocking for trackbar responsiveness
            
            # Check if filter parameters changed and auto-refresh if sensor is selected
            if sensor_tester.current_sensor is not None and current_image is not None:
                filter_changed = sensor_tester.filter_controller.update_filter_parameters()
                if filter_changed:
                    # Auto-refresh: re-test sensor with updated filters
                    sensor_name = None
                    for name, sensor in sensor_tester.sensors.items():
                        if sensor == sensor_tester.current_sensor:
                            sensor_name = name
                            break
                    
                    if sensor_name:
                        roi_cache.clear_cache()
                        result, roi_image = sensor_tester.test_sensor(current_image, sensor_name)
                        roi_info = sensor_tester.get_sensor_input_roi(current_image, sensor_name)
                        roi_label = roi_info[1] if roi_info else "N/A"
                        
                        roi_coords = None
                        if sensor_name == "MinimapStateSensor":
                            roi_coords = roi_cache.get_coords("minimap")
                        
                        display_img = create_display_image(
                            current_image, roi_image, roi_label, sensor_name, result, roi_coords
                        )
                        cv2.imshow("Sensor Test Result", display_img)
                        
                        # Display debug outputs if available
                        debug_outputs = sensor_tester.current_sensor.get_debug_outputs()
                        if debug_outputs:
                            # Add markers for EnemyPositionSensor
                            if sensor_name == "EnemyPositionSensor":
                                debug_outputs = add_enemy_position_markers(debug_outputs, result)
                            debug_display = create_debug_outputs_display(debug_outputs)
                            if debug_display is not None:
                                cv2.imshow("Debug Outputs", debug_display)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Load image
                image_name = input("\nEnter image filename (relative to templates/): ").strip()
                if image_name:
                    image_path = os.path.join(template_dir, image_name)
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        if img is not None:
                            current_image = img
                            current_image_path = image_name
                            use_live_capture = False
                            # Clear ROI cache for new image
                            roi_cache.clear_cache()
                            print(f"Loaded image: {image_path}")
                        else:
                            print(f"Error: Could not load image from {image_path}")
                    else:
                        print(f"Error: Image not found at {image_path}")
                print_menu(sensor_tester, current_image_path)
            elif key == ord('c'):
                # Switch to live capture
                use_live_capture = True
                current_image_path = None
                print("Switched to live capture mode")
                print_menu(sensor_tester, current_image_path)
            elif key == ord('t'):
                # Test selected sensor (manual refresh)
                if sensor_tester.current_sensor is None:
                    print("No sensor selected. Select a sensor first.")
                    continue
                
                if current_image is None:
                    print("No image available.")
                    continue
                
                sensor_name = None
                for name, sensor in sensor_tester.sensors.items():
                    if sensor == sensor_tester.current_sensor:
                        sensor_name = name
                        break
                
                if sensor_name:
                    # Update filter parameters first
                    sensor_tester.filter_controller.update_filter_parameters()
                    
                    # Clear ROI cache before testing to ensure fresh extraction
                    roi_cache.clear_cache()
                    result, roi_image = sensor_tester.test_sensor(current_image, sensor_name)
                    roi_info = sensor_tester.get_sensor_input_roi(current_image, sensor_name)
                    roi_label = roi_info[1] if roi_info else "N/A"
                    
                    # Get ROI coordinates from cache for drawing
                    roi_coords = None
                    if sensor_name == "MinimapStateSensor":
                        roi_coords = roi_cache.get_coords("minimap")
                    
                    display_img = create_display_image(
                        current_image, roi_image, roi_label, sensor_name, result, roi_coords
                    )
                    cv2.imshow("Sensor Test Result", display_img)
                    
                    # Display debug outputs if available
                    debug_outputs = sensor_tester.current_sensor.get_debug_outputs()
                    if debug_outputs:
                        # Add markers for EnemyPositionSensor
                        if sensor_name == "EnemyPositionSensor":
                            debug_outputs = add_enemy_position_markers(debug_outputs, result)
                        debug_display = create_debug_outputs_display(debug_outputs)
                        if debug_display is not None:
                            cv2.imshow("Debug Outputs", debug_display)
                    
                    print(f"\nSensor: {sensor_name}")
                    print(f"Result: {format_sensor_result(sensor_name, result)}")
            elif ord('1') <= key <= ord('9'):
                # Select sensor by number
                sensor_idx = key - ord('1')
                sensors = sensor_tester.list_sensors()
                if 0 <= sensor_idx < len(sensors):
                    sensor_name = sensors[sensor_idx]
                    # Get ROI image for filter control setup
                    roi_info = None
                    if current_image is not None:
                        roi_info = sensor_tester.get_sensor_input_roi(current_image, sensor_name)
                    roi_image = roi_info[0] if roi_info else None
                    if sensor_tester.select_sensor(sensor_name, roi_image):
                        print(f"Selected sensor: {sensor_name}")
                        print_menu(sensor_tester, current_image_path)
                    else:
                        print(f"Error: Could not select sensor {sensor_name}")
            
            # Display current image if available
            if current_image is not None:
                # Show a simple preview in live mode or when no sensor selected
                if use_live_capture and sensor_tester.current_sensor is None:
                    preview = cv2.resize(current_image, (640, 360))
                    cv2.imshow("Live Preview (press 't' to test sensor)", preview)
                elif not use_live_capture and sensor_tester.current_sensor is None:
                    # Show full image when no sensor selected
                    display = cv2.resize(current_image, (960, 540))
                    cv2.imshow("Image Preview", display)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sensor_tester.filter_controller.cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
