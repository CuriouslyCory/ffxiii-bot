#!/usr/bin/env python3
"""
Debug utility for testing sensors.

Allows testing individual sensors with specific images or live screen capture.
Shows sensor input (ROI) and results/telemetry data.

Usage:
    python src/debug-sensors.py [image_path]
    
    If image_path is provided (relative to templates/), loads that image.
    Otherwise, uses live screen capture.
"""
import cv2
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
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
from src.states.movement.constants import WINDOW_OFFSET, RESOLUTION


class SensorTester:
    """
    Utility for testing sensors with images.
    
    Provides interactive selection of sensors and displays ROI input and results.
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
        self._initialize_sensors()
    
    def _initialize_sensors(self):
        """Initialize all available sensors with default configurations."""
        # Minimap State Sensor (uses ROI cache)
        self.sensors["MinimapStateSensor"] = MinimapStateSensor(self.roi_cache)
        
        # Health Sensor (needs vision engine and HP bar ROIs)
        hp_bar_rois = [
            (1450, 850, 200, 10),  # Char 1
            (1450, 880, 200, 10),  # Char 2
            (1450, 910, 200, 10),  # Char 3
        ]
        self.sensors["HealthSensor"] = HealthSensor(self.vision, hp_bar_rois)
        
        # Minimap Sensor (needs vision engine and ROI)
        self.sensors["MinimapSensor"] = MinimapSensor(self.vision, roi=(1375, 57, 425, 320))
        
        # Compass Sensor (needs vision engine)
        self.sensors["CompassSensor"] = CompassSensor(self.vision)
        
        # Enable all sensors by default for testing
        for sensor in self.sensors.values():
            sensor.enable()
    
    def list_sensors(self) -> list:
        """Get list of available sensor names."""
        return list(self.sensors.keys())
    
    def select_sensor(self, sensor_name: str) -> bool:
        """
        Select a sensor by name.
        
        Args:
            sensor_name: Name of the sensor to select
            
        Returns:
            True if sensor was found and selected, False otherwise
        """
        if sensor_name in self.sensors:
            self.current_sensor = self.sensors[sensor_name]
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
        
        # Get ROI input
        roi_info = self.get_sensor_input_roi(image, sensor_name)
        roi_image = roi_info[0] if roi_info else None
        
        # Run sensor (sensors read from full image, not ROI)
        result = sensor.read(image)
        print(f"Sensor: {sensor_name}, Result: {result}, ROI: {roi_image.shape if roi_image is not None else 'None'}")
        
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
    
    try:
        while True:
            if use_live_capture:
                current_image = vision.capture_screen()
            
            # Wait for user input (non-blocking for live capture)
            if use_live_capture:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF
            
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
                # Test selected sensor
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
                    
                    print(f"\nSensor: {sensor_name}")
                    print(f"Result: {format_sensor_result(sensor_name, result)}")
            elif ord('1') <= key <= ord('9'):
                # Select sensor by number
                sensor_idx = key - ord('1')
                sensors = sensor_tester.list_sensors()
                if 0 <= sensor_idx < len(sensors):
                    sensor_name = sensors[sensor_idx]
                    if sensor_tester.select_sensor(sensor_name):
                        print(f"Selected sensor: {sensor_name}")
                        print_menu(sensor_tester, current_image_path)
                    else:
                        print(f"Error: Could not select sensor {sensor_name}")
            
            # Display current image if available
            if current_image is not None:
                # Show a simple preview in live mode
                if use_live_capture:
                    preview = cv2.resize(current_image, (640, 360))
                    cv2.imshow("Live Preview (press 't' to test sensor)", preview)
                elif sensor_tester.current_sensor is None:
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
