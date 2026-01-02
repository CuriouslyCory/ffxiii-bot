# Sensor Development Guide

This document explains how to create, implement, and debug sensors in the FFXIII bot.

## Overview

Sensors are specialized components that detect and extract information from game screens. They provide a clean interface for reading game state data (health percentages, minimap state, compass direction, etc.) that can be reused across different states and logic systems.

## Sensor Architecture

### Base Sensor Interface

All sensors inherit from the `Sensor` base class (`src/sensors/base.py`):

```python
from src.sensors.base import Sensor

class MySensor(Sensor):
    def __init__(self, ...):
        super().__init__("My Sensor", "Description of what it detects")
        # Initialize sensor-specific dependencies

    def read(self, image: np.ndarray) -> Any:
        """Read sensor data from the current image."""
        if not self.is_enabled:
            return None
        # Process image and return sensor data

    def is_available(self, image: np.ndarray) -> bool:
        """Check if sensor can provide data for current image."""
        return True  # Override if needed
```

### Key Concepts

- **Enable/Disable**: Sensors can be enabled/disabled for lazy evaluation and performance optimization
- **Image Input**: Sensors receive the full screen capture image in BGR format
- **Return Types**: Sensors can return any type (string, number, list, dict, etc.) based on their purpose
- **ROI Caching**: Use `ROICache` to avoid redundant screen region extractions
- **Filter Registration**: Sensors can register filters for debugging and reuse
- **Debug Outputs**: Sensors can register debug images at different processing stages for visualization

## Creating a New Sensor

### Step 1: Create Sensor File

Create a new file in `src/sensors/` with your sensor class:

```python
# src/sensors/my_sensor.py
from typing import Optional
import numpy as np
import cv2
from .base import Sensor

class MySensor(Sensor):
    def __init__(self, vision_engine):
        super().__init__("My Sensor", "Detects X from Y")
        self.vision = vision_engine

    def read(self, image: np.ndarray) -> Optional[str]:
        if not self.is_enabled:
            return None
        # Your detection logic here
        return "result"
```

### Step 2: Implement Required Methods

#### `__init__()`

- Call `super().__init__()` with sensor name and description
- Store dependencies (VisionEngine, ROICache, etc.)
- Initialize any configuration parameters

#### `read(image: np.ndarray) -> Any`

- **Required**: Check `self.is_enabled` first
- Process the input image
- Return sensor data (or `None` if unavailable)
- Should handle errors gracefully

#### `is_available(image: np.ndarray) -> bool` (Optional)

- Override to check if required UI elements are present
- Useful for sensors that only work in specific game states
- Default implementation returns `True`

### Step 3: Using ROI Cache

For sensors that extract specific screen regions, use the `ROICache` to avoid redundant extractions:

```python
from src.core.roi_cache import ROICache

class MySensor(Sensor):
    def __init__(self, roi_cache: ROICache):
        super().__init__("My Sensor", "Uses cached ROI")
        self.roi_cache = roi_cache

    def read(self, image: np.ndarray):
        if not self.is_enabled:
            return None

        # Get cached ROI (extracted once per frame, reused across sensors)
        roi_image = self.roi_cache.get_roi("my_roi_label", image)
        if roi_image is None:
            return None

        # Process roi_image...
        return result
```

**Important**: ROIs must be registered in `StateManager` (see ROI Registration below).

### Step 4: Export Sensor

Add your sensor to `src/sensors/__init__.py`:

```python
from .my_sensor import MySensor

__all__ = [
    # ... existing sensors ...
    "MySensor",
]
```

## Example Sensors

### Example 1: Simple ROI-Based Sensor

```python
# src/sensors/health.py (simplified)
from src.sensors.base import Sensor
import numpy as np
import cv2

class HealthSensor(Sensor):
    def __init__(self, vision_engine, hp_bar_rois):
        super().__init__("Health Sensor", "Detects character HP percentages")
        self.vision = vision_engine
        self.hp_bar_rois = hp_bar_rois  # List of (x, y, w, h) tuples

    def read(self, image: np.ndarray) -> list[float]:
        if not self.is_enabled:
            return []

        percentages = []
        for roi in self.hp_bar_rois:
            roi_image = self.vision.get_roi_slice(image, roi)
            # Calculate HP percentage from roi_image
            percent = self._calculate_hp(roi_image)
            percentages.append(percent)

        return percentages
```

### Example 2: Filter-Based Sensor with Composable Filters

```python
# src/sensors/minimap_state.py (simplified)
from src.sensors.base import Sensor
from src.core.roi_cache import ROICache
from src.filters.mask import EllipseMaskFilter
from src.filters.color import BlueFilter, AlertFilter
from src.filters.composite import CompositeFilter
import numpy as np

class MinimapStateSensor(Sensor):
    def __init__(self, roi_cache: ROICache):
        super().__init__("Minimap State Sensor", "Detects minimap frame color")
        self.roi_cache = roi_cache
        self._setup_filters()

    def _setup_filters(self):
        # Create mask filters for border region
        outer_mask = EllipseMaskFilter(center=(215, 160), axes=(212, 160), mask_inside=True)
        inner_mask = EllipseMaskFilter(center=(215, 160), axes=(181, 131), mask_inside=False)

        # Compose masks to isolate border region
        minimap_frame = CompositeFilter([outer_mask, inner_mask], mode="progressive")
        self.register_filter("minimap_frame", minimap_frame)

        # Create color detection filters
        blue_filter = BlueFilter()
        minimap_detected = CompositeFilter([minimap_frame, blue_filter], mode="progressive")
        self.register_filter("minimap_detected", minimap_detected)

        alert_filter = AlertFilter()
        minimap_hostile = CompositeFilter([minimap_frame, alert_filter], mode="progressive")
        self.register_filter("minimap_hostile_detected", minimap_hostile)

    def read(self, image: np.ndarray) -> Optional[str]:
        if not self.is_enabled:
            return None

        # Clear debug outputs from previous frame
        self.clear_debug_outputs()

        # Get cached minimap ROI
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        if minimap_roi is None:
            return None

        # Use registered filters to detect state
        filters = self.get_registered_filters()
        masked_frame = filters["minimap_frame"].apply(minimap_roi)
        self.register_debug_output("masked_frame", masked_frame)

        blue_filtered = filters["minimap_detected"].apply(minimap_roi)
        self.register_debug_output("blue_filtered", blue_filtered)

        red_filtered = filters["minimap_hostile_detected"].apply(minimap_roi)
        self.register_debug_output("red_filtered", red_filtered)

        # Determine state based on pixel counts...
        return "movement"  # or "hostile_detected"
```

### Example 3: Complex Detection Sensor

```python
# src/sensors/compass.py (simplified)
from src.sensors.base import Sensor
import numpy as np

class CompassSensor(Sensor):
    def __init__(self, vision_engine):
        super().__init__("Compass Sensor", "Determines player compass direction")
        self.vision = vision_engine

    def read(self, image: np.ndarray) -> Optional[float]:
        if not self.is_enabled:
            return None

        # Check if minimap is available first
        if not self.is_available(image):
            return None

        # Complex detection logic using feature matching, etc.
        direction = self._calculate_direction(image)
        return direction  # 0-360 degrees

    def is_available(self, image: np.ndarray) -> bool:
        # Check if required UI elements are present
        h, w = image.shape[:2]
        roi = (w // 2, 0, w // 2, h // 2)
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=roi)
        return match is not None
```

## ROI Registration

ROIs used by sensors should be registered in `StateManager` initialization (`src/core/manager.py`):

```python
# In StateManager.__init__()
self.roi_cache.register_roi("minimap", (1375, 57, 320, 425))
self.roi_cache.register_roi("my_roi", (x, y, width, height))
```

**ROI Format**: `(x, y, width, height)` where:

- `x, y`: Top-left corner coordinates
- `width, height`: Dimensions of the ROI

## Using Sensors in States

Sensors can be instantiated and used in state classes:

```python
from src.sensors.my_sensor import MySensor

class MyState(State):
    def __init__(self, manager):
        super().__init__(manager)
        # Create sensor instance
        self.my_sensor = MySensor(self.manager.roi_cache)

    def execute(self, image):
        # Enable sensor when needed
        self.my_sensor.enable()

        # Read sensor data
        result = self.my_sensor.read(image)
        if result:
            # Use sensor data...
            pass
```

## Using Filters in Sensors

### Filter Composition System

The filter system allows you to compose reusable filter pipelines combining mask filters and color filters in any order.

#### Available Filter Types

1. **Mask Filters** (`src/filters/mask.py`):

   - `EllipseMaskFilter`: Elliptical masks with configurable center, axes, and inside/outside flag
   - `RectangleMaskFilter`: Rectangular masks with configurable bounds
   - `CircleMaskFilter`: Circular masks with configurable center and radius

2. **Color Filters** (`src/filters/color.py`):

   - `HSVFilter`: Generic HSV color range filter
   - `BlueFilter`: Pre-configured blue color filter
   - `AlertFilter`: Pre-configured red/alert color filter (handles HSV wheel wrap)
   - `GoldFilter`: Pre-configured gold color filter

3. **Composite Filters** (`src/filters/composite.py`):
   - `CompositeFilter`: Chains multiple filters together
   - **Modes**:
     - `"progressive"`: Sequential application (mask then color = masked color filter)
     - `"additive"`: Combines results with OR operation (blue + gold = both visible)

#### Registering Filters with Sensors

Sensors can register filters for debugging and reuse:

```python
def _setup_filters(self):
    # Create and register individual filters
    outer_mask = EllipseMaskFilter(center=(100, 100), axes=(50, 50), mask_inside=True)
    self.register_filter("outer_mask", outer_mask)

    # Create and register composite filters
    blue_filter = BlueFilter()
    minimap_frame = CompositeFilter([outer_mask], mode="progressive")
    minimap_detected = CompositeFilter([minimap_frame, blue_filter], mode="progressive")
    self.register_filter("minimap_detected", minimap_detected)
```

#### Debug Output Registration

Register debug images at different processing stages:

```python
def read(self, image: np.ndarray):
    if not self.is_enabled:
        return None

    # Clear previous debug outputs
    self.clear_debug_outputs()

    # Process and register debug outputs
    roi_image = self.roi_cache.get_roi("minimap", image)
    masked = self.mask_filter.apply(roi_image)
    self.register_debug_output("masked_image", masked)

    filtered = self.color_filter.apply(masked)
    self.register_debug_output("filtered_image", filtered)

    # Return result...
```

## Debugging Sensors

### Sensor Debug Utility

Use the sensor debugging utility (`src/debug-sensors.py`) to test sensors independently:

```bash
# Test with live screen capture
python src/debug-sensors.py

# Test with a specific image
python src/debug-sensors.py example_battle.png
```

### Debug Utility Features

1. **Sensor Selection**: Select sensors from a numbered list
2. **Visual Display**:
   - Shows full image with ROI highlighted
   - Displays extracted ROI image
   - Shows sensor results/telemetry as text
3. **Image Loading**: Load images from `templates/` directory or use live capture
4. **Filter Parameter Controls** (NEW):
   - **HSV Filters**: Real-time trackbars for H, S, V lower/upper bounds
   - **Mask Filters**: Controls for shape parameters (center, dimensions, inside/outside flag)
   - Changes are automatically applied and reflected in debug outputs
5. **Debug Output Display** (NEW):
   - Grid view of all registered debug outputs from the sensor
   - Shows intermediate processing stages (masked images, filtered images, etc.)
   - Updates in real-time as filter parameters change
6. **Interactive Controls**:
   - `[1-9]` - Select sensor by number
   - `t` - Test selected sensor (manual refresh)
   - `l` - Load image from templates/ directory
   - `c` - Switch to live capture mode
   - `q` - Quit

### Using Filter Parameter Controls

When you select a sensor with registered filters:

1. **Filter Parameters Window**: Opens automatically showing trackbars for all registered filters
2. **HSV Filter Controls**:
   - 6 trackbars per filter: H_lower, S_lower, V_lower, H_upper, S_upper, V_upper
   - Adjust ranges in real-time to fine-tune color detection
3. **Mask Filter Controls**:
   - **Ellipse**: center_x, center_y, radius_x, radius_y, mask_inside (checkbox)
   - **Rectangle**: x, y, width, height, mask_inside (checkbox)
   - **Circle**: center_x, center_y, radius, mask_inside (checkbox)
4. **Real-time Updates**: As you adjust trackbars, the sensor automatically re-runs and updates:
   - Sensor Test Result window (shows ROI and final result)
   - Debug Outputs window (shows all registered debug images)

### Debug Workflow with Filters

1. **Load Test Image**:

   ```bash
   python src/debug-sensors.py my_test_image.png
   ```

2. **Select Sensor**: Press the number key corresponding to your sensor

3. **View Initial Results**: The Filter Parameters window opens automatically

4. **Adjust Filters**:

   - Use trackbars to fine-tune HSV ranges or mask dimensions
   - Watch debug outputs update in real-time
   - Iterate until results look correct

5. **Save Parameters**: Once satisfied, copy the filter parameter values into your sensor code

### Debug Workflow

1. **Capture Test Image**:

   - Run the bot or use `debug-vision.py` to capture a frame
   - Save the image to `templates/` directory

2. **Test Sensor**:

   ```bash
   python src/debug-sensors.py my_test_image.png
   ```

3. **Select Sensor**: Press the number key corresponding to your sensor

4. **Run Test**: Press `t` to test the sensor and view results

5. **Iterate**: Modify sensor code and re-test until working correctly

### Adding Sensor to Debug Utility

To make your sensor available in the debug utility, add it to `SensorTester._initialize_sensors()` in `src/debug-sensors.py`:

```python
def _initialize_sensors(self):
    # ... existing sensors ...

    # Your new sensor
    self.sensors["MySensor"] = MySensor(self.roi_cache)  # or self.vision, etc.
    self.sensors["MySensor"].enable()
```

## Best Practices

### 1. Always Check `is_enabled`

```python
def read(self, image: np.ndarray):
    if not self.is_enabled:
        return None  # or appropriate default value
    # ... processing ...
```

### 2. Use ROI Cache for Shared Regions

If multiple sensors need the same screen region, use `ROICache` to avoid redundant extractions:

```python
# Good: Uses cached ROI
roi = self.roi_cache.get_roi("minimap", image)

# Avoid: Direct extraction (unless ROI is sensor-specific)
roi = self.vision.get_roi_slice(image, (x, y, w, h))
```

### 3. Return Appropriate Types

- Use `None` to indicate unavailable data
- Use specific types (str, int, float, list) for available data
- Document return types in docstrings

### 4. Handle Errors Gracefully

```python
def read(self, image: np.ndarray):
    try:
        # Sensor logic
        return result
    except Exception:
        return None  # Fail gracefully
```

### 5. Implement `is_available()` for Conditional Sensors

If your sensor only works in specific game states:

```python
def is_available(self, image: np.ndarray) -> bool:
    # Check if required UI elements are present
    return self.vision.find_template("required_element", image) is not None
```

### 6. Document Sensor Behavior

- Add clear docstrings
- Document return value types and meanings
- Note any dependencies or requirements

## Common Patterns

### Pattern 1: Filter-Based Color Detection

```python
from src.filters.color import HSVFilter

# Create filter
color_filter = HSVFilter(lower=(84, 75, 100), upper=(97, 245, 245))
self.register_filter("color_filter", color_filter)

# Use filter
filtered_image = color_filter.apply(roi_image)
gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
count = np.count_nonzero(gray)
```

### Pattern 2: Mask + Color Filter Composition

```python
from src.filters.mask import EllipseMaskFilter
from src.filters.color import BlueFilter
from src.filters.composite import CompositeFilter

# Create mask to isolate region of interest
mask = EllipseMaskFilter(center=(100, 100), axes=(50, 50), mask_inside=True)
self.register_filter("region_mask", mask)

# Compose mask + color filter
color_filter = BlueFilter()
composed = CompositeFilter([mask, color_filter], mode="progressive")
self.register_filter("filtered_region", composed)

# Use composed filter
result = composed.apply(roi_image)
```

### Pattern 3: Reusable Filter Pipeline

```python
# Create base mask pipeline
outer_mask = EllipseMaskFilter(center=(100, 100), axes=(50, 50), mask_inside=True)
inner_mask = EllipseMaskFilter(center=(100, 100), axes=(40, 40), mask_inside=False)
border_mask = CompositeFilter([outer_mask, inner_mask], mode="progressive")
self.register_filter("border_mask", border_mask)

# Extend with different color filters
blue_detection = CompositeFilter([border_mask, BlueFilter()], mode="progressive")
red_detection = CompositeFilter([border_mask, AlertFilter()], mode="progressive")
self.register_filter("blue_detection", blue_detection)
self.register_filter("red_detection", red_detection)
```

### Pattern 2: Template Matching

```python
match = self.vision.find_template("template_name", image, threshold=0.8, roi=roi)
if match:
    x, y, confidence = match
    # Use match data...
```

### Pattern 3: Percentage Calculation

```python
# For health bars, fill meters, etc.
total_pixels = roi_image.size
filled_pixels = np.count_nonzero(mask)
percentage = (filled_pixels / total_pixels) * 100
```

## Troubleshooting

### Sensor Returns None

- Check if sensor is enabled: `sensor.enable()`
- Verify `is_available()` returns `True` (if implemented)
- Check if ROI is properly registered (if using ROI cache)
- Verify input image is valid (not None, correct format)

### ROI Extraction Fails

- Verify ROI coordinates are correct: `(x, y, width, height)`
- Ensure ROI is within image bounds
- Check if ROI is registered in `StateManager`

### Sensor Results Incorrect

- Use debug utility to inspect ROI input
- Verify image processing logic (color ranges, thresholds, etc.)
- Test with known good/bad examples
- Check if game UI has changed (may need new templates/ROIs)

## Related Documentation

- [Detection Guide](detection.md) - General vision detection architecture
- [ROI Cache](../src/core/roi_cache.py) - ROI caching implementation
- [Base Sensor](../src/sensors/base.py) - Sensor base class API
- [Filter System](../src/filters/) - Composable filter system for image processing
  - [Base Filter](../src/filters/base.py) - Filter base class
  - [Color Filters](../src/filters/color.py) - HSV color filters
  - [Mask Filters](../src/filters/mask.py) - Geometric mask filters
  - [Composite Filters](../src/filters/composite.py) - Filter composition
  - [Filter Registry](../src/filters/registry.py) - Global filter registry
