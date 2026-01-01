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

### Example 2: ROI Cache-Based Sensor

```python
# src/sensors/minimap_state.py (simplified)
from src.sensors.base import Sensor
from src.core.roi_cache import ROICache
import numpy as np
import cv2

class MinimapStateSensor(Sensor):
    def __init__(self, roi_cache: ROICache):
        super().__init__("Minimap State Sensor", "Detects minimap frame color")
        self.roi_cache = roi_cache

    def read(self, image: np.ndarray) -> Optional[str]:
        if not self.is_enabled:
            return None

        # Get cached minimap ROI
        minimap_roi = self.roi_cache.get_roi("minimap", image)
        if minimap_roi is None:
            return None

        # Analyze minimap frame color
        state = self._detect_color(minimap_roi)
        return state  # "movement" or "hostile_detected"
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
4. **Interactive Controls**:
   - `[1-9]` - Select sensor by number
   - `t` - Test selected sensor
   - `l` - Load image from templates/ directory
   - `c` - Switch to live capture mode
   - `q` - Quit

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

### Pattern 1: Color Detection

```python
import cv2
import numpy as np

hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_bound, upper_bound)
count = np.count_nonzero(mask)
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
