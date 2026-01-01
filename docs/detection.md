# Vision Detection and Debugging

This document describes how the bot detects game states using computer vision and how to debug or calibrate these detections.

## Detection Architecture

The bot uses template matching and Region of Interest (ROI) slicing via OpenCV (`cv2`) to identify game states.

### 1. Vision Engine (`src/core/vision.py`)
The `VisionEngine` is the core component responsible for:
- **Screen Capture**: Uses `mss` for high-performance screen grabbing.
- **Template Matching**: Uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED` for fuzzy matching.
- **ROI Optimization**: Restricts searches to specific screen areas to improve performance and reduce false positives.
- **Visualization**: Provides helper methods (`draw_roi`, `draw_match`) to overlay debug information.
- **Feature Matching**: Includes ORB-based feature detection and matching for visual odometry.

### 2. State Detection (`src/states/*.py`)
Each state (e.g., `BattleState`, `MovementState`) defines its own `is_active` logic:
- **Movement**: Looks for the `minimap_outline` in the upper-right quadrant.
- **Battle**: Looks for `paradigm_shift` or `hp_container` in the bottom half of the screen.
- **Results**: Looks for the `battle_results` header in the top-left quadrant.

### 3. Image Filters (`src/filters/*.py`)
The bot uses a composable filter system for image processing:
- **Color Filters**: HSV-based filters for isolating specific color ranges (blue, gold, red/alert)
- **Edge Filters**: Canny and edge detection filters for feature extraction
- **Composite Filters**: Chain multiple filters together with additive (union) or progressive (intersection) composition
- Filters are reusable and can be applied to any image processing pipeline

### 4. Sensors (`src/sensors/*.py`)
Sensors provide state-specific detection capabilities:
- **Base Sensor**: Abstract interface with enable/disable for lazy evaluation
- **Sensor Registry**: Manages which sensors are available to which states
- **Example Sensors**: HealthSensor (HP percentage), MinimapSensor (minimap extraction), CompassSensor (orientation)
- Sensors only process when enabled, allowing performance optimization

---

## Debugging Feature

To visually verify that the bot "sees" what it's supposed to, use the Vision Debugger.

### Vision Debugger (`src/debug-vision.py`)
This utility provides a real-time HUD of the bot's vision system.

#### How to run:
```bash
python src/debug-vision.py
```

**Note**: The VisionEngine has been moved to `src/core/vision.py` in the new architecture.

#### Visual Indicators:
- **Blue Rectangles**: Defined Regions of Interest (ROIs). These are the "search zones" for specific templates.
- **Red Rectangles**: Successful template matches. Includes the template name and confidence score (e.g., `minimap_outline (0.85)`).

#### Controls:
- `q`: Quit the debugger.
- `s`: Save a screenshot (`debug_capture.png`) of the current frame with overlays for offline analysis.

---

## Calibration Guide

If the bot is failing to detect a state or incorrectly identifying one, follow these steps:

### 1. Adjusting ROIs
If a UI element has moved or the game resolution changed, update the ROIs in the corresponding state file. 
- **Format**: `(x, y, width, height)`
- **Tip**: Use `debug-vision.py` to see where the current blue boxes are and adjust until they perfectly frame the intended UI element.

### 2. Adjusting Thresholds
If detection is too strict (missing matches) or too loose (false positives), adjust the `threshold` (0.0 to 1.0):
- **Higher Threshold (e.g., 0.9)**: More accurate, but may miss matches if the UI is slightly transparent or moving.
- **Lower Threshold (e.g., 0.4)**: More lenient, but risks "hallucinating" matches in complex backgrounds.

### 3. Updating Templates
If the game's UI has changed significantly:
1. Run `src/debug-vision.py`.
2. Press `s` to save a frame.
3. Use an image editor to crop the new UI element from the saved `debug_capture.png`.
4. Replace the corresponding file in the `templates/` directory.

---

## Architecture Overview

The bot follows a clean architecture with clear separation of concerns:

- **Core Engine** (`src/core/`): VisionEngine, Controller, StateManager
- **States** (`src/states/`): Game state classes with `is_active()` and `execute()` methods
- **Skills** (`src/skills/`): Bot interaction capabilities (movement, buttons, macros)
- **Sensors** (`src/sensors/`): Game state detection tools (health, minimap, compass)
- **Filters** (`src/filters/`): Reusable image processing filters
- **Visualizers** (`src/visualizers/`): Debug dashboards and visualization panels
- **UI Components** (`src/ui/`): Input handling, menus, and dialogs

This architecture improves maintainability, reusability, and testability.

## Extending Detection

To add a new state detection:
1. **Capture Template**: Add a new `.png` to `templates/`.
2. **Define ROI**: Determine the static area where this UI element appears.
3. **Create State**: Create a new `State` subclass in `src/states/` implementing `is_active()` using `self.vision.find_template(..., roi=...)`.
4. **Register State**: Add the state to StateManager in `src/main.py`.
5. **Update Debugger**: Add the new ROI and template to `src/debug-vision.py`'s `ROIs` and `TEMPLATE_SEARCH_CONFIG` for visual verification.

To add a new sensor:
1. **Create Sensor**: Subclass `Sensor` in `src/sensors/` implementing `read()` method.
2. **Register Sensor**: Register the sensor with `SensorRegistry` in the appropriate state.
3. **Enable/Disable**: Use sensor's `enable()`/`disable()` methods for lazy evaluation.

To add a new filter:
1. **Create Filter**: Subclass `Filter` in `src/filters/` implementing `apply()` method.
2. **Compose Filters**: Use `CompositeFilter` to chain multiple filters together.
3. **Reuse**: Filters are reusable across different image processing pipelines.

