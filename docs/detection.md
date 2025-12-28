# Vision Detection and Debugging

This document describes how the bot detects game states using computer vision and how to debug or calibrate these detections.

## Detection Architecture

The bot uses template matching and Region of Interest (ROI) slicing via OpenCV (`cv2`) to identify game states.

### 1. Vision Engine (`src/vision.py`)
The `VisionEngine` is the core component responsible for:
- **Screen Capture**: Uses `mss` for high-performance screen grabbing.
- **Template Matching**: Uses `cv2.matchTemplate` with `TM_CCOEFF_NORMED` for fuzzy matching.
- **ROI Optimization**: Restricts searches to specific screen areas to improve performance and reduce false positives.
- **Visualization**: Provides helper methods (`draw_roi`, `draw_match`) to overlay debug information.

### 2. State Detection (`src/states/*.py`)
Each state (e.g., `BattleState`, `MovementState`) defines its own `is_active` logic:
- **Movement**: Looks for the `minimap_outline` in the upper-right quadrant.
- **Battle**: Looks for `paradigm_shift` or `hp_container` in the bottom half of the screen.
- **Results**: Looks for the `battle_results` header in the top-left quadrant.

---

## Debugging Feature

To visually verify that the bot "sees" what it's supposed to, use the Vision Debugger.

### Vision Debugger (`src/debug-vision.py`)
This utility provides a real-time HUD of the bot's vision system.

#### How to run:
```bash
python src/debug-vision.py
```

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

## Extending Detection

To add a new state detection:
1. **Capture Template**: Add a new `.png` to `templates/`.
2. **Define ROI**: Determine the static area where this UI element appears.
3. **Update State**: Implement `is_active` in a new `State` subclass using `self.vision.find_template(..., roi=...)`.
4. **Update Debugger**: Add the new ROI and template to `src/debug-vision.py`'s `ROIs` and `TEMPLATE_SEARCH_CONFIG` for visual verification.

