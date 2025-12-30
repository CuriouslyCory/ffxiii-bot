---
name: Hybrid Visual Odometry Navigation
overview: Implement a Hybrid Navigation system that uses ORB feature matching on the minimap for "Visual Odometry". This converts visual differences (rotation/translation) between current and recorded frames directly into steering inputs (Camera Pan/WASD), solving the drift and orientation issues.
todos:
  - id: vision-orb
    content: Upgrade VisionEngine with ORB, Feature Matching, and Homography
    status: in_progress
  - id: vision-masking
    content: Implement Circular Masking for Minimap ROI
    status: pending
  - id: debug-odometry
    content: Create 'Drift Monitor' Debug Tool to visualize Homography/Error
    status: pending
  - id: recorder-nodes
    content: Update Recorder to save Breadcrumb Nodes (Minimap + Images)
    status: pending
  - id: playback-pid
    content: Implement Visual Servo/PID Controller for Playback
    status: pending
---

# Hybrid Visual Odometry Navigation Plan

## Core Concept: Visual Breadcrumbs

Instead of blind keylog replay or rigid landmark stops, the bot will follow a trail of "Visual Breadcrumbs".The system relies on the **Minimap** as the primary navigation sensor because:

1.  It is high-contrast and 2D.
2.  It rotates with the camera.
3.  **Visual Odometry**: By aligning the _current_ minimap with the _recorded_ minimap, we can extract error vectors that map directly to controls.

## Control Logic (The "Brain")

The `MovementState` playback loop will act as a **Visual Servo**:

- **Heading Error (`d_theta`)**: The rotation difference between current and target minimap.
  - _Action_: Press Camera Left/Right (4/6).
- **Lateral Error (`dx`)**: The horizontal shift.
  - _Action_: Press Strafe Left/Right (A/D).
- **Longitudinal Error (`dy`)**: The vertical shift.
  - _Action_: Press Forward/Back (W/S).

## Implementation Steps

### Phase 1: Vision Engine Upgrade (ORB)

Enhance [`src/vision.py`](src/vision.py) to support Feature Matching.

- **Add `ORB` Detector**: Robust to rotation and scale changes.
- **Add `FeatureMatcher` Class**:
  - Compute Keypoints & Descriptors.
  - Match using `BFMatcher` (Hamming distance).
  - Apply `Lowe's Ratio Test` to filter bad matches.
  - **Homography Calculation**: Use `cv2.findHomography` to derive the transformation matrix.
  - **Extract Movement Vectors**: Decompose the matrix to get `dx, dy` and `angle`.

### Phase 2: Minimap Preprocessing & Odometry

Create robust minimap handling in [`src/states/movement.py`](src/states/movement.py).

- **ROI & Masking**: Define the Minimap ROI. Crucially, apply a **Circular Mask** to ignore the static square border/HUD elements, ensuring we only track the moving map content.
- **Drift Monitor (Debug Tool)**:
  - Create a real-time overlay showing `Current Minimap` vs `Last Recorded Frame`.
  - Draw "Flow Vectors" showing which way the bot thinks it needs to move.
  - _Validation_: Verify that rotating the camera generates a pure `d_theta` signal.

### Phase 3: Hybrid Route Recorder

Update `MovementState` to record "Nodes" instead of just keys or landmarks.

- **Sampling Rate**: Record a Node every ~0.5s - 1.0s or when heading changes > 5 degrees.
- **Node Data Structure**:
  ```json
  {
    "id": 1,
    "timestamp": 12345.67,
    "minimap_path": "route_X/node_1_mm.png",
    "main_view_path": "route_X/node_1_main.png",
    "input_intent": ["w"] // Hints for the solver
  }
  ```

### Phase 4: Path Following Controller

Implement the playback logic in `MovementState`.

- **Target Selection**: Look ahead to the next Breadcrumb.
- **Steering Loop**:

  1.  Get `dx, dy, d_theta` from Vision.
  2.  Apply PID control to Inputs:

      - `Cam_X = PID_Rot(d_theta)`
      - `Move_X = PID_Lat(dx)`
      - `Move_Y = PID_Lon(dy)`
