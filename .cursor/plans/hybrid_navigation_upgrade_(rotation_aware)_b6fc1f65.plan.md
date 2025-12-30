---
name: Hybrid Navigation Upgrade (Rotation Aware)
overview: Implement a robust hybrid navigation system using ORB feature matching on the minimap. Because the minimap rotates with the camera, screen-space error correction directly maps to WASD/Camera inputs, allowing for simplified "Visual Odometry" without absolute world coordinates.
todos:
  - id: vision-orb
    content: Upgrade VisionEngine with ORB feature matching capabilities
    status: completed
  - id: minimap-odometry
    content: Implement Minimap Drift Calculator (Visual Odometry)
    status: pending
  - id: hybrid-recorder
    content: Create Hybrid Route Recorder (Inputs + Visual Breadcrumbs)
    status: pending
  - id: path-follower
    content: Implement Steering Logic for Path Following
    status: pending
---

# Hybrid Navigation & Vision Upgrade Plan (sonnet 4.5 plan)

## Problem Analysis

The current system suffers from two main issues:

1.  **Vision Fragility**: `matchTemplate` is sensitive to scale, rotation, and lighting, causing false positives/negatives in 3D environments.
2.  **Navigation Drift**: Keylog replay is "open-loop" (blind), and the Landmark system is too rigid.
3.  **Minimap Behavior**: The minimap rotates with the camera. This is actually an **advantage**. If we align the current minimap image with the recorded minimap image, we simultaneously correct for:

    -   **Position Error** (Translation) -> Maps to WASD inputs (since W is always "Up" on the minimap).
    -   **Heading Error** (Rotation) -> Maps to Camera Pan inputs (Left/Right).

## Proposed Solution: Visual Breadcrumbs

We will move to a **Hybrid Path Following** system.

### 1. Vision Core Upgrade (ORB Features)

We will implement **Feature Matching (ORB)** to replace/augment Template Matching.

-   **Why**: ORB is rotation-invariant and robust. It allows us to calculate the **Homography** (transformation matrix) between the current view and the recorded view.
-   **Output**: `dx, dy` (Screen translation), `d_theta` (Rotation angle).

### 2. Minimap Localizer (Visual Compass)

We will utilize the minimap for continuous drift correction.

-   **Logic**:
    -   Compare `Current Minimap` vs `Recorded Minimap`.
    -   **Rotation Error (`d_theta`)**: If the current map is rotated relative to the record, we need to Pan Camera.
    -   **Translation Error (`dx, dy`)**: If the current map features are shifted relative to the record, we need to Move Character.
    -   *Crucial Mapping*: Since the minimap is camera-relative, a vertical shift (`dy`) in the image corresponds directly to Forward/Backward movement (`W/S`), and horizontal (`dx`) to Strafe (`A/D`). We don't need to know "North".

### 3. Path Following Logic

Update `MovementState` to use a **Steering Behavior**.

-   **Recording**: 
    -   Save "Breadcrumbs" (Minimap ROI + Center View ROI) every ~1.0 second or significant movement.
    -   Record input "Intent" (Moving Forward, Turning).
-   **Playback**: 
    -   Load the next Breadcrumb.
    -   **PID Controller**:
        -   `Steer_X = Kp * d_theta` (Rotate camera to match map orientation).
        -   `Move_X = Kp * dx` (Strafe to match map center).
        -   `Move_Y = Kp * dy` (Throttle speed to match map center).

## Execution Steps

### Phase 1: Vision Engine Overhaul

1.  Update [`src/vision.py`](src/vision.py) to include an `ORB` detector.
2.  Create a `FeatureMatcher` class that handles:

    -   Detecting keypoints/descriptors.
    -   Matching descriptors (BFMatcher).
    -   Filtering good matches (Lowe's Ratio Test).
    -   **Computing Homography** to extract `dx, dy, angle`.

### Phase 2: Minimap Drift Monitor

1.  Update [`src/states/movement.py`](src/states/movement.py) to extract the Minimap ROI.
2.  Implement a debug mode that overlays the "Recorded" minimap features onto the "Current" minimap and displays the calculated offset vectors.
3.  *Validation*: Verify that rotating the camera produces a clean `angle` error, and moving forward produces a clean `dy` error.

### Phase 3: Hybrid Route Recorder

1.  Update `MovementState` recording logic to save "Route Nodes".
2.  Data Structure:
    ```json
                                    {
                                      "timestamp": 12345,
                                      "minimap_image": "node_1_mm.png",
                                      "main_image": "node_1_main.png",
                                      "inputs": ["w"] 
                                    }
        
    
    ```