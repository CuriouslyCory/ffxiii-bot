# Route Mapping and Navigation System

## Overview

The Route Mapping system allows the FFXIII bot to navigate the open world autonomously. The system supports three navigation methods:
1.  **Hybrid Visual Odometry (Recommended)**: Advanced navigation using the minimap as a high-frequency visual reference.
2.  **Landmark Routing**: Visual navigation using template matching to recognize specific environment features.
3.  **Keylog Routing**: Time-based replay of user inputs (movement and camera).

## Core Concepts

### 1. Hybrid Visual Odometry (HVO)
HVO is the most robust navigation method. It records "breadcrumbs" (nodes) every 0.5 seconds, capturing a snapshot of the minimap.
-   **Visual Tracking**: Uses ORB feature matching and homography between the current minimap and the recorded node.
-   **Auto-Calibration**: The system automatically detects and centers the minimap ROI on startup.
-   **Control Logic**: Prioritizes camera rotation to align with the target node's orientation before moving forward at high speed.
-   **Stability**: Uses Exponential Moving Averages (EMA) and circular averaging for smooth movement and to prevent overshoot.
-   **Arrival Check**: Uses a rolling buffer to average drift values, ensuring the bot truly reaches a node before advancing (default: <20px dist, <15° angle).

### 2. Visual Landmarks (Landmark Routing)
A "node" in a path is a 150x150 pixel image of a distinct object or texture in the game world. The bot navigates by finding these targets on screen, centering them, and moving forward.

### 3. Keylog Events (Keylog Routing)
A route consists of a precise timeline of key press and release events.
-   **Movement**: WASD keys.
-   **Camera**: Numpad keys (2, 4, 6, 8) mapped to the Right Analog Stick.

## Usage Guide

### Recording a Path

1.  **Enter Movement State**: Ensure you are in the open world (minimap visible).
2.  **Start Recording**: Press **`r`**.
3.  **Select Type**:
    -   Press **`1`** for **Landmark Routing**.
    -   Press **`2`** for **Hybrid Visual Odometry**.

#### Option A: Hybrid Visual Odometry (HVO)
1.  **Move Naturally**: Just play the game and move from point A to point B. The bot automatically captures minimap nodes every 0.5 seconds.
2.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

#### Option B: Landmark Routing
1.  **Capture Landmarks**:
    -   Navigate manually. Center a distinct object.
    -   Press **`t`** to capture it as a landmark.
    -   *Tip: Landmarks should be relatively close (3-5s distance).*
2.  **Correct Mistakes**: Press **`g`** to retake the last captured landmark.
3.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

### Playing a Path

1.  **Select Route**:
    -   Press **`p`** to list available routes.
    -   Press the number key (**`1-9`**) to load a route.
2.  **Start/Resume**:
    -   Press **`u`** to begin navigation.
    -   **Resume Logic**: If a route is already in progress, the bot will ask if you want to start from the beginning or resume from the last reached node.
    -   **Looping**: Hybrid routes automatically restart at Node 0 once the end is reached.
3.  **Stop**:
    -   Press **`ESC`** to stop playback/recording immediately. This clears the active route state.

### Debugging & Recovery (Hybrid)

-   **Debug Window**: During Hybrid playback, a window displays the target minimap, current minimap, feature matches, and virtual controller state.
-   **Recovery States**:
    -   **REVERSING**: If tracking is lost, the bot reverses the last camera rotation to try and find a match.
    -   **RETRY 1-3**: A staged recovery with slowing spin speeds to re-acquire tracking.
    -   **LOST**: Tracking is fully lost; the bot stops moving.
-   **Lookahead Recovery**: If the current node tracking is weak, the bot scans future nodes. If a strong match is found ahead, it will "jump" forward to recover the path.

## Technical Architecture

### Hybrid Navigation Engine
-   **Color Filtering**: Minimap features are isolated using HSV filters capturing Cyan/Blue lines and the Gold arrow indicator.
-   **Feature Detection**: Uses `ORB_create(nfeatures=500)` with a circular mask to focus only on the minimap content.
-   **Control Parameters**:
    -   `kp_rot`: Proportional gain for camera rotation.
    -   `kp_strafe`: Proportional gain for lateral character movement.
    -   `speed_factor`: Scales forward movement based on alignment.

### Database Storage
Routes are stored in `data/routes.db`.
-   **Routes Table**: Stores metadata and route `type` ('HYBRID', 'LANDMARK', or 'KEYLOG').
-   **Active State**: Tracks `current_idx` for the active route to allow resuming after interruptions.

## Key Bindings Summary

| Key | Mode | Action |
| :--- | :--- | :--- |
| **`r`** | IDLE | Start Recording (Prompts for type) |
| **`p`** | IDLE | List & Select Route |
| **`u`** | IDLE | Start/Resume Playback |
| **`ESC`** | ANY | Stop / Cancel (Clears state) |
| **`y`** | REC (Any) | Finish & Save |
| **`t`** | REC (Landmark) | Capture Landmark |
| **`g`** | REC (Landmark) | Retake Last Landmark |
| **`WASD`** | REC (Keylog) | Move Character |
| **`2,4,6,8`** | REC (Keylog) | Move Camera (Numpad) |
| **`[` / `]`** | MANAGING | Prev/Next Step (Landmark Only) |
| **`,` / `.`** | MANAGING | Prev/Next Image (Landmark Only) |
