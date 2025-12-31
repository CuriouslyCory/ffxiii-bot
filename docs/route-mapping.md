# Route Mapping and Navigation System

## Overview

The Route Mapping system allows the FFXIII bot to navigate the open world autonomously. The system supports three navigation methods:

1.  **Hybrid Visual Odometry (Recommended)**: Advanced navigation using the minimap as a high-frequency visual reference. Generates a composite "Master Map" artifact upon saving.
2.  **Landmark Routing**: Visual navigation using template matching to recognize specific environment features.
3.  **Keylog Routing**: Time-based replay of user inputs (movement and camera).

## Core Concepts

### 1. Hybrid Visual Odometry (HVO)

HVO is the most robust navigation method. It records "breadcrumbs" (nodes) every 0.5 seconds, capturing a snapshot of the minimap.

- **Visual Tracking**: Uses ORB feature matching and homography between the current minimap and the recorded node.
- **Auto-Calibration**: The system automatically detects and centers the minimap ROI on startup.
- **Control Logic**: Prioritizes camera rotation to align with the target node's orientation before moving forward at high speed.
- **Stability**: Uses Exponential Moving Averages (EMA) with rolling windows for smooth movement and to prevent overshoot. The EMA uses separate configurations for position (dx/dy) and angle, each with configurable alpha values and window sizes (default: 30 samples).
- **Arrival Check**: Uses a rolling buffer to average drift values, ensuring the bot truly reaches a node before advancing (default: <20px dist, <15° angle).
- **Master Map Generation**: Upon finishing a recording, the system stitches all minimap nodes into a single composite PNG with a breadcrumb trail showing the recorded path. This artifact is stored in the database for route preview and debugging.

### 2. Visual Landmarks (Landmark Routing)

A "node" in a path is a 150x150 pixel image of a distinct object or texture in the game world. The bot navigates by finding these targets on screen, centering them, and moving forward.

### 3. Keylog Events (Keylog Routing)

A route consists of a precise timeline of key press and release events.

- **Movement**: WASD keys.
- **Camera**: Numpad keys (2, 4, 6, 8) mapped to the Right Analog Stick.

## Usage Guide

### Recording a Path

1.  **Enter Movement State**: Ensure you are in the open world (minimap visible).
2.  **Start Recording**: Press **`r`**.
3.  **Select Type**:
    - Press **`1`** for **Landmark Routing**.
    - Press **`2`** for **Hybrid Visual Odometry**.

#### Option A: Hybrid Visual Odometry (HVO)

1.  **Move Naturally**: Just play the game and move from point A to point B. The bot automatically captures minimap nodes every 0.5 seconds.
2.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

#### Option B: Landmark Routing

1.  **Capture Landmarks**:
    - Navigate manually. Center a distinct object.
    - Press **`t`** to capture it as a landmark.
    - _Tip: Landmarks should be relatively close (3-5s distance)._
2.  **Correct Mistakes**: Press **`g`** to retake the last captured landmark.
3.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

### Playing a Path

1.  **Select Route**:
    - Press **`p`** to list available routes.
    - Press the number key (**`1-9`**) to load a route.
2.  **Start/Resume**:
    - Press **`u`** to begin navigation.
    - **Resume Logic**: If a route is already in progress, the bot will ask if you want to start from the beginning or resume from the last reached node.
    - **Looping**: Hybrid routes automatically restart at Node 0 once the end is reached.
3.  **Stop**:
    - Press **`ESC`** to stop playback/recording immediately. This clears the active route state.

### Debugging & Recovery (Hybrid)

- **Debug Window**: During Hybrid playback, a window displays the target minimap, current minimap, feature matches, and virtual controller state.
- **HSV Filter Debug Mode**: Press **`d`** during playback to toggle an interactive HSV filter debug window. This allows real-time adjustment of color filter ranges using trackbars to fine-tune minimap feature detection.
- **Recovery States**:
  - **REVERSING**: If tracking is lost, the bot reverses the last camera rotation to try and find a match.
  - **RETRY 1-3**: A staged recovery with slowing spin speeds to re-acquire tracking.
  - **LOST**: Tracking is fully lost; the bot stops moving.
- **Lookahead Recovery**: If the current node tracking is weak, the bot scans future nodes. If a strong match is found ahead, it will "jump" forward to recover the path.

### Editing Routes During Playback

- **Add Images**: Press **`t`** during playback to add a new landmark image to the current step.
- **Add to Next Step**: Press **`n`** during playback to add a new landmark image to the next step.
- **Delete Current Image**: Press **`2`** during playback to delete the currently matched image from the current step.
- **Delete Next Image**: Press **`3`** during playback to delete an image from the next step.

## Technical Architecture

### Modular Component Structure

The movement state system has been refactored into a modular architecture for improved maintainability and testability:

- **MovementState**: Main orchestrator that coordinates all components
- **InputHandler**: Processes keyboard input and converts to action requests
- **RouteManager**: Handles route loading, selection, and database operations
- **RouteRecorder**: Manages recording logic (LandmarkRecorder and HybridRecorder subclasses)
- **RoutePlayer**: Manages playback logic (LandmarkPlayer and HybridPlayer subclasses)
- **NavigationController**: Implements drift smoothing, PID control, and arrival detection
- **SeekStrategy**: Handles recovery logic when tracking is lost
- **DebugVisualizer**: Manages all debug windows and visualization
- **Constants**: Centralized configuration values in `src/states/movement/constants.py`

### Hybrid Navigation Engine

- **Color Filtering**: Minimap features are isolated using HSV filters capturing Cyan/Blue lines and the Gold arrow indicator.
- **Feature Detection**: Uses `ORB_create(nfeatures=500)` with a circular mask to focus only on the minimap content.
- **EMA Smoothing**: Exponential Moving Average with rolling windows:
  - Separate alpha values and window sizes for position (dx/dy) vs angle
  - Default: 30-sample windows for both, configurable via constants
  - Angle smoothing uses circular mean to handle ±180° wrap-around
- **Control Parameters** (configurable in `constants.py`):
  - `CAMERA_ROTATION_KP`: Proportional gain for camera rotation (default: 0.03)
  - `STRAFE_KP`: Proportional gain for lateral character movement (default: 0.01)
  - `APPROACH_SLOWDOWN_DISTANCE`: Distance threshold for slowdown near nodes (default: 40px)
  - `EMA_ALPHA_DX_DY`: EMA alpha for position smoothing (default: 0.25)
  - `EMA_ALPHA_ANGLE`: EMA alpha for angle smoothing (default: 0.25)
  - `EMA_WINDOW_SIZE_DX_DY`: Window size for position EMA (default: 30)
  - `EMA_WINDOW_SIZE_ANGLE`: Window size for angle EMA (default: 30)

### Database Storage

Routes are stored in `data/routes.db`.

- **Routes Table**: Stores metadata and route `type` ('HYBRID', 'LANDMARK', or 'KEYLOG').
- **Active State**: Tracks `current_idx` for the active route to allow resuming after interruptions.
- **Master Map**: Hybrid routes store a path to a composite minimap image showing the recorded path.

## Key Bindings Summary

| Key           | Mode                      | Action                               |
| :------------ | :------------------------ | :----------------------------------- |
| **`r`**       | IDLE                      | Start Recording (Prompts for type)   |
| **`p`**       | IDLE                      | List & Select Route                  |
| **`u`**       | IDLE                      | Start/Resume Playback                |
| **`ESC`**     | ANY                       | Stop / Cancel (Clears state)         |
| **`d`**       | PLAYBACK                  | Toggle HSV Filter Debug Mode         |
| **`y`**       | REC (Any)                 | Finish & Save                        |
| **`t`**       | REC (Landmark) / PLAYBACK | Capture Landmark / Add Image to Step |
| **`n`**       | REC (Landmark) / PLAYBACK | Finish Step / Add Image to Next Step |
| **`g`**       | REC (Landmark)            | Retake Last Landmark                 |
| **`2`**       | PLAYBACK                  | Delete Current Image from Step       |
| **`3`**       | PLAYBACK                  | Delete Image from Next Step          |
| **`WASD`**    | REC (Keylog)              | Move Character                       |
| **`2,4,6,8`** | REC (Keylog)              | Move Camera (Numpad)                 |

## Configuration

All navigation parameters are centralized in `src/states/movement/constants.py` for easy tuning:

### EMA Smoothing Configuration

- `EMA_ALPHA_DX_DY`: Smoothing factor for position (dx/dy) - higher = more responsive, lower = smoother
- `EMA_WINDOW_SIZE_DX_DY`: Number of samples in rolling window for position (default: 30)
- `EMA_ALPHA_ANGLE`: Smoothing factor for angle - higher = more responsive, lower = smoother
- `EMA_WINDOW_SIZE_ANGLE`: Number of samples in rolling window for angle (default: 30)

### Navigation Control

- `CAMERA_ROTATION_KP`: Proportional gain for camera rotation correction
- `CAMERA_ROTATION_MAX`: Maximum camera rotation speed
- `STRAFE_KP`: Proportional gain for lateral movement correction
- `APPROACH_SLOWDOWN_DISTANCE`: Distance threshold for automatic slowdown near nodes

### Recovery Parameters

- `COAST_DURATION`: Time to hold last valid input when tracking lost
- `RETRY_ATTEMPTS`: Number of retry phases with progressively slower spin
- `LOOKAHEAD_DEPTH`: Number of future nodes to scan for recovery

## Maintenance & Utilities

### Cleanup Utility

To reset the system (delete all recorded routes and clear all landmark images), run:

```bash
python cleanup.py
```

This will prompt for confirmation before truncating the database and deleting image artifacts.

### Code Organization

The movement state codebase is organized into focused modules:

- `src/states/movement/movement_state.py`: Main orchestrator
- `src/states/movement/input_handler.py`: Keyboard input processing
- `src/states/movement/route_manager.py`: Route database operations
- `src/states/movement/route_recorder.py`: Recording logic
- `src/states/movement/route_player.py`: Playback logic
- `src/states/movement/navigation_controller.py`: Control algorithms
- `src/states/movement/seek_strategy.py`: Recovery strategies
- `src/states/movement/debug_visualizer.py`: Debug windows
- `src/states/movement/constants.py`: Configuration values
