# Route Mapping and Navigation System

## Overview

The Route Mapping system allows the FFXIII bot to navigate the open world autonomously by following a sequence of visual landmarks. Since the game does not provide accessible coordinate data, this system relies on computer vision (Template Matching) to recognize specific visual features in the environment and orient the character towards them.

## Core Concepts

### 1. Visual Landmarks
Instead of (x, y) coordinates, a "node" in a path is a 150x150 pixel image of a distinct object or texture in the game world (e.g., a specific rock, a signpost, a distant mountain peak).

### 2. The Path
A route is simply an ordered list of these landmarks. The bot navigates by:
1.  Finding the current target landmark on screen.
2.  Rotating the camera to center the landmark.
3.  Moving forward while keeping it centered.
4.  Scanning for the *next* landmark in the list.
5.  Seamlessly switching targets when the next one becomes visible.

## Usage Guide

### Recording a Path

1.  **Enter Movement State**: Ensure you are in the open world (minimap visible).
2.  **Start Recording**: Press **`r`**.
    - A "Landmark Preview" window will appear, showing the center crop of your screen.
3.  **Capture Landmarks**:
    - Navigate your character manually.
    - Center a distinct object in your view.
    - Press **`t`** to capture it as a landmark.
    - *Tip: Landmarks should be relatively close to each other (3-5 seconds of running distance).*
4.  **Correct Mistakes**:
    - If you take a bad picture, don't move. Press **`g`** to retake (overwrite) the last captured landmark.
5.  **Finish**:
    - Press **`y`** when done.
    - The bot will pause and ask for a **Route Name** in the terminal window.
    - Enter a name and press Enter. The route is saved to the database.

### Playing a Path

1.  **Select Route**:
    - Press **`p`** to list available routes in the terminal.
    - Press the corresponding number key (**`1-9`**) to load a route.
2.  **Start/Resume**:
    - Press **`u`** to begin navigation.
3.  **Stop**:
    - Press **`ESC`** to stop playback/recording immediately.

## Technical Architecture

### Database Storage
Routes are stored in a SQLite database (`data/routes.db`). This allows for:
-   **Persistence**: Routes survive bot restarts.
-   **Management**: Multiple routes can be stored and named.
-   **State Tracking**: The current route progress (index) is saved to the `active_state` table. If the bot crashes or is stopped, it can resume exactly where it left off.

### Navigation Logic

#### Look-Ahead System
To prevent "stop-and-turn" behavior, the bot employs a look-ahead mechanism. While moving towards Landmark A, it is actively searching for Landmark B. As soon as Landmark B is detected with sufficient confidence, the bot abandons A and starts moving towards B. This creates smooth, continuous curves in movement.

#### Camera Control
The system uses a virtual controller's right analog stick for camera panning. The `move_camera(x, y)` function sends axis values to the virtual gamepad:
-   `x`: -1.0 (Left) to 1.0 (Right)
-   `y`: -1.0 (Up) to 1.0 (Down)
-   **Linux Specifics**: The bot creates a virtual gamepad using evdev/uinput with `ABS_RX` and `ABS_RY` axes for camera control.

#### Recovery Algorithm (Complex Seek)
In FFXIII, entering a battle disorients the camera. When the bot returns to Movement State, it may be facing the wrong way. If the target landmark is not seen for 3 seconds, the bot enters a **Search Mode**:

1.  **360° Horizontal Scan**: Pans right in 7 increments (1s hold, 1s wait).
2.  **Down Phase**: Pans down (0.5s), then performs a 360° scan. Repeats 4 times.
3.  **Up Phase**: Pans up (0.5s), then performs a 360° scan. Repeats 8 times.
4.  **Failure**: If the landmark is not found after 2 full cycles of this pattern, the bot stops to prevent getting stuck in a loop.

## Key Bindings Summary

| Key | Mode | Action |
| :--- | :--- | :--- |
| **`r`** | IDLE | Start Recording Mode |
| **`t`** | RECORDING | Capture Landmark |
| **`g`** | RECORDING | Retake Last Landmark |
| **`y`** | RECORDING | Finish & Save |
| **`p`** | IDLE | List & Select Routes |
| **`u`** | IDLE | Start/Resume Playback |
| **`ESC`** | ANY | Stop All Actions |
| **`1-9`** | SELECTING | Select Route ID |

## Why this approach?

-   **Robustness**: Works purely on visual data, making it independent of memory reading or coordinate injection which can be detected as cheating.
-   **Flexibility**: Users can create custom farming routes for any area of the game.
-   **Resilience**: The complex seek logic ensures the bot can recover from post-battle disorientation, which is critical for long-term automation.

