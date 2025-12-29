# Route Mapping and Navigation System

## Overview

The Route Mapping system allows the FFXIII bot to navigate the open world autonomously. The system now supports two distinct navigation methods:
1.  **Landmark Routing**: Visual navigation using template matching to recognize environment features.
2.  **Keylog Routing**: Time-based replay of user inputs (movement and camera).

## Core Concepts

### 1. Visual Landmarks (Landmark Routing)
A "node" in a path is a 150x150 pixel image of a distinct object or texture in the game world. The bot navigates by finding these targets on screen, centering them, and moving forward.

### 2. Keylog Events (Keylog Routing)
A route consists of a precise timeline of key press and release events.
-   **Movement**: WASD keys.
-   **Camera**: Numpad keys (2, 4, 6, 8) mapped to the Right Analog Stick.
    -   `2`: Pan Down
    -   `8`: Pan Up
    -   `4`: Pan Left
    -   `6`: Pan Right
-   **Overlap Support**: The system records overlapping inputs (e.g., holding `W` to run while tapping `6` to look right), ensuring complex movement patterns are replicated faithfully.

## Usage Guide

### Recording a Path

1.  **Enter Movement State**: Ensure you are in the open world (minimap visible).
2.  **Start Recording**: Press **`r`**.
3.  **Select Type**:
    -   Press **`1`** for **Landmark Routing**.
    -   Press **`2`** for **Keylog Routing**.

#### Option A: Landmark Routing
1.  **Capture Landmarks**:
    -   Navigate manually. Center a distinct object.
    -   Press **`t`** to capture it as a landmark.
    -   *Tip: Landmarks should be relatively close (3-5s distance).*
2.  **Correct Mistakes**: Press **`g`** to retake the last captured landmark.
3.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

#### Option B: Keylog Routing
1.  **Record Movement**:
    -   Move your character using **WASD**.
    -   Control camera using **Numpad 2, 4, 6, 8**.
    -   *Note: Recording pauses automatically if you enter a battle or menu, and resumes when you return to the open world.*
2.  **Finish**: Press **`y`**. Enter a name in the terminal when prompted.

### Playing a Path

1.  **Select Route**:
    -   Press **`p`**.
    -   Select Filter: **`1`** (Landmark) or **`2`** (Keylog).
    -   Press the number key (**`1-9`**) corresponding to the route you wish to load.
2.  **Start/Resume**:
    -   Press **`u`** to begin navigation.
    -   **Resume Logic**:
        -   **Landmark**: Attempts to find the last known target.
        -   **Keylog**: Pauses playback when entering battle. Upon returning to the open world, it waits **3 seconds** before resuming. It reconstructs the state of held keys (e.g., if you were holding `W` and looking right when battle started) and continues the timeline.
3.  **Stop**:
    -   Press **`ESC`** to stop playback/recording immediately.

### Managing Routes (Review & Delete Images)
*Note: Management is currently only supported for Landmark Routes.*

1.  **Select Route**: Press **`m`**, select route ID.
2.  **Navigation**:
    -   **`[` / `]`**: Previous / Next Step.
    -   **`,` / `.`**: Previous / Next Image.
3.  **Editing**: **`x`** to toggle deletion mark.
4.  **Save/Exit**: **`s`** to save, **`ESC`** to cancel.

## Technical Architecture

### Database Storage
Routes are stored in `data/routes.db`.
-   **Routes Table**: Stores metadata and route `type` ('LANDMARK' or 'KEYLOG').
-   **Keylog Events**: Stores precise `time_offset` and `event_type` ('down'/'up') for each key.
-   **State Tracking**: Current progress (step index or event index) is saved to `active_state`. This allows the bot to resume routes after a crash or restart.

### Navigation Logic (Landmark)
Uses a look-ahead system to detect the next target while moving towards the current one, creating smooth curves. Includes a complex seek/recovery algorithm (360° scan) if the target is lost.

### Navigation Logic (Keylog)
Uses an event-driven replay engine. It handles "drift" caused by pauses (battles) by shifting the start time anchor. Camera inputs (Numpad) are translated into virtual analog stick coordinates to emulate smooth controller movement.

## Key Bindings Summary

| Key | Mode | Action |
| :--- | :--- | :--- |
| **`r`** | IDLE | Start Recording (Select Type) |
| **`p`** | IDLE | Playback Route (Select Type) |
| **`m`** | IDLE | Manage Route (Landmark Only) |
| **`u`** | IDLE | Start/Resume Playback |
| **`ESC`** | ANY | Stop / Cancel |
| **`1` / `2`** | MENU | Select Route Type (Landmark/Keylog) |
| **`1-9`** | MENU | Select Route ID |
| **`t`** | REC (Landmark) | Capture Landmark |
| **`g`** | REC (Landmark) | Retake Last Landmark |
| **`y`** | REC (Any) | Finish & Save |
| **`WASD`** | REC (Keylog) | Move Character |
| **`2,4,6,8`** | REC (Keylog) | Move Camera (Numpad) |
| **`[` / `]`** | MANAGING | Prev/Next Step |
| **`,` / `.`** | MANAGING | Prev/Next Image |
| **`x`** | MANAGING | Toggle Delete Mark |
| **`s`** | MANAGING | Save Changes |
