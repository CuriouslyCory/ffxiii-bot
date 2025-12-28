# Gamepad Configuration Notes

## Overview

This document describes the gamepad configuration for the FFXIII bot. The bot uses `vgamepad` to emulate an Xbox 360 controller.

## Virtual Controller

The bot creates a virtual **Xbox 360 Controller**.

## Button Mapping

The bot maps internal action names to Xbox 360 buttons as follows:

- **Action "a"** (Confirm/Select) -> **Xbox A** (South Button)
- **Action "b"** (Cancel/Back) -> **Xbox B** (East Button)
- **Action "x"** (Menu/Auto-Battle) -> **Xbox Y** (North Button)
- **Action "y"** (Map/Abilities) -> **Xbox X** (West Button)

### Note on Mapping

The mapping follows the physical location convention for Xbox controllers:

- `gamepad_a` -> Xbox A (Bottom)
- `gamepad_b` -> Xbox B (Right)
- `gamepad_x` -> Xbox Y (Top)
- `gamepad_y` -> Xbox X (Left)

## Camera Control

Camera movement is mapped to the **Right Analog Stick** of the virtual Xbox 360 controller.

- **Horizontal Pan** -> Right Stick X Axis
- **Vertical Tilt** -> Right Stick Y Axis

## Troubleshooting

### Virtual Controller Not Detected

If the game does not detect the virtual controller:

1. Ensure the bot is running with sufficient permissions (e.g., user is in `input` group on Linux).
2. Check if other controllers are connected that might be taking priority.
3. Restart the game after the bot has initialized the virtual controller.

### Buttons Not Working

If buttons are not triggering actions:

1. Check the console output of the bot for any error messages regarding `vgamepad`.
2. Ensure the game is focused.
