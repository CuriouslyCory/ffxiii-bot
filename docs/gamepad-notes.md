# Gamepad Configuration Notes

## Overview

This document describes the gamepad configuration for the FFXIII bot, specifically for Nintendo Switch controllers. This configuration was tested and verified to work correctly.

## Controller Type

**Nintendo Switch Controller** (Pro Controller or Joy-Con)

## Physical Button to Evdev Code Mapping

The physical buttons on the Nintendo Switch controller map to evdev codes as follows:

- **B button** (south) → `BTN_SOUTH` (evdev code 304)
- **A button** (east) → `BTN_EAST` (evdev code 305)
- **X button** (north) → `BTN_NORTH` (evdev code 307)
- **Y button** (west) → `BTN_WEST` (evdev code 308)

## Pygame Button Index Mapping

When using pygame to detect button presses, the Nintendo Switch controller uses these indices:

- **Pygame button 0** → Physical B button
- **Pygame button 1** → Physical A button
- **Pygame button 2** → Physical X button
- **Pygame button 3** → Physical Y button

## Code Button Name to Evdev Code Mapping

In the code, button names map to evdev codes as follows:

```python
gamepad_button_map = {
    "a": ecodes.BTN_SOUTH,  # A button -> BTN_SOUTH (B button)
    "b": ecodes.BTN_EAST,   # B button -> BTN_EAST (A button)
    "x": ecodes.BTN_NORTH,  # X button -> BTN_NORTH
    "y": ecodes.BTN_WEST,   # Y button -> BTN_WEST
}
```

### Important: A/B Button Swap

**CRITICAL**: The `"a"` and `"b"` button names in code are swapped from their physical labels:

- `controller.tap("gamepad_a")` sends `BTN_SOUTH` (physical B button)
- `controller.tap("gamepad_b")` sends `BTN_EAST` (physical A button)

This swap was necessary because:

1. The game expects the A button (confirm/select) to be `BTN_SOUTH`
2. The code uses `"gamepad_a"` for the confirm action
3. Therefore, `"gamepad_a"` must map to `BTN_SOUTH` (which is the physical B button)

## Virtual Gamepad Device

The bot uses `evdev.UInput` to create a virtual gamepad device that sends button presses to the system.

### Key Points:

1. **Separate Device**: The virtual gamepad is a separate input device from the physical controller
2. **Device Recognition**: The game must recognize and use the virtual device for inputs to work
3. **Device Path**: The virtual device appears as `/dev/input/eventX` where X is a number
4. **Permissions**: Requires appropriate permissions (user in `input` group or sudo)

### Virtual Device Configuration

- **Name**: Uses the physical gamepad name (e.g., "Nintendo Switch Pro Controller")
- **Vendor/Product**: Uses Xbox controller IDs (0x045e, 0x028e) for compatibility
- **Capabilities**: Supports the four main buttons (BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST)

## Usage in Code

### Battle State

```python
# In battle.py - taps the A button (confirm/auto-battle)
self.controller.tap("gamepad_a")  # Sends BTN_SOUTH (physical B button)
```

### Results State

```python
# In results.py - taps the A button (confirm/next)
self.controller.tap("gamepad_a")  # Sends BTN_SOUTH (physical B button)
```

## Debugging

Use `src/debug-controller.py` to verify button mappings:

```bash
python src/debug-controller.py
```

This utility will show:

- Pygame button indices when buttons are pressed
- The corresponding code mapping (e.g., `'gamepad_a'`)
- The evdev code that will be sent (e.g., `BTN_SOUTH`)

## Troubleshooting

### Buttons Not Working in Game

1. **Check virtual device**: Ensure the virtual gamepad device was created successfully
2. **Game recognition**: The game must recognize the virtual device (may need to configure input settings)
3. **Permissions**: Ensure user has access to `/dev/uinput` (add to `input` group)
4. **Device conflict**: If physical gamepad is connected, game may prefer it over virtual device

### Button Mapping Issues

- If `gamepad_a` triggers the wrong button, check the `gamepad_button_map` in `controller.py`
- Verify pygame button indices match physical buttons using `debug-controller.py`
- Ensure both `controller.py` and `debug-controller.py` have matching mappings

## Files Modified

- `src/controller.py`: Main controller implementation with button mappings
- `src/debug-controller.py`: Debug utility for verifying button mappings
- `src/states/battle.py`: Uses `gamepad_a` for battle actions
- `src/states/results.py`: Uses `gamepad_a` for results screen navigation

## Future Considerations

- If supporting other controller types, add controller detection and mapping selection
- Consider adding configuration file for button mappings
- May need to adjust mappings for different games or controller types
