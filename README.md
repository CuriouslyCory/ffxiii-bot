# FFXIII Bot

Automation bot for Final Fantasy XIII that uses computer vision to detect game states and execute actions.

## Features

- **State Detection**: Automatically detects movement, battle, and results screens
- **Gamepad Support**: Supports both keyboard and gamepad inputs
- **Vision-Based**: Uses template matching to identify game states

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Add template images to the `templates/` directory:
   - `minimap_outline.png` - For detecting movement state
   - `paradigm_shift.png` - For detecting battle state
   - `hp_container.png` - For detecting battle state
   - `battle_results.png` - For detecting results state

3. Configure game window settings in `src/main.py`:
   - Set `WINDOW_OFFSET` to match your game window position
   - Ensure the game is running in 1920x1080 windowed mode

## Gamepad Setup

The bot uses `vgamepad` to emulate an Xbox 360 controller. This should work out-of-the-box on most systems.

### Gamepad Permissions (Linux)

If you encounter permission errors when creating the virtual gamepad, you may need to:

1. Add your user to the `input` group or ensure you have permissions to access `/dev/uinput` (if used by the backend):
   ```bash
   sudo usermod -a -G input $USER
   ```
   (Requires logout/login to take effect)

## Usage

Run the bot:
```bash
python src/main.py
```

The bot will:
1. Capture screenshots of the game window
2. Detect the current game state
3. Execute appropriate actions (movement, battle commands, etc.)
4. Transition between states automatically

Press `Ctrl+C` to stop the bot.

## States

- **MovementState**: Detects minimap and handles navigation
- **BattleState**: Detects battle UI and executes combat actions
- **ResultsState**: Detects post-battle screens and skips through them

