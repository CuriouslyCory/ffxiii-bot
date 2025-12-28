import os
import sys
from pathlib import Path

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision import VisionEngine
from src.controller import Controller
from src.states.manager import StateManager
from src.states.movement import MovementState
from src.states.battle import BattleState
from src.states.results import ResultsState

def main():
    """
    Main entry point for the FFXIII automation bot.
    
    Initializes vision, controller, and state manager, then enters the main loop.
    """
    # Configuration: Adjust these to match your environment
    # WINDOW_OFFSET should be the (x, y) coordinate of the game's top-left corner.
    WINDOW_OFFSET = (0, 0) 
    RESOLUTION = (1920, 1080)
    
    vision = VisionEngine(window_offset=WINDOW_OFFSET, resolution=RESOLUTION)
    controller = Controller()
    
    # Load templates from the templates/ directory
    # These images must be provided by the user for the bot to function correctly.
    template_dir = "templates"
    required_templates = [
        "minimap_outline.png",
        "paradigm_shift.png",
        "hp_container.png",
        "battle_results.png"
    ]
    
    print("--- FFXIII Automation Bot ---")
    print(f"Loading templates from {template_dir}...")
    
    templates_loaded = 0
    for t in required_templates:
        path = os.path.join(template_dir, t)
        if os.path.exists(path):
            try:
                vision.load_template(t.split('.')[0], path)
                templates_loaded += 1
                print(f" [OK] {t}")
            except Exception as e:
                print(f" [ERROR] Failed to load {t}: {e}")
        else:
            print(f" [MISSING] {t} - State detection using this template will not work.")

    if templates_loaded == 0:
        print("\nError: No templates were loaded. Please add template images to the 'templates/' directory.")
        # We continue anyway to allow the user to see the initialization, 
        # but the bot won't find any states.

    manager = StateManager(vision, controller)
    
    # Register game states
    manager.add_state(MovementState(manager))
    manager.add_state(BattleState(manager))
    manager.add_state(ResultsState(manager))
    
    print("\nBot initialized. Ready to run.")
    print("Instructions:")
    print(" 1. Ensure FFXIII is running in 1920x1080 windowed mode.")
    print(f" 2. Ensure the window is at screen position {WINDOW_OFFSET}.")
    print(" 3. Press Ctrl+C in this terminal to stop the bot.\n")
    
    # Start the state manager loop
    # Frequency defines how often the screen is captured and analyzed.
    manager.run(frequency=0.2) 

if __name__ == "__main__":
    main()

