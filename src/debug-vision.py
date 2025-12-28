#!/usr/bin/env python3
"""
Debug utility for visually verifying ROIs and template matching.

This script captures the screen and draws rectangles around defined ROIs
and any detected templates, providing real-time visual feedback.

Usage:
    python src/debug-vision.py
"""

import cv2
import os
import sys
from pathlib import Path

# Add project root to Python path for absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vision import VisionEngine

def main():
    # Configuration: Match these with your game settings/main.py
    WINDOW_OFFSET = (0, 60)
    RESOLUTION = (1920, 1080)
    
    vision = VisionEngine(window_offset=WINDOW_OFFSET, resolution=RESOLUTION)
    
    # Define ROIs from various states for visualization
    ROIS = {
        "Movement (Minimap)": (960, 0, 960, 540),
        "Battle (Paradigm)": (0, 540, 960, 540),
        "Battle (HP UI)": (960, 540, 960, 540),
        "Results (Header)": (0, 0, 960, 540),
        "Battle (HP Bar 1)": (1450, 850, 200, 10),
        "Battle (HP Bar 2)": (1450, 880, 200, 10),
        "Battle (HP Bar 3)": (1450, 910, 200, 10),
    }

    # Map templates to their specific ROIs for searching
    TEMPLATE_SEARCH_CONFIG = [
        {"name": "minimap_outline", "roi_key": "Movement (Minimap)", "threshold": 0.3},
        {"name": "paradigm_shift", "roi_key": "Battle (Paradigm)", "threshold": 0.7},
        {"name": "hp_container", "roi_key": "Battle (HP UI)", "threshold": 0.7},
        {"name": "battle_results", "roi_key": "Results (Header)", "threshold": 0.7},
    ]

    # Load templates
    template_dir = "templates"
    for config in TEMPLATE_SEARCH_CONFIG:
        name = config["name"]
        path = os.path.join(template_dir, f"{name}.png")
        if os.path.exists(path):
            vision.load_template(name, path)
            print(f"Loaded template: {name}")
        else:
            print(f"Warning: Template {name} not found at {path}")

    print("\n--- Vision Debugger ---")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current frame to 'debug_capture.png'")
    print("-----------------------\n")

    try:
        while True:
            # Capture frame
            frame = vision.capture_screen()
            debug_frame = frame.copy()

            # Draw all ROIs
            for label, roi in ROIS.items():
                vision.draw_roi(debug_frame, roi, label=label, color=(255, 200, 0))

            # Search for templates and draw matches
            for config in TEMPLATE_SEARCH_CONFIG:
                name = config["name"]
                roi_key = config.get("roi_key")
                roi = ROIS.get(roi_key) if roi_key else None
                threshold = config.get("threshold", 0.8)

                match = vision.find_template(name, frame, threshold=threshold, roi=roi)
                if match:
                    vision.draw_match(debug_frame, match, name)

            # Display the result
            # Resize for display if the resolution is large
            display_frame = cv2.resize(debug_frame, (960, 540))
            cv2.imshow("Vision Debugger", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("debug_capture.png", debug_frame)
                print("Saved debug frame to debug_capture.png")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

