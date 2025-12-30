import os
import shutil
from src.db import truncate_db

LANDMARK_DIR = "templates/landmarks"

def clear_landmarks():
    """
    Clears all files in the landmark images directory.
    """
    if not os.path.exists(LANDMARK_DIR):
        print(f"[CLEANUP] Landmark directory '{LANDMARK_DIR}' does not exist. Skipping.")
        return

    print(f"[CLEANUP] Clearing files in '{LANDMARK_DIR}'...")
    count = 0
    for filename in os.listdir(LANDMARK_DIR):
        file_path = os.path.join(LANDMARK_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                count += 1
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                count += 1
        except Exception as e:
            print(f"[CLEANUP] Failed to delete {file_path}. Reason: {e}")
    
    print(f"[CLEANUP] Deleted {count} items from '{LANDMARK_DIR}'.")

def main():
    print("--- FFXIII Bot Cleanup Utility ---")
    confirm = input("This will DELETE all routes from the database and ALL landmark images. Are you sure? (y/N): ").strip().lower()
    
    if confirm == 'y':
        truncate_db()
        clear_landmarks()
        print("[CLEANUP] Cleanup complete.")
    else:
        print("[CLEANUP] Aborted.")

if __name__ == "__main__":
    main()
