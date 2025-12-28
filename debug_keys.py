from pynput import keyboard
import time

def on_press(key):
    try:
        if hasattr(key, 'vk'):
            print(f"Key: {key} | VK Code: {key.vk}")
        else:
            print(f"Key: {key} | No VK Code")
    except AttributeError:
        print(f"Key: {key} (special)")
    if key == keyboard.Key.esc:
        return False

print("Press keys to see their codes. Press ESC to quit.")
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
