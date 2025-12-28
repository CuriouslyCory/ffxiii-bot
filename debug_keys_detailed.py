from pynput import keyboard

def on_press(key):
    try:
        # Check for virtual key code first (Windows/Linux usually have this)
        vk = getattr(key, 'vk', None)
        
        # Check for char (printable characters)
        char = getattr(key, 'char', None)
        
        # Check for name (special keys)
        name = getattr(key, 'name', None)

        print(f"Key: {key} | VK: {vk} | Char: {char} | Name: {name}")
        
    except Exception as e:
        print(f"Error reading key: {e}")

    if key == keyboard.Key.esc:
        return False

print("Press Numpad keys (with NumLock ON and OFF). Press ESC to quit.")
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

