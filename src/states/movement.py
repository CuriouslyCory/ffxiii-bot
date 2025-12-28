from src.states.base import State
import time
import os
import json
import cv2
import numpy as np
from pynput import keyboard
from src.db import init_db, save_route, load_route, list_routes, set_active_route, get_active_route, clear_active_route

LANDMARK_DIR = "templates/landmarks"

class MovementState(State):
    """
    MovementState handles navigation in the open world using visual landmarks.
    """
    
    def __init__(self, manager):
        super().__init__(manager)
        self.mode = "IDLE"
        self.landmarks = []
        self.current_landmark_idx = 0
        self.listening = False
        self.current_route_id = None
        self.blocking_input = False
        
        # Playback/Seek state
        self.last_seen_time = 0
        self.search_start_time = 0
        self.is_seeking = False
        
        # Seek Logic State (simplified)
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1  # 1 = down, -1 = up
        self.seek_pan_speed = 0.35  # Right stick X value for panning
        self.seek_tilt_speed = 1  # Right stick Y value for tilting
        
        # Input flags
        self.req_record_mode = False
        self.req_capture = False
        self.req_retake = False
        self.req_save_exit = False
        self.req_playback = False
        self.req_list_routes = False
        self.req_select_route = None # Stores integer key (1-9)
        self.req_stop = False # Escape key to stop
        
        # Ensure landmark directory exists
        os.makedirs(LANDMARK_DIR, exist_ok=True)
        init_db()
        
        # Setup keyboard listener (non-blocking)
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        # Check for auto-resume
        self.check_auto_resume()

    def check_auto_resume(self):
        active = get_active_route()
        if active:
            print(f"[RESUME] Found active route {active['route_id']} at step {active['current_idx']}")
            route_data = load_route(active['route_id'])
            if route_data:
                self.load_route_data(route_data)
                self.current_landmark_idx = active['current_idx']
                self.mode = "PLAYBACK"
                self.is_seeking = True
                self.search_start_time = time.time()
                self.current_route_id = active['route_id']

    def load_route_data(self, route_data):
        self.landmarks = route_data['landmarks']
        self.current_route_id = route_data['id']
        # Load templates
        for lm in self.landmarks:
            path = os.path.join(LANDMARK_DIR, lm['filename'])
            if os.path.exists(path):
                self.vision.load_template(lm['name'], path)
        print(f"Loaded route '{route_data['name']}' with {len(self.landmarks)} landmarks.")

    def is_active(self, image) -> bool:
        roi = (960, 0, 960, 540)
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=roi)
        return match is not None

    def on_enter(self):
        self.listening = True
        print(f"\n--- Movement State ({self.mode}) ---")
        print("Controls:")
        print("  'r': Start New Recording")
        print("  'p': Playback Route (List/Select)")
        print("  'u': Resume Current Loaded Route")
        print("  'ESC': Stop Playback/Recording")
        
        if self.mode == "RECORDING":
            print("  [RESUMED] Recording in progress.")
        elif self.mode == "PLAYBACK":
            print("  [RESUMED] Playback in progress.")

    def on_exit(self):
        self.listening = False
        self.controller.release('w')
        self.controller.move_camera(0.0, 0.0) # Stop camera movement explicit
        self.controller.release_all() # Safety release all
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        if self.mode == "PLAYBACK" and self.current_route_id:
             set_active_route(self.current_route_id, self.current_landmark_idx)
             print("[PAUSED] Playback paused (entered another state).")
        elif self.mode == "RECORDING":
            print("[PAUSED] Recording paused (entered another state).")

    def on_press(self, key):
        if not self.listening or self.blocking_input: return
        try:
            # Handle numpad numbers being typed
            k_char = None
            if hasattr(key, 'char') and key.char:
                k_char = key.char.lower()
            elif hasattr(key, 'vk') and key.vk in [87, 88, 89, 83, 84, 85, 79, 80, 81, 90]:
                # Map Numpad keys to their number char equivalent if needed
                # But actually, '1'.. '9' strings usually come from the top row.
                pass

            if key == keyboard.Key.esc:
                self.req_stop = True
                return

            if k_char:
                k = k_char
                if k == 'r': self.req_record_mode = True
                elif k == 't': self.req_capture = True
                elif k == 'g': self.req_retake = True
                elif k == 'y': self.req_save_exit = True
                elif k == 'u': self.req_playback = True
                elif k == 'p': self.req_list_routes = True
                elif k in [str(i) for i in range(1, 10)]: 
                    self.req_select_route = int(k)
        except AttributeError:
            pass

    def execute(self, image):
        if self.req_stop:
            self.req_stop = False
            self.stop_all()
        if self.req_record_mode:
            self.req_record_mode = False
            self.start_recording()
        if self.req_list_routes:
            self.req_list_routes = False
            self.list_available_routes()
        if self.req_select_route is not None:
             idx = self.req_select_route
             self.req_select_route = None
             if self.mode == "SELECTING":
                 self.select_route(idx)
        if self.req_playback:
            self.req_playback = False
            self.start_playback()
        if self.req_save_exit and self.mode == "RECORDING":
            self.req_save_exit = False
            self.finish_recording()

        if self.mode == "RECORDING":
            self.show_recording_preview(image)
            if self.req_capture:
                self.req_capture = False
                self.capture_landmark(image)
            elif self.req_retake:
                self.req_retake = False
                self.retake_last_landmark(image)
        elif self.mode == "PLAYBACK":
            self.handle_playback(image)

    def stop_all(self):
        print("[STOP] Stopping all activities. Returning to IDLE.")
        self.mode = "IDLE"
        self.controller.release_all()
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        clear_active_route()

    def start_recording(self):
        self.mode = "RECORDING"
        self.landmarks = []
        print("\n[RECORDING] Started. Center the first landmark and press 't'.")

    def finish_recording(self):
        self.controller.release_all()
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        self.blocking_input = True
        print("\n[INPUT REQUIRED] Enter name for this route:")
        try:
             name = input("> ").strip()
             if not name: name = f"Route_{int(time.time())}"
             if save_route(name, self.landmarks):
                 print(f"[SAVED] Route '{name}' saved to database.")
             else:
                 print("[ERROR] Failed to save route.")
        except Exception as e:
            print(f"[ERROR] Input error: {e}")
        finally:
            self.blocking_input = False
        self.mode = "IDLE"

    def list_available_routes(self):
        self.mode = "SELECTING"
        routes = list_routes()
        self.available_routes = routes 
        print("\n--- Available Routes ---")
        for i, r in enumerate(routes):
            print(f" {i+1}. {r[1]} (Created: {r[2]})")
        print("Press number key (1-9) to select a route to load.")

    def select_route(self, index):
        if hasattr(self, 'available_routes') and 0 <= index-1 < len(self.available_routes):
            route_id = self.available_routes[index-1][0]
            route_data = load_route(route_id)
            if route_data:
                self.load_route_data(route_data)
                print(f"[SELECTED] Route '{route_data['name']}' loaded. Press 'u' to start.")
                self.mode = "IDLE"
            else:
                print("[ERROR] Failed to load route.")
        else:
            print("[ERROR] Invalid selection.")

    def start_playback(self):
        if not self.landmarks:
            print("[ERROR] No route loaded. Press 'p' to select a route.")
            return
        self.mode = "PLAYBACK"
        if self.current_route_id:
             set_active_route(self.current_route_id, self.current_landmark_idx)
        self.is_seeking = True
        self.search_start_time = time.time()
        self.seek_state = "IDLE" # Reset seeking state
        print(f"\n[PLAYBACK] Starting/Resuming at landmark {self.current_landmark_idx}.")

    def get_center_crop(self, image):
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_w, crop_h = 150, 150
        x = max(0, center_x - crop_w // 2)
        y = max(0, center_y - crop_h // 2)
        return image[y:y+crop_h, x:x+crop_w]

    def show_recording_preview(self, image):
        crop = self.get_center_crop(image)
        preview = crop.copy()
        cv2.rectangle(preview, (0,0), (149,149), (0,255,0), 2)
        cv2.imshow("Landmark Preview", preview)
        cv2.waitKey(1)

    def retake_last_landmark(self, image):
        if not self.landmarks: return
        last_lm = self.landmarks[-1]
        name = last_lm['name']
        filename = last_lm['filename']
        path = os.path.join(LANDMARK_DIR, filename)
        crop = self.get_center_crop(image)
        try:
            cv2.imwrite(path, crop)
            self.vision.load_template(name, path)
            print(f"[RECORDING] Retook landmark {len(self.landmarks)}: {name}")
        except Exception as e:
            print(f"[ERROR] {e}")

    def capture_landmark(self, image):
        crop = self.get_center_crop(image)
        timestamp = int(time.time())
        filename = f"landmark_{timestamp}.png"
        path = os.path.join(LANDMARK_DIR, filename)
        cv2.imwrite(path, crop)
        name = f"lm_{timestamp}"
        self.landmarks.append({"name": name, "filename": filename})
        self.vision.load_template(name, path)
        print(f"[RECORDING] Captured landmark {len(self.landmarks)}: {name}")

    def reset_seek_state(self):
        """Resets the seek logic variables and stops camera movement."""
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1
        self.controller.move_camera(0.0, 0.0)

    def execute_seek(self):
        """
        Executes a simple seek pattern to find a lost landmark.
        
        Behavior:
        - Continuously pans right using the right joystick
        - Simultaneously tilts up or down, alternating every 7 seconds
        - Continues until landmark is found or timeout
        """
        current_time = time.time()
        
        # Initialize if just starting
        if not self.seek_active:
            print("[SEEK] Lost landmark. Starting scan (pan right + alternating tilt).")
            self.seek_active = True
            self.seek_start_time = current_time
            self.seek_last_toggle_time = current_time
            self.seek_vertical_direction = -1  # Start tilting up (negative Y)
        
        # Check for timeout (e.g., 2 minutes of seeking = give up)
        if current_time - self.seek_start_time > 120.0:
            print("[SEEK] FAILED. Timeout after 2 minutes. Stopping playback.")
            self.stop_all()
            return
        
        # Toggle vertical direction every 7 seconds
        if current_time - self.seek_last_toggle_time >= 7.0:
            self.seek_vertical_direction *= -1  # Flip between -1 (up) and 1 (down)
            direction_name = "down" if self.seek_vertical_direction > 0 else "up"
            print(f"[SEEK] Toggling tilt direction: {direction_name}")
            self.seek_last_toggle_time = current_time
        
        # Continuously send camera movement: pan right + current tilt direction
        # Positive X = look right (based on tracking logic where negative = look left)
        cam_x = self.seek_pan_speed  # Always pan right (positive)
        cam_y = self.seek_tilt_speed * self.seek_vertical_direction  # Tilt up (-) or down (+)
        
        # Debug output every few seconds
        elapsed = int(current_time - self.seek_start_time)
        if elapsed % 3 == 0 and not hasattr(self, '_last_seek_debug') or getattr(self, '_last_seek_debug', -1) != elapsed:
            self._last_seek_debug = elapsed
            print(f"[SEEK DEBUG] Sending camera: x={cam_x:.2f}, y={cam_y:.2f}")
        
        self.controller.move_camera(cam_x, cam_y)

    def handle_playback(self, image):
        if not self.landmarks: return

        target = self.landmarks[self.current_landmark_idx]
        target_name = target['name']
        
        # Look for current landmark
        match = self.vision.find_template(target_name, image, threshold=0.7)
        
        # Look for NEXT landmark
        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
        next_target = self.landmarks[next_idx]
        next_match = self.vision.find_template(next_target['name'], image, threshold=0.7)
        
        if next_match:
            print(f"[PLAYBACK] Found next landmark {next_idx} ({next_target['name']}). Switching target.")
            self.current_landmark_idx = next_idx
            match = next_match
            target_name = next_target['name']
            self.search_start_time = time.time()
            self.reset_seek_state() # Found it, reset seek logic
            if self.current_route_id:
                set_active_route(self.current_route_id, self.current_landmark_idx)

        # Show debug visualization with bounding box
        self._show_debug_view(image, match, target_name)

        if match:
            self.reset_seek_state() # Found it, reset seek logic
            self.last_seen_time = time.time()
            self.search_start_time = time.time()
            
            mx, my, _ = match
            mx += 75; my += 75  # Get center of the 150x150 bounding box
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            
            dx = mx - cx  # Positive = landmark is RIGHT of center, Negative = LEFT
            dy = my - cy  # Positive = landmark is BELOW center, Negative = ABOVE
            deadzone = 50
            
            # Proportional camera control with deadzone
            # Scale movement based on distance from center for smoother tracking
            max_offset_x = w // 3  # Scale factor for X
            max_offset_y = h // 3  # Scale factor for Y
            
            if abs(dx) > deadzone:
                # Proportional control: further from center = faster movement
                # dx < 0 (LEFT of center) -> look LEFT (cam_x negative)
                # dx > 0 (RIGHT of center) -> look RIGHT (cam_x positive)
                proportion = min(1.0, abs(dx) / max_offset_x)
                cam_x = -proportion if dx < 0 else proportion
            else:
                cam_x = 0.0
                
            if abs(dy) > deadzone:
                # dy < 0 (ABOVE center) -> look UP (cam_y negative)
                # dy > 0 (BELOW center) -> look DOWN (cam_y positive)
                proportion = min(1.0, abs(dy) / max_offset_y)
                cam_y = -proportion if dy < 0 else proportion
            else:
                cam_y = 0.0
            
            # Apply camera movement (both axes simultaneously)
            self.controller.move_camera(cam_x, cam_y)
            
            if abs(dx) < 200: self.controller.press('w')
            else: self.controller.release('w')
                
        else:
            self.controller.release('w')
            # If lost for > 3 seconds, start seeking
            if time.time() - self.search_start_time > 3.0:
                self.execute_seek()
            else:
                # Only stop camera if NOT seeking (let seek control the camera)
                self.controller.move_camera(0.0, 0.0)

    def _show_debug_view(self, image, match, target_name: str):
        """
        Displays a debug window showing the captured image with landmark detection overlay.
        
        :param image: Current captured frame
        :param match: Match tuple (x, y, confidence) or None
        :param target_name: Name of the current target landmark
        """
        # Scale down for display (50%)
        scale = 0.5
        h, w = image.shape[:2]
        debug_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        dh, dw = debug_img.shape[:2]
        
        # Screen center crosshair (yellow)
        cx, cy = dw // 2, dh // 2
        cv2.line(debug_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(debug_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        # Status
        status = "TRACKING" if match else ("SEEKING" if self.seek_active else "SEARCHING")
        color = (0, 255, 0) if match else (0, 165, 255)
        
        if match:
            mx, my, conf = match
            # Scale coordinates
            bx, by = int(mx * scale), int(my * scale)
            bs = int(150 * scale)
            
            # Green bounding box
            cv2.rectangle(debug_img, (bx, by), (bx + bs, by + bs), (0, 255, 0), 2)
            
            # Line to center
            lcx, lcy = bx + bs // 2, by + bs // 2
            cv2.line(debug_img, (lcx, lcy), (cx, cy), (0, 255, 0), 1)
            
            # Confidence
            cv2.putText(debug_img, f"{conf:.2f}", (bx, by - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Overlay text
        cv2.putText(debug_img, f"[{status}]", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_img, f"Target: {target_name}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Landmark {self.current_landmark_idx + 1}/{len(self.landmarks)}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show window
        win = "FFXIII Bot - Debug"
        cv2.imshow(win, debug_img)
        
        # Position to right of game window (once)
        if not hasattr(self, '_debug_win_pos'):
            cv2.moveWindow(win, 1930, 50)
            self._debug_win_pos = True
        
        cv2.waitKey(1)
