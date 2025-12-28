from src.states.base import State
import time
import os
import json
import cv2
import numpy as np
from pynput import keyboard
from src.db import init_db, save_route, update_route_structure, load_route, list_routes, set_active_route, get_active_route, clear_active_route

LANDMARK_DIR = "templates/landmarks"

class MovementState(State):
    """
    MovementState handles navigation in the open world using visual landmarks.
    """
    
    def __init__(self, manager):
        super().__init__(manager)
        self.mode = "IDLE"
        self.landmarks = [] # Now a list of steps: [{'name': 'Step_X', 'images': [{'filename': '...', 'name': '...'}]}]
        self.current_landmark_idx = 0 # Step index
        self.listening = False
        self.current_route_id = None
        self.blocking_input = False
        
        # Recording state
        self.current_recording_step_images = [] # List of images for the current step being recorded
        
        # Playback/Seek state
        self.last_seen_time = 0
        self.search_start_time = 0
        self.is_seeking = False
        
        # Seek Logic State (simplified)
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1  # 1 = down, -1 = up
        self.seek_pan_speed = 0.25  # Right stick X value for panning
        self.seek_tilt_speed = 0.15  # Right stick Y value for tilting
        
        # Input flags
        self.req_record_mode = False
        self.req_capture = False # 't' key
        self.req_next_step = False # 'n' key
        self.req_retake = False # 'g' key
        self.req_save_exit = False
        self.req_playback = False
        self.req_list_routes = False
        self.req_select_route = None # Stores integer key (1-9)
        self.req_stop = False # Escape key to stop
        self.req_playback_action = None # For '2', '3' actions during playback
        
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
        # Load templates for all images in all steps
        for step in self.landmarks:
            for img in step['images']:
                path = os.path.join(LANDMARK_DIR, img['filename'])
                if os.path.exists(path):
                    self.vision.load_template(img['name'], path)
        print(f"Loaded route '{route_data['name']}' with {len(self.landmarks)} steps.")

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
        self.controller.move_character(0.0, 0.0)
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
                pass

            if key == keyboard.Key.esc:
                self.req_stop = True
                return

            if k_char:
                k = k_char
                if k == 'r': self.req_record_mode = True
                elif k == 't': self.req_capture = True
                elif k == 'n': self.req_next_step = True
                elif k == 'g': self.req_retake = True
                elif k == 'y': self.req_save_exit = True
                elif k == 'u': self.req_playback = True
                elif k == 'p': self.req_list_routes = True
                elif k in [str(i) for i in range(1, 10)]: 
                    self.req_select_route = int(k)
                elif k == '2': self.req_playback_action = 'delete_current_image'
                elif k == '3': self.req_playback_action = 'delete_next_image'
                
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
        
        # Handle Playback Actions (Edit/Delete)
        if self.req_playback_action:
            action = self.req_playback_action
            self.req_playback_action = None
            if self.mode == "PLAYBACK":
                if action == 'delete_current_image':
                    self.delete_image_from_step(self.current_landmark_idx)
                elif action == 'delete_next_image':
                    if self.landmarks:
                        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                        self.delete_image_from_step(next_idx)

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
                self.capture_image(image)
            elif self.req_next_step:
                self.req_next_step = False
                self.finish_step()
            elif self.req_retake:
                self.req_retake = False
                self.retake_last_image(image)
        elif self.mode == "PLAYBACK":
            # Check for Add Image request during playback
            if self.req_capture:
                self.req_capture = False
                self.add_image_to_playback_step(self.current_landmark_idx, image)
            elif self.req_next_step:
                self.req_next_step = False
                if self.landmarks:
                    next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                    self.add_image_to_playback_step(next_idx, image)

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
        self.current_recording_step_images = []
        print("\n[RECORDING] Started.")
        print("  't': Capture image for current step.")
        print("  'n': Finish step and move to next.")
        print("  'g': Undo last captured image.")
        print("  'y': Finish recording.")

    def finish_recording(self):
        # If there are pending images in the current step, save them as the last step
        if self.current_recording_step_images:
            self.finish_step()
            
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

    def update_db(self):
        if self.current_route_id:
            if update_route_structure(self.current_route_id, self.landmarks):
                 print("[SAVED] Updated route in database.")
            else:
                 print("[ERROR] Failed to update route in database.")
    
    def delete_image_from_step(self, step_idx):
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)): return
        
        # We need to know WHICH image to delete. 
        # For simplicity, if we are in playback, we delete the one that matched best?
        # Since this is async, let's look at the last frame's match if available.
        # But we don't store "last matched image index" easily accessible here.
        # HACK: If there are multiple images, we delete the LAST one added?
        # OR: We delete the one that is currently the "best match" found in handle_playback.
        # Let's use `self.last_match_details` if we store it.
        
        step = self.landmarks[step_idx]
        images = step['images']
        
        if not images:
            # Empty step, delete the step
            self.delete_step(step_idx)
            return

        # Ideally we delete the one currently being tracked.
        # If we are tracking step_idx, we should have info on which image matched.
        target_img_idx = -1
        
        if step_idx == self.current_landmark_idx and hasattr(self, 'current_match_img_index'):
             target_img_idx = self.current_match_img_index
        elif hasattr(self, 'next_match_img_index') and step_idx == (self.current_landmark_idx + 1) % len(self.landmarks):
             target_img_idx = self.next_match_img_index
        
        if target_img_idx != -1 and 0 <= target_img_idx < len(images):
            deleted = images.pop(target_img_idx)
            print(f"[DELETE] Removed image '{deleted['name']}' from step {step_idx}.")
        else:
            # Fallback: Delete last image
            deleted = images.pop()
            print(f"[DELETE] Removed last image '{deleted['name']}' from step {step_idx} (fallback).")
            
        if not images:
            print(f"[DELETE] Step {step_idx} is now empty. Removing step.")
            self.delete_step(step_idx)
        else:
            self.update_db()

    def delete_step(self, index):
        if 0 <= index < len(self.landmarks):
            deleted = self.landmarks.pop(index)
            print(f"[DELETE] Removed step {index}: {deleted['name']}")
            
            # Adjust current index
            if index < self.current_landmark_idx:
                self.current_landmark_idx -= 1
            elif index == self.current_landmark_idx:
                if self.current_landmark_idx >= len(self.landmarks):
                    self.current_landmark_idx = 0
                
                self.search_start_time = time.time()
                self.reset_seek_state()
                self.last_seen_time = 0
            
            self.update_db()
            
            if not self.landmarks:
                 print("[WARN] Route is now empty.")
                 self.stop_all()

    def start_playback(self):
        if not self.landmarks:
            print("[ERROR] No route loaded. Press 'p' to select a route.")
            return
        self.mode = "PLAYBACK"
        if self.current_route_id:
             set_active_route(self.current_route_id, self.current_landmark_idx)
        self.is_seeking = True
        self.search_start_time = time.time()
        self.seek_state = "IDLE" 
        print(f"\n[PLAYBACK] Starting/Resuming at step {self.current_landmark_idx}.")

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
        
        # Show count of images in current step
        count = len(self.current_recording_step_images)
        cv2.putText(preview, f"Step Imgs: {count}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        cv2.imshow("Landmark Preview", preview)
        cv2.waitKey(1)

    def retake_last_image(self, image):
        # In new logic: remove last image from current_recording_step_images
        if self.current_recording_step_images:
            removed = self.current_recording_step_images.pop()
            print(f"[RECORDING] Removed image: {removed['name']}")
        else:
            print("[RECORDING] No images in current step to remove.")

    def capture_image(self, image):
        crop = self.get_center_crop(image)
        timestamp = int(time.time())
        # Add random suffix to avoid name collision in rapid fire
        suffix = np.random.randint(0, 1000)
        filename = f"landmark_{timestamp}_{suffix}.png"
        path = os.path.join(LANDMARK_DIR, filename)
        cv2.imwrite(path, crop)
        name = f"lm_{timestamp}_{suffix}"
        
        self.current_recording_step_images.append({"name": name, "filename": filename})
        self.vision.load_template(name, path)
        print(f"[RECORDING] Captured image {len(self.current_recording_step_images)} for current step.")

    def finish_step(self):
        if not self.current_recording_step_images:
            print("[RECORDING] Cannot finish step: No images captured. Press 't' to capture.")
            return
        
        step_name = f"Step_{len(self.landmarks)}"
        self.landmarks.append({
            "name": step_name,
            "images": list(self.current_recording_step_images)
        })
        print(f"[RECORDING] Finished Step {len(self.landmarks)-1} with {len(self.current_recording_step_images)} images.")
        self.current_recording_step_images = [] # Reset for next step

    def add_image_to_playback_step(self, step_idx, image):
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)): return
        
        crop = self.get_center_crop(image)
        timestamp = int(time.time())
        suffix = np.random.randint(0, 1000)
        filename = f"landmark_{timestamp}_{suffix}.png"
        path = os.path.join(LANDMARK_DIR, filename)
        cv2.imwrite(path, crop)
        name = f"lm_{timestamp}_{suffix}"
        
        new_img = {"name": name, "filename": filename}
        self.landmarks[step_idx]['images'].append(new_img)
        self.vision.load_template(name, path)
        
        print(f"[PLAYBACK] Added new image to step {step_idx}.")
        self.update_db()

    def reset_seek_state(self):
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1
        self.controller.move_camera(0.0, 0.0)

    def execute_seek(self):
        current_time = time.time()
        if not self.seek_active:
            print("[SEEK] Lost landmark. Starting scan (pan right + alternating tilt).")
            self.seek_active = True
            self.seek_start_time = current_time
            self.seek_last_toggle_time = current_time
            self.seek_vertical_direction = -1  # Start tilting up (negative Y)
        
        if current_time - self.seek_start_time > 300.0:
            print("[SEEK] FAILED. Timeout after 5 minutes. Stopping playback.")
            self.stop_all()
            return
        
        if current_time - self.seek_last_toggle_time >= 4.5:
            self.seek_vertical_direction *= -1 
            direction_name = "down" if self.seek_vertical_direction > 0 else "up"
            print(f"[SEEK] Toggling tilt direction: {direction_name}")
            self.seek_last_toggle_time = current_time
        
        cam_x = self.seek_pan_speed 
        cam_y = self.seek_tilt_speed * self.seek_vertical_direction
        
        elapsed = int(current_time - self.seek_start_time)
        if elapsed % 3 == 0 and not hasattr(self, '_last_seek_debug') or getattr(self, '_last_seek_debug', -1) != elapsed:
            self._last_seek_debug = elapsed
            # print(f"[SEEK DEBUG] Sending camera: x={cam_x:.2f}, y={cam_y:.2f}")
        
        self.controller.move_camera(cam_x, cam_y)

    def find_best_match_in_step(self, step_idx, image, threshold=0.87, preferred_img_idx=-1):
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)): return None, -1
        
        step = self.landmarks[step_idx]
        best_match = None
        best_conf = -1
        best_img_idx = -1
        
        # Create a list of indices to check, prioritizing the preferred one
        indices = list(range(len(step['images'])))
        if preferred_img_idx != -1 and preferred_img_idx < len(indices):
            indices.remove(preferred_img_idx)
            indices.insert(0, preferred_img_idx)
        
        for idx in indices:
            img = step['images'][idx]
            match = self.vision.find_template(img['name'], image, threshold=threshold)
            if match:
                _, _, conf = match
                
                # Optimization: Early exit if we find a very high confidence match
                # This prevents checking every single image if we already found a great one.
                if conf > 0.95:
                    return match, idx

                if conf > best_conf:
                    best_conf = conf
                    best_match = match
                    best_img_idx = idx
        
        return best_match, best_img_idx

    def handle_playback(self, image):
        if not self.landmarks: return

        # 1. Find best match for CURRENT step
        # Pass the last matched image index as preferred to speed up tracking
        preferred_idx = getattr(self, 'current_match_img_index', -1)
        match, img_idx = self.find_best_match_in_step(self.current_landmark_idx, image, threshold=0.87, preferred_img_idx=preferred_idx)
        
        # Save index for potential deletion and optimization next frame
        if match:
            self.current_match_img_index = img_idx
        
        # 2. Check NEXT steps (lookahead)
        max_lookahead = len(self.landmarks)
        for _ in range(max_lookahead):
            next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
            next_match, next_img_idx = self.find_best_match_in_step(next_idx, image, threshold=0.87)
            
            if next_match:
                print(f"[PLAYBACK] Found next step {next_idx}. Skipping ahead.")
                self.current_landmark_idx = next_idx
                match = next_match
                self.current_match_img_index = next_img_idx
                self.search_start_time = time.time()
                self.reset_seek_state() 
                if self.current_route_id:
                    set_active_route(self.current_route_id, self.current_landmark_idx)
            else:
                break

        # Debug View Info
        if self.current_landmark_idx < len(self.landmarks):
            current_step = self.landmarks[self.current_landmark_idx]
            target_name = current_step['name']
            
            # For display, pick the image that matched, or the first one if none matched
            disp_img_idx = self.current_match_img_index if match and hasattr(self, 'current_match_img_index') else 0
            if current_step['images']:
                 # Safety check for index
                 if disp_img_idx >= len(current_step['images']): disp_img_idx = 0
                 target_filename = current_step['images'][disp_img_idx]['filename']
            else:
                 target_filename = None
        else:
            target_name = "End of Route"
            target_filename = None

        # Next target info
        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
        if next_idx < len(self.landmarks):
             next_step = self.landmarks[next_idx]
             next_filename = next_step['images'][0]['filename'] if next_step['images'] else None
        else:
             next_filename = None
        
        self._show_debug_view(
            image, match, target_name, 
            target_filename=target_filename,
            next_target_filename=next_filename
        )

        if match:
            self.reset_seek_state()
            self.last_seen_time = time.time()
            self.search_start_time = time.time()
            
            mx, my, _ = match
            mx += 75; my += 75 
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            
            dx = mx - cx
            dy = my - cy
            deadzone = 50
            
            max_offset_x = w // 4
            max_offset_y = h // 3
            max_speed = 0.35
            
            if abs(dx) > deadzone:
                proportion = min(1.0, abs(dx) / max_offset_x)
                cam_x = (-proportion if dx < 0 else proportion) * max_speed
            else:
                cam_x = 0.0
                
            if abs(dy) > deadzone:
                proportion = min(1.0, abs(dy) / max_offset_y)
                cam_y = (-proportion if dy < 0 else proportion) * max_speed
            else:
                cam_y = 0.0
            
            self.controller.move_camera(cam_x, cam_y)
            
            if abs(dx) < 200: self.controller.move_character(0.0, 1.0)
            else: self.controller.move_character(0.0, 0.0)
                
        else:
            if time.time() - self.last_seen_time < 2.0:
                 self.controller.move_character(0.0, 1.0)
            else:
                 self.controller.move_character(0.0, 0.0)

            if time.time() - self.search_start_time > 3.0:
                self.execute_seek()
            else:
                self.controller.move_camera(0.0, 0.0)

    def _show_debug_view(self, image, match, target_name: str, target_filename: str = None, next_target_filename: str = None):
        """
        Displays a debug window showing the captured image with landmark detection overlay.
        """
        scale = 0.5
        h, w = image.shape[:2]
        debug_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        dh, dw = debug_img.shape[:2]
        
        cx, cy = dw // 2, dh // 2
        cv2.line(debug_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(debug_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        status = "TRACKING" if match else ("SEEKING" if self.seek_active else "SEARCHING")
        color = (0, 255, 0) if match else (0, 165, 255)
        
        if match:
            mx, my, conf = match
            bx, by = int(mx * scale), int(my * scale)
            bs = int(150 * scale)
            cv2.rectangle(debug_img, (bx, by), (bx + bs, by + bs), (0, 255, 0), 2)
            lcx, lcy = bx + bs // 2, by + bs // 2
            cv2.line(debug_img, (lcx, lcy), (cx, cy), (0, 255, 0), 1)
            cv2.putText(debug_img, f"{conf:.2f}", (bx, by - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(debug_img, f"[{status}]", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_img, f"Target: {target_name}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Step {self.current_landmark_idx + 1}/{len(self.landmarks)}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        final_img = debug_img
        
        if target_filename:
            path = os.path.join(LANDMARK_DIR, target_filename)
            if os.path.exists(path):
                target_img = cv2.imread(path)
                
                next_target_img = None
                if next_target_filename:
                    next_path = os.path.join(LANDMARK_DIR, next_target_filename)
                    if os.path.exists(next_path):
                        next_target_img = cv2.imread(next_path)
                
                if target_img is not None:
                    t_scale = 1.5
                    th, tw = target_img.shape[:2]
                    new_th, new_tw = int(th * t_scale), int(tw * t_scale)
                    target_img = cv2.resize(target_img, (new_tw, new_th))
                    
                    if next_target_img is not None:
                        # Resize next target to match current target width if possible, or just scale similarly
                        # Just reuse the same scale logic for simplicity
                        nth, ntw = next_target_img.shape[:2]
                        new_nth, new_ntw = int(nth * t_scale), int(ntw * t_scale)
                        next_target_img = cv2.resize(next_target_img, (new_ntw, new_nth))
                    
                    padding = 20
                    # Calculate height needed for next target
                    next_h = 0
                    if next_target_img is not None:
                        next_h = 40 + next_target_img.shape[0]

                    targets_h = 40 + new_th + next_h + 20
                    canvas_h = max(dh, targets_h)
                    # Width: max of (debug_img + padding + current_target, debug_img + padding + next_target)
                    max_target_w = new_tw
                    if next_target_img is not None:
                        max_target_w = max(max_target_w, next_target_img.shape[1])
                        
                    canvas_w = dw + max_target_w + padding * 2
                    
                    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                    canvas[0:dh, 0:dw] = debug_img
                    
                    tx = dw + padding
                    ty = 40
                    
                    cv2.putText(canvas, "CURRENT TARGET:", (tx, ty - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                               
                    if ty + new_th <= canvas_h:
                        canvas[ty:ty+new_th, tx:tx+new_tw] = target_img
                        
                    if next_target_img is not None:
                        ty_next = ty + new_th + 40
                        cv2.putText(canvas, "NEXT TARGET:", (tx, ty_next - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        nth, ntw = next_target_img.shape[:2]
                        if ty_next + nth <= canvas_h:
                            canvas[ty_next:ty_next+nth, tx:tx+ntw] = next_target_img
                        
                    final_img = canvas
            else:
                print(f"[DEBUG] Target image not found at path: {path}")
        else:
             # If target_filename is None (e.g. End of Route), we still might want to show next_target if valid?
             # But logic says target_filename is None if index out of bounds.
             pass

        win = "FFXIII Bot - Debug"
        cv2.imshow(win, final_img)
        if not hasattr(self, '_debug_win_pos'):
            cv2.moveWindow(win, 1930, 50)
            self._debug_win_pos = True
        cv2.waitKey(1)
