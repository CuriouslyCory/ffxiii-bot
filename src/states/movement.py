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
    MovementState handles navigation in the open world using visual landmarks or keylogs.
    """
    
    def __init__(self, manager):
        super().__init__(manager)
        self.mode = "IDLE"
        self.landmarks = [] 
        self.keylog_events = [] # For storing loaded events
        self.route_type = "LANDMARK" # "LANDMARK" or "KEYLOG"
        
        self.current_landmark_idx = 0 # Step index for landmark routes
        self.listening = False
        self.current_route_id = None
        self.blocking_input = False
        
        # Recording state
        self.current_recording_step_images = [] # List of images for the current step being recorded
        self.recording_events = [] # List of key events for keylog recording
        self.recording_start_time = 0
        self.recording_pause_start = 0
        self.total_pause_duration = 0
        
        # Playback/Seek state
        self.last_seen_time = 0
        self.search_start_time = 0
        self.is_seeking = False
        
        # Keylog Playback State
        self.replay_events = []
        self.replay_start_time = 0
        self.replay_pause_time = 0
        self.replay_resume_scheduled_time = 0
        self.active_replay_keys = set()
        self.replay_event_idx = 0
        
        self.held_camera_keys = set() # Track camera keys (2,4,6,8) during playback
        
        # Seek Logic State (simplified)
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1  # 1 = down, -1 = up
        self.seek_pan_speed = 0.25  # Right stick X value for panning
        self.seek_tilt_speed = 0.15  # Right stick Y value for tilting
        
        # Input flags
        self.req_record_type_select = False
        self.req_play_type_select = False
        
        self.req_start_rec_landmark = False
        self.req_start_rec_keylog = False
        
        self.req_list_routes_type = None # "LANDMARK" or "KEYLOG"
        
        self.req_capture = False # 't' key
        self.req_next_step = False # 'n' key
        self.req_retake = False # 'g' key
        self.req_save_exit = False
        self.req_playback = False # Resume 'u'
        self.req_manage_routes = False
        
        self.req_select_route = None # Stores integer key (1-9)
        self.req_stop = False # Escape key to stop
        self.req_playback_action = None # For '2', '3' actions during playback
        self.req_manage_action = None # For manage mode actions
        
        # Management State
        self.manage_step_idx = 0
        self.manage_img_idx = 0
        self.manage_deletion_marks = set() # Set of (step_idx, img_idx) tuples
        
        # Tracking optimization
        self.last_match_pos = None
        
        # Ensure landmark directory exists
        os.makedirs(LANDMARK_DIR, exist_ok=True)
        init_db()
        
        # Setup keyboard listener (non-blocking)
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        # Check for auto-resume
        self.check_auto_resume()

    def check_auto_resume(self):
        active = get_active_route()
        if active:
            print(f"[RESUME] Found active route {active['route_id']} at index {active['current_idx']}")
            route_data = load_route(active['route_id'])
            if route_data:
                self.load_route_data(route_data)
                self.current_route_id = active['route_id']
                
                if self.route_type == "LANDMARK":
                    self.current_landmark_idx = int(active['current_idx'])
                    self.mode = "PLAYBACK"
                    self.is_seeking = True
                    self.search_start_time = time.time()
                elif self.route_type == "KEYLOG":
                    # For keylog, current_idx is likely the time offset or event index?
                    # Let's assume it's the time offset for simplicity of resume.
                    # Or simpler: store the last executed event index.
                    # Let's say current_idx is event index.
                    self.replay_event_idx = int(active['current_idx'])
                    self.mode = "PLAYBACK"
                    # We need to handle the 3s delay resume logic here too?
                    # Yes, logic in execute/handle_playback should handle "resume after pause".
                    self.replay_pause_time = time.time() 
                    self.replay_resume_scheduled_time = time.time() + 3.0
                    print("[RESUME] Keylog playback will resume in 3 seconds...")

    def load_route_data(self, route_data):
        self.route_type = route_data['type']
        self.current_route_id = route_data['id']
        
        if self.route_type == "LANDMARK":
            self.landmarks = route_data['landmarks']
            # Load templates for all images in all steps
            for step in self.landmarks:
                for img in step['images']:
                    path = os.path.join(LANDMARK_DIR, img['filename'])
                    if os.path.exists(path):
                        self.vision.load_template(img['name'], path)
            print(f"Loaded LANDMARK route '{route_data['name']}' with {len(self.landmarks)} steps.")
        
        elif self.route_type == "KEYLOG":
            self.keylog_events = route_data['events']
            self.replay_events = self.keylog_events
            print(f"Loaded KEYLOG route '{route_data['name']}' with {len(self.keylog_events)} events.")

    def is_active(self, image) -> bool:
        roi = (960, 0, 960, 540)
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=roi)
        return match is not None

    def on_enter(self):
        self.listening = True
        print(f"\n--- Movement State ({self.mode}) ---")
        
        if self.mode == "IDLE":
            print("Controls:")
            print("  'r': Record New Route")
            print("  'p': Playback Route")
            print("  'm': Manage Route Images")
            print("  'u': Resume Current Loaded Route")
            print("  'ESC': Stop Playback/Recording")
        
        elif self.mode == "RECORDING":
            if self.recording_pause_start > 0:
                duration = time.time() - self.recording_pause_start
                self.total_pause_duration += duration
                self.recording_pause_start = 0
                print(f"  [RESUMED] Recording resumed (Paused for {duration:.1f}s).")
            else:
                print("  [RESUMED] Recording in progress.")
                
        elif self.mode == "PLAYBACK":
            if self.route_type == "KEYLOG":
                if self.replay_pause_time > 0:
                     # Schedule resume
                     self.replay_resume_scheduled_time = time.time() + 3.0
                     print(f"  [RESUMED] Keylog playback resumed. Starting in 3 seconds...")
            else:
                print("  [RESUMED] Playback in progress.")

    def on_exit(self):
        self.listening = False
        self.controller.move_character(0.0, 0.0)
        self.controller.move_camera(0.0, 0.0) # Stop camera movement explicit
        self.controller.release_all() # Safety release all
        self.held_camera_keys.clear() # Clear held camera keys
        
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        
        if self.mode == "PLAYBACK" and self.current_route_id:
             idx = self.current_landmark_idx if self.route_type == "LANDMARK" else self.replay_event_idx
             set_active_route(self.current_route_id, idx)
             
             if self.route_type == "KEYLOG":
                 self.replay_pause_time = time.time()
                 print("[PAUSED] Keylog playback paused (entered another state).")
             else:
                 print("[PAUSED] Playback paused (entered another state).")
                 
        elif self.mode == "RECORDING":
            if self.route_type == "KEYLOG":
                self.recording_pause_start = time.time()
                print("[PAUSED] Keylog recording paused (entered another state).")
            else:
                print("[PAUSED] Recording paused (entered another state).")

    def on_press(self, key):
        if not self.listening or self.blocking_input: return
        try:
            k_char = None
            if hasattr(key, 'char') and key.char:
                k_char = key.char.lower()
            elif hasattr(key, 'vk') and key.vk in [87, 88, 89, 83, 84, 85, 79, 80, 81, 90]:
                pass

            if key == keyboard.Key.esc:
                self.req_stop = True
                return
            
            # Keylog Recording - Capture WASD and Camera (2,4,6,8)
            if self.mode == "RECORDING" and self.route_type == "KEYLOG":
                if k_char and k_char in ['w', 'a', 's', 'd', '2', '4', '6', '8']:
                    elapsed = time.time() - self.recording_start_time - self.total_pause_duration
                    self.recording_events.append({
                        "time_offset": elapsed,
                        "event_type": "down",
                        "key": k_char
                    })
                # Allow 'y' to finish recording
                if k_char == 'y': self.req_save_exit = True
                return 

            if k_char:
                k = k_char
                
                if self.mode == "IDLE":
                    if k == 'r': self.req_record_type_select = True
                    elif k == 'p': self.req_play_type_select = True
                    elif k == 'm': self.req_manage_routes = True
                    elif k == 'u': self.req_playback = True
                
                elif self.mode == "SELECT_REC_TYPE":
                    if k == '1': self.req_start_rec_landmark = True
                    elif k == '2': self.req_start_rec_keylog = True
                    
                elif self.mode == "SELECT_PLAY_TYPE":
                    if k == '1': self.req_list_routes_type = "LANDMARK"
                    elif k == '2': self.req_list_routes_type = "KEYLOG"
                
                elif self.mode.startswith("SELECTING_"):
                    if k in [str(i) for i in range(1, 10)]: 
                        self.req_select_route = int(k)
                        
                elif self.mode == "RECORDING" and self.route_type == "LANDMARK":
                    if k == 't': self.req_capture = True
                    elif k == 'n': self.req_next_step = True
                    elif k == 'g': self.req_retake = True
                    elif k == 'y': self.req_save_exit = True
                    
                elif self.mode == "PLAYBACK":
                    if self.route_type == "LANDMARK":
                        if k == 'p': self.req_play_type_select = True 
                        elif k == '2': self.req_playback_action = 'delete_current_image'
                        elif k == '3': self.req_playback_action = 'delete_next_image'
                        elif k == 't': self.req_capture = True
                        elif k == 'n': self.req_next_step = True

                elif self.mode == "MANAGING":
                    if k == '[': self.req_manage_action = 'prev_step'
                    elif k == ']': self.req_manage_action = 'next_step'
                    elif k == ',': self.req_manage_action = 'prev_img'
                    elif k == '.': self.req_manage_action = 'next_img'
                    elif k == 'x': self.req_manage_action = 'toggle_delete'
                    elif k == 's': self.req_manage_action = 'save'
                
        except AttributeError:
            pass
    
    def on_release(self, key):
        if not self.listening or self.blocking_input: return
        
        if self.mode == "RECORDING" and self.route_type == "KEYLOG":
             k_char = None
             if hasattr(key, 'char') and key.char:
                k_char = key.char.lower()
             
             if k_char and k_char in ['w', 'a', 's', 'd', '2', '4', '6', '8']:
                elapsed = time.time() - self.recording_start_time - self.total_pause_duration
                self.recording_events.append({
                    "time_offset": elapsed,
                    "event_type": "up",
                    "key": k_char
                })

    def execute(self, image):
        if self.req_stop:
            self.req_stop = False
            self.stop_all()
        
        # Menu Transitions
        if self.req_record_type_select:
            self.req_record_type_select = False
            self.mode = "SELECT_REC_TYPE"
            print("\nSelect Recording Type:")
            print("  1. Landmark Routing")
            print("  2. Keylog Routing")
            print("  ESC. Cancel")
            
        if self.req_play_type_select:
            self.req_play_type_select = False
            self.mode = "SELECT_PLAY_TYPE"
            print("\nSelect Playback Type:")
            print("  1. Landmark Routing")
            print("  2. Keylog Routing")
            print("  ESC. Cancel")
            
        if self.req_start_rec_landmark:
            self.req_start_rec_landmark = False
            self.start_recording("LANDMARK")
        
        if self.req_start_rec_keylog:
            self.req_start_rec_keylog = False
            self.start_recording("KEYLOG")
            
        if self.req_list_routes_type:
            rtype = self.req_list_routes_type
            self.req_list_routes_type = None
            self.list_available_routes(reason="PLAYBACK", route_type=rtype)

        if self.req_manage_routes:
            self.req_manage_routes = False
            self.list_available_routes(reason="MANAGE", route_type="LANDMARK") # Manage only makes sense for landmarks currently
            
        if self.req_select_route is not None:
             idx = self.req_select_route
             self.req_select_route = None
             if self.mode.startswith("SELECTING"):
                 self.select_route(idx)
        
        # Actions
        if self.req_playback:
            self.req_playback = False
            self.start_playback()
            
        if self.req_save_exit and self.mode == "RECORDING":
            self.req_save_exit = False
            self.finish_recording()

        # Mode Specific Logic
        if self.mode == "MANAGING":
            self.show_management_interface(image)
            if self.req_manage_action:
                action = self.req_manage_action
                self.req_manage_action = None
                self.handle_management_action(action)

        elif self.mode == "RECORDING":
            if self.route_type == "LANDMARK":
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
            elif self.route_type == "KEYLOG":
                # Maybe show a small indicator?
                pass
                
        elif self.mode == "PLAYBACK":
            if self.route_type == "LANDMARK":
                if self.req_playback_action:
                    action = self.req_playback_action
                    self.req_playback_action = None
                    if action == 'delete_current_image':
                        self.delete_image_from_step(self.current_landmark_idx)
                    elif action == 'delete_next_image':
                        if self.landmarks:
                            next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                            self.delete_image_from_step(next_idx)

                if self.req_capture:
                    self.req_capture = False
                    self.add_image_to_playback_step(self.current_landmark_idx, image)
                elif self.req_next_step:
                    self.req_next_step = False
                    if self.landmarks:
                        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                        self.add_image_to_playback_step(next_idx, image)

                self.handle_playback(image)
            
            elif self.route_type == "KEYLOG":
                self.handle_keylog_playback()

    def stop_all(self):
        print("[STOP] Stopping all activities. Returning to IDLE.")
        self.mode = "IDLE"
        self.controller.release_all()
        self.held_camera_keys.clear()
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        clear_active_route()

    def start_recording(self, route_type="LANDMARK"):
        self.mode = "RECORDING"
        self.route_type = route_type
        
        if route_type == "LANDMARK":
            self.landmarks = []
            self.current_recording_step_images = []
            print("\n[RECORDING LANDMARK] Started.")
            print("  't': Capture image for current step.")
            print("  'n': Finish step and move to next.")
            print("  'g': Undo last captured image.")
            print("  'y': Finish recording.")
            
        elif route_type == "KEYLOG":
            self.recording_events = []
            self.recording_start_time = time.time()
            self.total_pause_duration = 0
            self.recording_pause_start = 0
            print("\n[RECORDING KEYLOG] Started.")
            print("  Move using W, A, S, D.")
            print("  Camera using 2, 4, 6, 8.")
            print("  'y': Finish recording.")

    def finish_recording(self):
        if self.route_type == "LANDMARK":
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
             
             data = self.landmarks if self.route_type == "LANDMARK" else self.recording_events
             
             if save_route(name, data, self.route_type):
                 print(f"[SAVED] Route '{name}' saved to database.")
             else:
                 print("[ERROR] Failed to save route.")
        except Exception as e:
            print(f"[ERROR] Input error: {e}")
        finally:
            self.blocking_input = False
        self.mode = "IDLE"

    def list_available_routes(self, reason="PLAYBACK", route_type=None):
        self.selection_reason = reason
        self.mode = f"SELECTING_{reason}"
        routes = list_routes(route_type)
        self.available_routes = routes 
        type_str = f" ({route_type})" if route_type else ""
        print(f"\n--- Available Routes{type_str} ---")
        for i, r in enumerate(routes):
            # r: (id, name, created_at, type)
            r_type = r[3] if len(r) > 3 else "Unknown"
            print(f" {i+1}. {r[1]} [{r_type}] (Created: {r[2]})")
        print("Press number key (1-9) to select a route to load.")

    def select_route(self, index):
        if hasattr(self, 'available_routes') and 0 <= index-1 < len(self.available_routes):
            route_id = self.available_routes[index-1][0]
            route_data = load_route(route_id)
            if route_data:
                self.load_route_data(route_data)
                
                if self.selection_reason == "MANAGE":
                    if self.route_type == "KEYLOG":
                        print("[INFO] Management not supported for Keylog routes yet.")
                        self.mode = "IDLE"
                        return
                        
                    print(f"[SELECTED] Route '{route_data['name']}' loaded for Management.")
                    self.start_management()
                else:
                    print(f"[SELECTED] Route '{route_data['name']}' loaded. Press 'u' to start.")
                    self.mode = "IDLE"
            else:
                print("[ERROR] Failed to load route.")
        else:
            print("[ERROR] Invalid selection.")

    # ... (update_db, start_management, handle_management_action, save_management_changes, show_management_interface, delete_image_from_step, delete_step - UNCHANGED) ...
    # I will paste them unchanged
    
    def update_db(self):
        if self.current_route_id and self.route_type == "LANDMARK":
            if update_route_structure(self.current_route_id, self.landmarks):
                 print("[SAVED] Updated route in database.")
            else:
                 print("[ERROR] Failed to update route in database.")

    def start_management(self):
        self.mode = "MANAGING"
        self.manage_step_idx = 0
        self.manage_img_idx = 0
        self.manage_deletion_marks = set()
        print("\n--- Route Management ---")
        print("  '[' / ']': Prev/Next Step")
        print("  ',' / '.': Prev/Next Image")
        print("  'x': Toggle Deletion Mark")
        print("  's': Save Changes")
        print("  'ESC': Cancel")

    def handle_management_action(self, action):
        if not self.landmarks: return
        
        # Ensure step index is valid
        if self.manage_step_idx >= len(self.landmarks):
            self.manage_step_idx = 0
            
        current_step = self.landmarks[self.manage_step_idx]
        images = current_step['images']
        
        if action == 'prev_step':
            self.manage_step_idx = (self.manage_step_idx - 1) % len(self.landmarks)
            self.manage_img_idx = 0
        elif action == 'next_step':
            self.manage_step_idx = (self.manage_step_idx + 1) % len(self.landmarks)
            self.manage_img_idx = 0
        elif action == 'prev_img':
            if images:
                self.manage_img_idx = (self.manage_img_idx - 1) % len(images)
        elif action == 'next_img':
            if images:
                self.manage_img_idx = (self.manage_img_idx + 1) % len(images)
        elif action == 'toggle_delete':
            if images:
                key = (self.manage_step_idx, self.manage_img_idx)
                if key in self.manage_deletion_marks:
                    self.manage_deletion_marks.remove(key)
                else:
                    self.manage_deletion_marks.add(key)
        elif action == 'save':
            self.save_management_changes()

    def save_management_changes(self):
        if not self.manage_deletion_marks:
            print("[MANAGE] No changes to save.")
            self.mode = "IDLE"
            try: cv2.destroyWindow("Route Manager")
            except: pass
            return

        new_landmarks = []
        for s_idx, step in enumerate(self.landmarks):
            new_images = []
            for i_idx, img in enumerate(step['images']):
                if (s_idx, i_idx) not in self.manage_deletion_marks:
                    new_images.append(img)
            
            if new_images:
                step['images'] = new_images
                new_landmarks.append(step)
            else:
                print(f"[MANAGE] Step {s_idx} became empty and will be removed.")
        
        self.landmarks = new_landmarks
        self.update_db()
        print("[MANAGE] Changes saved.")
        self.mode = "IDLE"
        try: cv2.destroyWindow("Route Manager")
        except: pass

    def show_management_interface(self, image):
        canvas_h, canvas_w = 600, 800
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        if not self.landmarks:
            cv2.putText(canvas, "No landmarks loaded.", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Route Manager", canvas)
            cv2.waitKey(1)
            return
            
        # Safety check for indices
        if self.manage_step_idx >= len(self.landmarks):
            self.manage_step_idx = 0
            
        step = self.landmarks[self.manage_step_idx]
        images = step['images']
        
        # Info Text
        cv2.putText(canvas, f"Route Manager", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Step: {self.manage_step_idx + 1} / {len(self.landmarks)} ({step['name']})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        if not images:
             cv2.putText(canvas, "No images in this step.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
             if self.manage_img_idx >= len(images): self.manage_img_idx = 0
             
             current_img_data = images[self.manage_img_idx]
             filename = current_img_data['filename']
             path = os.path.join(LANDMARK_DIR, filename)
             
             if os.path.exists(path):
                 img = cv2.imread(path)
                 if img is not None:
                     scale = 2.0
                     h, w = img.shape[:2]
                     new_h, new_w = int(h * scale), int(w * scale)
                     resized = cv2.resize(img, (new_w, new_h))
                     
                     y_pos = 120
                     x_pos = (canvas_w - new_w) // 2
                     
                     if y_pos + new_h <= canvas_h and x_pos + new_w <= canvas_w:
                        canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = resized
                     
                     color = (255, 255, 255)
                     thickness = 2
                     
                     is_marked = (self.manage_step_idx, self.manage_img_idx) in self.manage_deletion_marks
                     
                     if is_marked:
                         color = (0, 0, 255)
                         thickness = 5
                         cv2.line(canvas, (x_pos, y_pos), (x_pos + new_w, y_pos + new_h), (0, 0, 255), 5)
                         cv2.line(canvas, (x_pos + new_w, y_pos), (x_pos, y_pos + new_h), (0, 0, 255), 5)
                         cv2.putText(canvas, "MARKED FOR DELETION", (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                     
                     cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + new_w, y_pos + new_h), color, thickness)
                     
                     cv2.putText(canvas, f"Image {self.manage_img_idx + 1} / {len(images)}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                     cv2.putText(canvas, f"File: {filename}", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                 else:
                     cv2.putText(canvas, "Error reading image file.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
             else:
                 cv2.putText(canvas, "Image file not found!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.putText(canvas, "[ / ]: Step | , / . : Image | x: Toggle Delete | s: Save | ESC: Cancel", (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Route Manager", canvas)
        cv2.waitKey(1)

    def delete_image_from_step(self, step_idx):
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)): return
        
        step = self.landmarks[step_idx]
        images = step['images']
        
        if not images:
            self.delete_step(step_idx)
            return

        target_img_idx = -1
        
        if step_idx == self.current_landmark_idx and hasattr(self, 'current_match_img_index'):
             target_img_idx = self.current_match_img_index
        elif hasattr(self, 'next_match_img_index') and step_idx == (self.current_landmark_idx + 1) % len(self.landmarks):
             target_img_idx = self.next_match_img_index
        
        if target_img_idx != -1 and 0 <= target_img_idx < len(images):
            deleted = images.pop(target_img_idx)
            print(f"[DELETE] Removed image '{deleted['name']}' from step {step_idx}.")
        else:
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
        if self.route_type == "LANDMARK":
            if not self.landmarks:
                print("[ERROR] No LANDMARK route loaded. Press 'p' to select a route.")
                return
            self.mode = "PLAYBACK"
            if self.current_route_id:
                 set_active_route(self.current_route_id, self.current_landmark_idx)
            self.is_seeking = True
            self.search_start_time = time.time()
            self.seek_state = "IDLE" 
            print(f"\n[PLAYBACK] Starting/Resuming at step {self.current_landmark_idx}.")
            
        elif self.route_type == "KEYLOG":
            if not self.keylog_events:
                print("[ERROR] No KEYLOG route loaded. Press 'p' to select a route.")
                return
            self.mode = "PLAYBACK"
            self.replay_event_idx = 0
            
            # If resuming, logic handles it. But here we start fresh or resume?
            # User said "Resume ... from where it left off".
            # The 'u' key calls start_playback. If we just loaded, idx is 0.
            # If we auto-resumed, idx is restored.
            # We need to set the start_time such that elapsed time matches event offsets.
            # elapsed = time.time() - start_time
            # event.offset = current_offset
            # So start_time = time.time() - current_offset
            
            # Where do we get current_offset? 
            # If idx > 0, we can approximate offset from previous event.
            if self.replay_event_idx > 0 and self.replay_event_idx < len(self.replay_events):
                last_event = self.replay_events[self.replay_event_idx - 1]
                start_offset = last_event['time_offset']
            elif self.replay_event_idx >= len(self.replay_events):
                start_offset = self.replay_events[-1]['time_offset']
            else:
                start_offset = 0
            
            self.replay_start_time = time.time() - start_offset
            self.replay_resume_scheduled_time = 0
            self.held_camera_keys.clear() # Clear held keys on start
            
            if self.current_route_id:
                 set_active_route(self.current_route_id, self.replay_event_idx)
                 
            print(f"\n[PLAYBACK] Starting/Resuming Keylog at event {self.replay_event_idx}.")

    # ... (get_center_crop, show_recording_preview, retake_last_image, capture_image, finish_step, add_image_to_playback_step, reset_seek_state, execute_seek, find_best_match_in_step - UNCHANGED) ...
    # I will paste them.

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
        count = len(self.current_recording_step_images)
        cv2.putText(preview, f"Step Imgs: {count}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Landmark Preview", preview)
        cv2.waitKey(1)

    def retake_last_image(self, image):
        if self.current_recording_step_images:
            removed = self.current_recording_step_images.pop()
            print(f"[RECORDING] Removed image: {removed['name']}")
        else:
            print("[RECORDING] No images in current step to remove.")

    def capture_image(self, image):
        crop = self.get_center_crop(image)
        timestamp = int(time.time())
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
        self.current_recording_step_images = []

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
        self.last_match_pos = None

    def execute_seek(self):
        current_time = time.time()
        if not self.seek_active:
            print("[SEEK] Lost landmark. Starting scan (pan right + alternating tilt).")
            self.seek_active = True
            self.seek_start_time = current_time
            self.seek_last_toggle_time = current_time
            self.seek_vertical_direction = -1
        
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
        self.controller.move_camera(cam_x, cam_y)

    def find_best_match_in_step(self, step_idx, image, threshold=0.87, preferred_img_idx=-1, roi=None):
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)): return None, -1
        
        step = self.landmarks[step_idx]
        best_match = None
        best_conf = -1
        best_img_idx = -1
        
        indices = list(range(len(step['images'])))
        if preferred_img_idx != -1 and preferred_img_idx < len(indices):
            indices.remove(preferred_img_idx)
            indices.insert(0, preferred_img_idx)
        
        for idx in indices:
            img = step['images'][idx]
            match = self.vision.find_template(img['name'], image, threshold=threshold, roi=roi)
            if match:
                _, _, conf = match
                
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
        preferred_idx = getattr(self, 'current_match_img_index', -1)
        
        roi = None
        if self.last_match_pos:
            lx, ly = self.last_match_pos
            rw, rh = 400, 400
            rx = max(0, int(lx - rw / 2 + 75))
            ry = max(0, int(ly - rh / 2 + 75))
            roi = (rx, ry, rw, rh)

        match, img_idx = self.find_best_match_in_step(self.current_landmark_idx, image, threshold=0.87, preferred_img_idx=preferred_idx, roi=roi)
        
        if not match and roi is not None:
            match, img_idx = self.find_best_match_in_step(self.current_landmark_idx, image, threshold=0.87, preferred_img_idx=preferred_idx, roi=None)

        if match:
            self.current_match_img_index = img_idx
            self.last_match_pos = (match[0], match[1])
        else:
            self.last_match_pos = None
        
        # 2. Check NEXT steps
        max_lookahead = len(self.landmarks)
        for _ in range(max_lookahead):
            next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
            next_match, next_img_idx = self.find_best_match_in_step(next_idx, image, threshold=0.87)
            
            if next_match:
                print(f"[PLAYBACK] Found next step {next_idx}. Skipping ahead.")
                self.current_landmark_idx = next_idx
                match = next_match
                self.current_match_img_index = next_img_idx
                self.last_match_pos = (match[0], match[1]) 
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
            
            disp_img_idx = self.current_match_img_index if match and hasattr(self, 'current_match_img_index') else 0
            if current_step['images']:
                 if disp_img_idx >= len(current_step['images']): disp_img_idx = 0
                 target_filename = current_step['images'][disp_img_idx]['filename']
            else:
                 target_filename = None
        else:
            target_name = "End of Route"
            target_filename = None

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

    def _update_camera_from_keys(self):
        cam_x, cam_y = 0.0, 0.0
        # User defined: 2=Down(y=-1), 8=Up(y=1), 4=Left(x=-1), 6=Right(x=1)
        if '4' in self.held_camera_keys: cam_x -= 1.0
        if '6' in self.held_camera_keys: cam_x += 1.0
        if '2' in self.held_camera_keys: cam_y -= 1.0
        if '8' in self.held_camera_keys: cam_y += 1.0
        
        self.controller.move_camera(cam_x, cam_y)

    def handle_keylog_playback(self):
        # Check for resume delay
        if self.replay_resume_scheduled_time > 0:
            remaining = self.replay_resume_scheduled_time - time.time()
            if remaining > 0:
                return # Waiting to resume
            else:
                # Resume time reached
                self.replay_resume_scheduled_time = 0
                
                # Restore timer. 
                pause_duration = time.time() - self.replay_pause_time
                self.replay_start_time += pause_duration
                
                print("[RESUME] Resuming keylog playback events...")
                
                # Restore keys that should be held
                current_offset = time.time() - self.replay_start_time
                keys_to_hold = set()
                self.held_camera_keys.clear()
                
                # Scan events up to current_idx to find held keys
                for i in range(self.replay_event_idx):
                    ev = self.replay_events[i]
                    k = ev['key']
                    
                    if ev['event_type'] == 'down':
                        if k in ['2', '4', '6', '8']:
                            self.held_camera_keys.add(k)
                        else:
                            keys_to_hold.add(k)
                    elif ev['event_type'] == 'up':
                        if k in ['2', '4', '6', '8']:
                            self.held_camera_keys.discard(k)
                        else:
                            keys_to_hold.discard(k)
                
                for k in keys_to_hold:
                    self.controller.press(k)
                    self.active_replay_keys.add(k)
                    print(f"[RESUME] Holding key: {k}")
                
                # Apply initial camera state
                self._update_camera_from_keys()

        current_time = time.time()
        elapsed = current_time - self.replay_start_time
        
        while self.replay_event_idx < len(self.replay_events):
            event = self.replay_events[self.replay_event_idx]
            if event['time_offset'] <= elapsed:
                # Execute event
                k = event['key']
                is_camera_key = k in ['2', '4', '6', '8']
                
                if event['event_type'] == 'down':
                    if is_camera_key:
                        self.held_camera_keys.add(k)
                        self._update_camera_from_keys()
                    else:
                        self.controller.press(k)
                        self.active_replay_keys.add(k)
                        
                elif event['event_type'] == 'up':
                    if is_camera_key:
                        self.held_camera_keys.discard(k)
                        self._update_camera_from_keys()
                    else:
                        self.controller.release(k)
                        self.active_replay_keys.discard(k)
                
                self.replay_event_idx += 1
                
                if self.current_route_id and self.replay_event_idx % 10 == 0:
                    set_active_route(self.current_route_id, self.replay_event_idx)
            else:
                break
        
        if self.replay_event_idx >= len(self.replay_events):
            print("[PLAYBACK] Route finished.")
            self.stop_all()

    def _show_debug_view(self, image, match, target_name: str, target_filename: str = None, next_target_filename: str = None):
        # ... (UNCHANGED) ...
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
                        nth, ntw = next_target_img.shape[:2]
                        new_nth, new_ntw = int(nth * t_scale), int(ntw * t_scale)
                        next_target_img = cv2.resize(next_target_img, (new_ntw, new_nth))
                    
                    padding = 20
                    next_h = 0
                    if next_target_img is not None:
                        next_h = 40 + next_target_img.shape[0]

                    targets_h = 40 + new_th + next_h + 20
                    canvas_h = max(dh, targets_h)
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
                pass
        
        win = "FFXIII Bot - Debug"
        cv2.imshow(win, final_img)
        if not hasattr(self, '_debug_win_pos'):
            cv2.moveWindow(win, 1930, 50)
            self._debug_win_pos = True
        cv2.waitKey(1)
