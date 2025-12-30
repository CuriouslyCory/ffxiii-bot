from src.states.base import State
import time
import os
import json
import cv2
import numpy as np
from pynput import keyboard
from src.db import init_db, save_route, update_route_structure, load_route, list_routes, set_active_route, get_active_route, clear_active_route
from src.states.visual_navigator import VisualNavigator

LANDMARK_DIR = "templates/landmarks"

class MovementState(State):
    """
    MovementState handles navigation in the open world using visual landmarks.
    Now supports Hybrid Visual Odometry Navigation.
    """
    
    def __init__(self, manager):
        super().__init__(manager)
        self.mode = "IDLE"
        self.landmarks = [] # Now a list of steps: [{'name': 'Step_X', 'images': [{'filename': '...', 'name': '...'}]}]
        self.current_landmark_idx = 0 # Step index
        self.listening = False
        self.current_route_id = None
        self.blocking_input = False
        
        self.navigator = VisualNavigator(self.vision)
        self.route_type = "LANDMARK" # "LANDMARK", "KEYLOG", "HYBRID"
        
        # Recording state
        self.current_recording_step_images = [] # List of images for the current step being recorded
        self.recording_nodes = [] # List of Hybrid nodes
        self.last_node_time = 0
        
        # Playback/Seek state
        self.last_seen_time = 0
        self.search_start_time = 0
        self.is_seeking = False
        
        # Seek Logic State (simplified)
        self.seek_active = False
        self.seek_start_time = 0
        self.seek_last_toggle_time = 0
        self.seek_vertical_direction = 1  # 1 = down, -1 = up
        self.seek_pan_speed = 0.15  # Right stick X value for panning
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
        self.route_type = route_data.get('type', 'LANDMARK')
        self.current_route_id = route_data['id']
        
        if self.route_type == "HYBRID":
            self.landmarks = route_data.get('nodes', []) # Using 'landmarks' field to store nodes for now
            # Preload templates if needed, or VisualNavigator handles it?
            # VisualNavigator needs full images to extract features.
            # We should probably load them on demand or preload them.
            # For now, let's preload the minimap images into memory?
            # Or just rely on cv2.imread in real-time loop? (might be slow)
            # Let's preload templates as before.
            pass
        else:
            self.landmarks = route_data['landmarks']
            # Load templates for all images in all steps
            for step in self.landmarks:
                for img in step['images']:
                    path = os.path.join(LANDMARK_DIR, img['filename'])
                    if os.path.exists(path):
                        self.vision.load_template(img['name'], path)
        print(f"Loaded {self.route_type} route '{route_data['name']}' with {len(self.landmarks)} steps.")

    def is_active(self, image) -> bool:
        roi = (960, 0, 960, 540)
        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=roi)
        return match is not None

    def on_enter(self):
        self.listening = True
        print(f"\n--- Movement State ({self.mode}) ---")
        print("Controls:")
        print("  'r': Start New Recording (will prompt for type)")
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
        try: cv2.destroyWindow(self.navigator.debug_window_name)
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
                
                # Hybrid Recording Inputs
                if self.mode == "RECORDING" and self.route_type == "HYBRID":
                    # Track intent? Usually handled by sampling held keys in the loop
                    pass
                
        except AttributeError:
            pass

    def execute(self, image):
        if self.req_stop:
            self.req_stop = False
            self.stop_all()
        if self.req_record_mode:
            self.req_record_mode = False
            self.start_recording_dialog()
            
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
            if self.route_type == "HYBRID":
                self.handle_hybrid_recording(image)
            else:
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
            if self.route_type == "HYBRID":
                self.handle_hybrid_playback(image)
            else:
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
        try: cv2.destroyWindow(self.navigator.debug_window_name)
        except: pass
        clear_active_route()

    def start_recording_dialog(self):
        self.blocking_input = True
        try:
            # Flush stdin to clear the 'r' key press that triggered this
            try:
                import sys, termios
                termios.tcflush(sys.stdin, termios.TCIOFLUSH)
            except:
                pass

            while True:
                print("\nSelect Recording Type:")
                print("  1. Landmark Routing")
                print("  2. Hybrid Visual Odometry")
                print("  q. Cancel")
                
                choice = input("> ").strip().lower()
                
                if choice == '1':
                    self.start_recording("LANDMARK")
                    break
                elif choice == '2':
                    self.start_recording("HYBRID")
                    break
                elif choice == 'q':
                    print("Cancelled.")
                    break
                elif choice == '':
                    # Ignore empty lines (e.g. accidental extra Enters)
                    continue
                else:
                    print(f"Invalid selection: '{choice}'. Please enter 1 or 2.")
        finally:
            self.blocking_input = False

    def start_recording(self, r_type="LANDMARK"):
        self.mode = "RECORDING"
        self.route_type = r_type
        
        if r_type == "HYBRID":
            self.recording_nodes = []
            self.last_node_time = time.time()
            print("\n[RECORDING HYBRID] Started.")
            print("  Move naturally. Breadcrumbs are collected automatically.")
            print("  'y': Finish recording.")
        else:
            self.landmarks = []
            self.current_recording_step_images = []
            print("\n[RECORDING LANDMARK] Started.")
            print("  't': Capture image for current step.")
            print("  'n': Finish step and move to next.")
            print("  'g': Undo last captured image.")
            print("  'y': Finish recording.")

    def finish_recording(self):
        if self.route_type == "LANDMARK":
            if self.current_recording_step_images:
                self.finish_step()
        
        self.controller.release_all()
        try: cv2.destroyWindow("Landmark Preview") 
        except: pass
        try: cv2.destroyWindow(self.navigator.debug_window_name)
        except: pass
        
        self.blocking_input = True
        print("\n[INPUT REQUIRED] Enter name for this route:")
        try:
             name = input("> ").strip()
             if not name: name = f"Route_{int(time.time())}"
             
             data_to_save = self.recording_nodes if self.route_type == "HYBRID" else self.landmarks
             
             if save_route(name, data_to_save, self.route_type):
                 print(f"[SAVED] Route '{name}' saved to database.")
             else:
                 print("[ERROR] Failed to save route.")
        except Exception as e:
            print(f"[ERROR] Input error: {e}")
        finally:
            self.blocking_input = False
        self.mode = "IDLE"

    # --- Hybrid Logic ---

    def handle_hybrid_recording(self, image):
        current_time = time.time()
        # Sample every 1.0 second
        if current_time - self.last_node_time >= 1.0:
            self.record_hybrid_node(image)
            self.last_node_time = current_time

    def record_hybrid_node(self, image):
        # Save minimap image and main view
        timestamp = int(time.time() * 1000)
        
        # We need a unique folder for this session? Or just flat files with timestamp
        # Ideally, we should create a session ID. For now, flat files.
        mm_filename = f"hybrid_mm_{timestamp}.png"
        mm_path = os.path.join(LANDMARK_DIR, mm_filename)
        
        # Save the UNMASKED minimap ROI so we can feature match against it later
        # Use get_minimap_crop to ensure calibration
        roi_img = self.navigator.get_minimap_crop(image)
        if roi_img is not None:
             cv2.imwrite(mm_path, roi_img)
        else:
             print(f"[REC HYBRID] Failed to extract minimap for node {len(self.recording_nodes)}")
        
        node = {
            "id": len(self.recording_nodes),
            "timestamp": timestamp,
            "minimap_path": mm_filename,
            "main_view_path": None, # Optional for now
            "inputs": [] 
        }
        self.recording_nodes.append(node)
        print(f"[REC HYBRID] Saved Node {node['id']}")

    def handle_hybrid_playback(self, image):
        if not self.landmarks: return
        
        # Target node
        if self.current_landmark_idx >= len(self.landmarks):
            print("[PLAYBACK] Route finished.")
            self.stop_all()
            return

        target_node = self.landmarks[self.current_landmark_idx]
        
        # Load target minimap image
        filename = target_node.get('minimap_path')
        if not filename:
             print("[ERROR] Node missing minimap_path")
             self.stop_all()
             return

        path = os.path.join(LANDMARK_DIR, filename)
        if not os.path.exists(path):
            print(f"[ERROR] Missing file {path}")
            self.stop_all()
            return
            
        # Cache the target minimap to avoid disk I/O every frame
        if not hasattr(self, '_current_target_mm_path') or self._current_target_mm_path != path:
            self._current_target_mm = cv2.imread(path)
            self._current_target_mm_path = path
            
        target_mm = self._current_target_mm
        
        if target_mm is None:
             print(f"[ERROR] Failed to load {path}")
             self.stop_all()
             return
        
        # Calculate Drift
        drift = self.navigator.compute_drift(image, target_mm)
        
        # COASTING & RECOVERY LOGIC:
        # If tracking is lost (drift is None), we attempt to recover by holding the last valid command
        # or slowing down the spin for a few attempts.
        
        if drift is None:
             current_time = time.time()
             time_since_last_seen = current_time - self.last_seen_time
             
             # Coasting duration (hold last input)
             coast_duration = 0.5
             
             # Retrieve last valid controls (or current if none stored)
             last_ctrl = getattr(self, 'last_valid_controls', self.controller.state)
             last_rx = last_ctrl.get('rx', 0.0)
             last_ly = last_ctrl.get('ly', 0.0)
             last_ry = last_ctrl.get('ry', 0.0)
             
             # Extended coast if we were turning
             if abs(last_rx) > 0.3 and abs(last_ly) < 0.2:
                 coast_duration = 2.0 
             
             retry_duration = 1.0 # Duration of each retry attempt
             
             if time_since_last_seen < coast_duration:
                 # Phase 1: Reverse Rotation, Hold Forward
                 # User Request: "Coasting on forward movement is fine, but instead of coasting rotation, 
                 # you should actually reverse towards the last known readable orientation."
                 
                 # Reverse camera pan
                 cam_x = -last_rx
                 cam_y = last_ry
                 
                 self.controller.move_camera(cam_x, cam_y)
                 
                 # Hold character movement (explicitly or implicitly)
                 # Implicitly holding last state is fine, but let's be explicit for clarity if we have last_lx/ly
                 last_lx = last_ctrl.get('lx', 0.0)
                 self.controller.move_character(last_lx, last_ly)
                 
                 self.navigator.show_debug_view(image, target_mm, self.controller.state, tracking_active=False, status_msg="REVERSING")
                 return
                 
             elif time_since_last_seen < coast_duration + retry_duration * 3:
                 # Phase 2: Recovery / Retry (Slower Spin)
                 retry_time = time_since_last_seen - coast_duration
                 attempt = int(retry_time // retry_duration) + 1 # 1, 2, 3
                 
                 # Slow down factor: 0.6, 0.4, 0.2
                 scale = max(0.2, 0.8 - (0.2 * attempt))
                 
                 # Apply scaled controls based on LAST VALID inputs
                 cam_x = last_rx * scale
                 cam_y = last_ry * scale
                 
                 # Stop character movement to avoid walking blindly
                 self.controller.move_character(0.0, 0.0)
                 self.controller.move_camera(cam_x, cam_y)
                 
                 self.navigator.show_debug_view(image, target_mm, self.controller.state, tracking_active=False, status_msg=f"RETRY {attempt}")
                 return
                 
             else:
                 # Phase 3: Give Up
                 if time_since_last_seen < (coast_duration + retry_duration * 3 + 0.5):
                     print(f"[PLAYBACK] Lost visual tracking for {time_since_last_seen:.2f}s. Stopping.")
                     
                 self.controller.move_character(0, 0)
                 self.controller.move_camera(0, 0)
                 self.navigator.show_debug_view(image, target_mm, self.controller.state, tracking_active=False, status_msg="LOST")
                 return

        # Drift is valid
        self.last_seen_time = time.time()
        dx, dy, angle = drift
        
        # Smoothing: Use EMA (Exponential Moving Average)
        # Initialize if not present OR if this is the first valid reading after reset (optional)
        if not hasattr(self, '_smoothed_drift'):
            self._smoothed_drift = {'dx': dx, 'dy': dy, 'angle': angle}
            
        # Increased alpha from 0.3 to 0.6 to reduce lag/overshoot
        # Higher alpha = more responsive, less smooth. Lower alpha = smoother, more lag.
        alpha = 0.6 
        
        self._smoothed_drift['dx'] = alpha * dx + (1 - alpha) * self._smoothed_drift['dx']
        self._smoothed_drift['dy'] = alpha * dy + (1 - alpha) * self._smoothed_drift['dy']
        
        # Angle smoothing with circular mean (vector averaging) to handle +/- 180 wrap-around
        a_rad = np.radians(angle)
        prev_a_rad = np.radians(self._smoothed_drift['angle'])
        
        # EMA on sin/cos components
        s_sin = alpha * np.sin(a_rad) + (1 - alpha) * np.sin(prev_a_rad)
        s_cos = alpha * np.cos(a_rad) + (1 - alpha) * np.cos(prev_a_rad)
        
        # Reconstruct angle
        self._smoothed_drift['angle'] = np.degrees(np.arctan2(s_sin, s_cos))
        
        sdx, sdy, sangle = self._smoothed_drift['dx'], self._smoothed_drift['dy'], self._smoothed_drift['angle']
        
        # PID / Visual Servo Control
        # If we are "close enough", advance to next node
        # Dist threshold: dx, dy small. Angle small.
        # Use RAW values for "reached" check to allow snap completion, 
        # but smoothed values for control to avoid jitter.
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Thresholds need tuning
        # Enforce valid match (drift not None) and check tolerances
        if dist < 30 and abs(angle) < 5:
            # Maybe add a "consecutive frames" check to avoid noise?
            # For now, simple check.
            print(f"[PLAYBACK] Reached Node {self.current_landmark_idx}")
            self.current_landmark_idx += 1
            if self.current_route_id:
                set_active_route(self.current_route_id, self.current_landmark_idx)
            return

        # Apply Controls
        # Objective: Keep angle at 0 and move forward.
        # Character moves in direction of camera (mostly).
        
        # 1. Camera Control (Pan Left/Right to fix Angle)
        # Target angle is 0.
        # If angle is positive (rotated Left?), we need to Pan Left? 
        # Visual Odometry: Angle is rotation from Target -> Current.
        # If Angle > 0, Current is rotated CW relative to Target.
        # So we need to rotate CCW (Left) to fix it.
        # move_camera: x=-1.0 is Left, x=1.0 is Right.
        # So Angle > 0 -> cam_x < 0.
        # Wait, if Current is rotated CW (Right) relative to Target, 
        # we need to rotate Left (CCW) to match Target?
        # Yes.
        # BUT, moving the camera Right (positive X) rotates the view CW (Right).
        # So if Angle (Target -> Current) is positive, Current is CW of Target.
        # To make Current match Target, we need to rotate CCW (Left).
        # Camera X < 0 rotates Left.
        
        # DEBUG: The user says "always turns to the right".
        # This implies our sign logic was actually correct initially (negative feedback), 
        # OR the decompose angle sign is inverted.
        
        # However, let's look at the result. "Always turns to the right" -> cam_x > 0.
        # If I had `cam_x = sangle * kp_rot`, and `sangle` was positive, it would turn right.
        # If `sangle` was negative, it would turn left.
        # If the bot is stuck turning right, maybe sangle is consistently positive?
        # OR maybe the initial logic `cam_x = -sangle` was correct but my understanding of 'turning right' was wrong.
        
        # Let's try INVERTING the control logic again to see if it fixes the "always turns right" issue.
        # If `sangle` > 0 means "Need to turn Right", then `cam_x` should be > 0.
        
        cam_x = 0.0
        kp_rot = 0.03 
        
        # Trying inverted logic from previous step:
        # cam_x = sangle * kp_rot
        
        # Wait, I previously wrote: "New: cam_x = sangle * kp_rot".
        # If I revert to negative feedback: `cam_x = -sangle * kp_rot`.
        # Let's try the POSITIVE feedback loop first, assuming the decompose angle is "Error to be applied".
        
        # Actually, let's look at the debug overlay in your screenshot.
        # "ang: 36.51". The view looks rotated to the LEFT compared to target? 
        # No, the Minimap on Right (Current) has the cone pointing UP-RIGHT. 
        # Minimap on Left (Target) has cone pointing UP.
        # So Current is rotated CW (Right) relative to Target.
        # Angle is +36.51.
        # So Positive Angle = CW Rotation.
        # To fix CW rotation, we need to rotate CCW (Left).
        # Camera Left is X < 0.
        # So if Angle > 0, we want X < 0.
        # So `cam_x = -angle * kp` is correct.
        
        # BUT, user says it "always turns to the right".
        # This means we are sending X > 0.
        # If Angle is +36 (CW), and we send X > 0 (Right/CW), we are making it worse!
        # So `cam_x = angle * kp` (Positive feedback) causes runaway Right turn if angle is positive.
        # Wait, my previous code was `cam_x = sangle * kp_rot` (Positive).
        # Ah, I changed it to positive in the last turn? No, I tried to.
        # The file content shows: `cam_x = sangle * kp_rot` (Line 493).
        # So the code WAS using positive feedback, which explains "always turns right" if angle is positive.
        
        # User requested inversion: "actively steers away from 0" implies current logic is positive feedback.
        # So we switch sign to stabilize.
        cam_x = sangle * kp_rot
        
        # Reduced max speed further to preventing flying past target
        cam_x = max(-0.4, min(0.4, cam_x)) 

        # Min speed boost
        # If absolute value is too small but not zero, boost it to overcome friction?
        if abs(cam_x) < 0.05:
             cam_x = 0.0
        elif abs(cam_x) < 0.2:
             # Boost small values to minimum movement threshold
             cam_x = 0.2 if cam_x > 0 else -0.2
            
        self.controller.move_camera(cam_x, 0)
        
        # 2. Movement Control (Forward/Back + Strafe)
        # Logic: If angle is small (< 10 deg), move forward.
        # Speed scales with alignment (closer to 0 angle -> faster).
        
        move_x = 0.0
        move_y = 0.0
        
        if abs(sangle) < 15.0: # Increased tolerance slightly to allow movement while correcting
            # Forward Speed (y)
            # Max speed 1.0. Scale down as angle increases.
            # 0 deg -> 1.0 speed
            # 30 deg -> 0.0 speed (Relaxed from 15)
            speed_factor = 1.0 - (abs(sangle) / 30.0)
            move_y = 1.0 * speed_factor
            
            # Simple Forward/Back correction based on dy?
            # Actually, the user says "move forward" mainly.
            # But we might need to strafe if dx is large.
            # +dx means we are shifted right? -> Strafe Left?
            # Let's add slight strafe correction if needed.
            kp_strafe = 0.005 # Reduced from 0.01 to reduce "hard strafing"
            move_x = -sdx * kp_strafe
            move_x = max(-0.5, min(0.5, move_x)) # Cap strafe
            
        else:
            # Angle too large, stop moving and just rotate
            move_y = 0.0
            move_x = 0.0
            
        self.controller.move_character(move_x, move_y)
        
        self.navigator.show_debug_view(image, target_mm, self.controller.state, tracking_active=True)
        
        # Store valid controls for coasting/recovery reference
        self.last_valid_controls = self.controller.state.copy()

    # --- Existing Helper Methods ---

    def list_available_routes(self):
        self.mode = "SELECTING"
        routes = list_routes()
        self.available_routes = routes 
        print("\n--- Available Routes ---")
        for i, r in enumerate(routes):
            print(f" {i+1}. {r[1]} [{r[3]}] (Created: {r[2]})")
        print("Press number key (1-9) to select a route to load.")

    def select_route(self, index):
        if hasattr(self, 'available_routes') and 0 <= index-1 < len(self.available_routes):
            route_id = self.available_routes[index-1][0]
            route_data = load_route(route_id)
            if route_data:
                # Check if this route is currently in-progress
                active = get_active_route()
                start_idx = 0
                
                if active and str(active['route_id']) == str(route_id):
                    # It is the active route. Ask user.
                    self.blocking_input = True
                    print(f"\n[RESUME] Route '{route_data['name']}' is in progress at step {active['current_idx']}.")
                    print("  Do you want to Resume from there? (y/n)")
                    try:
                        ans = input("> ").strip().lower()
                        if ans == 'y':
                            start_idx = int(active['current_idx'])
                            print(f"Resuming at step {start_idx}.")
                        else:
                            print("Starting from beginning.")
                            # Clear the active state so it's no longer considered "in progress"
                            clear_active_route()
                    finally:
                        self.blocking_input = False
                
                self.load_route_data(route_data)
                self.current_landmark_idx = start_idx
                
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
        
        self.controller.move_camera(cam_x, cam_y)

    def find_best_match_in_step(self, step_idx, image, threshold=0.87, preferred_img_idx=-1):
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
            match = self.vision.find_template(img['name'], image, threshold=threshold)
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

        preferred_idx = getattr(self, 'current_match_img_index', -1)
        match, img_idx = self.find_best_match_in_step(self.current_landmark_idx, image, threshold=0.87, preferred_img_idx=preferred_idx)
        
        if match:
            self.current_match_img_index = img_idx
        
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

    def _show_debug_view(self, image, match, target_name: str, target_filename: str = None, next_target_filename: str = None):
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
