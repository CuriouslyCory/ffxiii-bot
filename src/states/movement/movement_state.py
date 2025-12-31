"""
Movement state for handling navigation in the open world.

Main orchestrator that delegates to specialized components for recording,
playback, navigation, and visualization.
"""

import time
import os
import sys
import termios
from typing import Optional
import numpy as np
import cv2
from src.states.base import State
from src.states.visual_navigator import VisualNavigator
from .input_handler import InputHandler, Action
from .route_manager import RouteManager
from .route_recorder import RouteRecorder
from .route_player import RoutePlayer, RandomMovementPlayer
from .navigation_controller import NavigationController
from .seek_strategy import SeekStrategy
from .debug_visualizer import DebugVisualizer


class MovementState(State):
    """
    MovementState handles navigation in the open world using visual landmarks.
    
    Supports both LANDMARK and HYBRID visual odometry navigation modes.
    Acts as a thin orchestrator delegating to specialized components.
    """
    
    def __init__(self, manager):
        """Initialize the movement state."""
        super().__init__(manager)
        
        # Core components
        self.navigator = VisualNavigator(self.vision)
        self.input_handler = InputHandler()
        self.route_manager = RouteManager(self.vision)
        self.nav_controller = NavigationController()
        self.seek_strategy = SeekStrategy()
        self.debug_visualizer = DebugVisualizer(self.navigator)
        
        # Recording and playback components
        self.route_recorder = RouteRecorder(self.vision, self.navigator)
        self.route_player = RoutePlayer(
            self.vision, self.controller, self.navigator,
            self.nav_controller, self.seek_strategy
        )
        self.random_player = RandomMovementPlayer(self.controller)
        
        # State
        self.mode = "IDLE"
        self.route_type: Optional[str] = None
        self.landmarks = []
        self.current_landmark_idx = 0
        self.current_route_id: Optional[int] = None
        
        # Check for auto-resume
        self.check_auto_resume()
    
    def check_auto_resume(self):
        """Check for active route and resume if found."""
        active = self.route_manager.get_active_route()
        if active:
            print(f"[RESUME] Found active route {active['route_id']} at step {active['current_idx']}")
            route_data = self.route_manager.load_route(active['route_id'])
            if route_data:
                self.load_route_data(route_data)
                self.current_landmark_idx = active['current_idx']
                self.mode = "PLAYBACK"
                self.current_route_id = active['route_id']
                # Initialize the route player with the correct route type
                self.route_player.start_playback(self.route_type)
    
    def load_route_data(self, route_data: dict):
        """
        Load route data into state.
        
        Args:
            route_data: Route data dictionary from database.
        """
        self.route_type = route_data.get('type', 'LANDMARK')
        self.current_route_id = route_data['id']
        
        if self.route_type == "HYBRID":
            self.landmarks = route_data.get('nodes', [])
        else:
            self.landmarks = route_data.get('landmarks', [])
        
        print(f"Loaded {self.route_type} route '{route_data['name']}' with {len(self.landmarks)} steps.")
    
    def is_active(self, image) -> bool:
        """Check if movement state is active."""
        roi = (960, 0, 960, 540)
        match = self.vision.find_template("minimap_outline", image, threshold=0.25, roi=roi)
        return match is not None
    
    def on_enter(self):
        """Called when entering the state."""
        self.input_handler.set_listening(True)
        print(f"\n--- Movement State ({self.mode}) ---")
        print("Controls:")
        print("  'r': Start New Recording (will prompt for type)")
        print("  'p': Playback Menu (0=Random Movement, 1-9=Select Route)")
        print("  'u': Resume Current Loaded Route")
        print("  'd': Toggle HSV Filter Debug Mode (during playback)")
        print("  '2': Delete Current Image (during playback)")
        print("  '3': Delete Next Image (during playback)")
        print("  '4': Delete Next Node (during playback)")
        print("  'ESC': Stop Playback/Recording/Random Movement")
        
        if self.mode == "RECORDING":
            print("  [RESUMED] Recording in progress.")
        elif self.mode == "PLAYBACK":
            print("  [RESUMED] Playback in progress.")
    
    def on_exit(self):
        """Called when exiting the state."""
        self.input_handler.set_listening(False)
        self.controller.move_character(0.0, 0.0)
        self.controller.move_camera(0.0, 0.0)
        self.controller.release_all()
        self.debug_visualizer.cleanup_windows()
        
        # Disable HSV debug if active
        if self.navigator.hsv_debug_enabled:
            self.navigator.disable_hsv_debug()
        
        if self.mode == "RANDOM_MOVEMENT":
            self.random_player.stop()
            print("[PAUSED] Random movement paused (entered another state).")
        elif self.mode == "PLAYBACK" and self.current_route_id:
            self.route_manager.set_active_route(self.current_route_id, self.current_landmark_idx)
            print("[PAUSED] Playback paused (entered another state).")
        elif self.mode == "RECORDING":
            print("[PAUSED] Recording paused (entered another state).")
    
    def execute(self, image):
        """Main execution loop."""
        # Process input actions
        actions = self.input_handler.get_pending_actions()
        
        for action in actions:
            if action == Action.STOP:
                self.stop_all()
            elif action == Action.RECORD_MODE:
                self.start_recording_dialog()
            elif action == Action.LIST_ROUTES:
                self.list_available_routes()
            elif action == Action.SELECT_ROUTE:
                idx = self.input_handler.get_select_route_value()
                if idx is not None and self.mode == "SELECTING":
                    if idx == 0:
                        self.start_random_movement()
                    else:
                        self.select_route(idx)
            elif action == Action.RANDOM_MOVEMENT:
                self.start_random_movement()
            elif action == Action.DELETE_CURRENT_IMAGE:
                if self.mode == "PLAYBACK":
                    self.delete_image_from_step(self.current_landmark_idx)
            elif action == Action.DELETE_NEXT_IMAGE:
                if self.mode == "PLAYBACK" and self.landmarks:
                    next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                    self.delete_image_from_step(next_idx)
            elif action == Action.DELETE_NEXT_NODE:
                if self.mode == "PLAYBACK" and self.landmarks:
                    next_idx = self.current_landmark_idx + 1
                    if next_idx < len(self.landmarks):
                        self.delete_next_node(next_idx)
                    else:
                        print("[DELETE] No next node to delete (at end of route).")
            elif action == Action.TOGGLE_HSV_DEBUG:
                if self.navigator.hsv_debug_enabled:
                    self.navigator.disable_hsv_debug()
                else:
                    self.navigator.enable_hsv_debug()
            elif action == Action.PLAYBACK:
                self.start_playback()
            elif action == Action.SAVE_EXIT:
                if self.mode == "RECORDING":
                    self.finish_recording()
            elif action == Action.CAPTURE:
                if self.mode == "RECORDING":
                    self.route_recorder.capture_image(image)
                elif self.mode == "PLAYBACK":
                    self.add_image_to_playback_step(self.current_landmark_idx, image)
            elif action == Action.NEXT_STEP:
                if self.mode == "RECORDING":
                    self.route_recorder.finish_step()
                elif self.mode == "PLAYBACK":
                    if self.landmarks:
                        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                        self.add_image_to_playback_step(next_idx, image)
            elif action == Action.RETAKE:
                if self.mode == "RECORDING":
                    self.route_recorder.retake_last_image()
        
        # Execute mode-specific logic
        if self.mode == "RECORDING":
            self.handle_recording(image)
        elif self.mode == "PLAYBACK":
            self.handle_playback(image)
        elif self.mode == "RANDOM_MOVEMENT":
            self.handle_random_movement(image)
    
    def handle_recording(self, image: np.ndarray):
        """Handle recording mode."""
        if self.route_type == "HYBRID":
            self.route_recorder.process_frame(image)
        else:
            # LANDMARK mode
            step_count = self.route_recorder.get_step_count()
            self.debug_visualizer.show_recording_preview(image, step_count)
    
    def handle_random_movement(self, image: np.ndarray):
        """Handle random movement mode."""
        result = self.random_player.process_frame(image)
        
        # Apply controls
        self.controller.move_camera(result['cam_x'], result['cam_y'])
        self.controller.move_character(result['move_x'], result['move_y'])
    
    def handle_playback(self, image: np.ndarray):
        """Handle playback mode."""
        if not self.landmarks:
            return
        
        if self.route_type == "HYBRID":
            result = self.route_player.process_frame(
                self.landmarks, self.current_landmark_idx, image,
                self.current_route_id, self.route_manager.set_active_route
            )
            
            # Update index if advanced
            if 'new_idx' in result:
                self.current_landmark_idx = result['new_idx']
            elif 'current_idx' in result:
                self.current_landmark_idx = result['current_idx']
            
            # Apply controls
            self.controller.move_camera(result['cam_x'], result['cam_y'])
            self.controller.move_character(result['move_x'], result['move_y'])
            
            # Show debug
            self.debug_visualizer.show_odometry_debug(
                image, result.get('target_mm'),
                self.controller.state,
                tracking_active=(result.get('status') == 'TRACKING'),
                status_msg=result.get('status', '')
            )
            
            if self.navigator.hsv_debug_enabled:
                self.debug_visualizer.show_hsv_debug(image)
        else:
            # LANDMARK mode
            result = self.route_player.process_frame(
                self.landmarks, self.current_landmark_idx, image
            )
            
            # Update index if lookahead advanced
            if 'current_idx' in result:
                self.current_landmark_idx = result['current_idx']
                if self.current_route_id:
                    self.route_manager.set_active_route(
                        self.current_route_id, self.current_landmark_idx
                    )
            
            # Apply controls
            self.controller.move_camera(result['cam_x'], result['cam_y'])
            self.controller.move_character(result['move_x'], result['move_y'])
            
            # Get next target filename for debug view
            next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
            next_filename = None
            if next_idx < len(self.landmarks):
                next_step = self.landmarks[next_idx]
                next_filename = next_step['images'][0]['filename'] if next_step.get('images') else None
            
            # Show debug
            self.debug_visualizer.show_playback_debug(
                image, result.get('match'),
                result.get('target_name', 'Unknown'),
                self.current_landmark_idx, len(self.landmarks),
                result.get('target_filename'),
                next_filename,
                self.route_player.landmark_player.seek_active
            )
    
    def stop_all(self):
        """Stop all activities and return to IDLE."""
        print("[STOP] Stopping all activities. Returning to IDLE.")
        if self.mode == "RANDOM_MOVEMENT":
            self.random_player.stop()
        self.mode = "IDLE"
        self.controller.release_all()
        self.debug_visualizer.cleanup_windows()
        self.route_manager.clear_active_route()
    
    def start_recording_dialog(self):
        """Start recording with type selection dialog."""
        self.input_handler.set_blocking(True)
        try:
            # Flush stdin
            try:
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
                    continue
                else:
                    print(f"Invalid selection: '{choice}'. Please enter 1 or 2.")
        finally:
            self.input_handler.set_blocking(False)
    
    def start_recording(self, r_type: str):
        """Start recording a route."""
        self.mode = "RECORDING"
        self.route_type = r_type
        self.route_recorder.start_recording(r_type)
    
    def finish_recording(self):
        """Finish recording and save route."""
        data, master_map_path = self.route_recorder.finish_recording()
        
        self.controller.release_all()
        self.debug_visualizer.cleanup_windows()
        
        self.input_handler.set_blocking(True)
        print("\n[INPUT REQUIRED] Enter name for this route:")
        try:
            name = input("> ").strip()
            if not name:
                name = f"Route_{int(time.time())}"
            
            if self.route_manager.save_route(name, data, self.route_type, master_map_path):
                print(f"[SAVED] Route '{name}' saved to database.")
            else:
                print("[ERROR] Failed to save route.")
        except Exception as e:
            print(f"[ERROR] Input error: {e}")
        finally:
            self.input_handler.set_blocking(False)
        
        self.mode = "IDLE"
    
    def list_available_routes(self):
        """List available routes for selection."""
        self.mode = "SELECTING"
        routes = self.route_manager.list_available_routes()
        print("\n--- Playback Menu ---")
        print(" 0. Random Movement Mode")
        print("\n--- Available Routes ---")
        for i, r in enumerate(routes):
            print(f" {i+1}. {r[1]} [{r[3]}] (Created: {r[2]})")
        print("Press number key (0-9) to select an option.")
    
    def select_route(self, index: int):
        """Select a route by index."""
        result = self.route_manager.select_route(index, self.current_route_id)
        if not result:
            print("[ERROR] Invalid selection.")
            return
        
        route_data = result['route_data']
        start_idx = result['start_idx']
        route_id = result['route_id']
        
        # Check if route is in progress
        active = self.route_manager.get_active_route()
        if active and str(active['route_id']) == str(route_id):
            self.input_handler.set_blocking(True)
            print(f"\n[RESUME] Route '{route_data['name']}' is in progress at step {active['current_idx']}.")
            print("  Do you want to Resume from there? (y/n)")
            try:
                ans = input("> ").strip().lower()
                if ans == 'y':
                    start_idx = int(active['current_idx'])
                    print(f"Resuming at step {start_idx}.")
                else:
                    print("Starting from beginning.")
                    self.route_manager.clear_active_route()
            finally:
                self.input_handler.set_blocking(False)
        
        self.load_route_data(route_data)
        self.current_landmark_idx = start_idx
        self.current_route_id = route_id
        
        print(f"[SELECTED] Route '{route_data['name']}' loaded. Press 'u' to start.")
        self.mode = "IDLE"
    
    def start_random_movement(self):
        """Start random movement mode."""
        self.mode = "RANDOM_MOVEMENT"
        self.random_player.start()
        print("\n[RANDOM] Random movement mode started. Press ESC to stop.")
    
    def start_playback(self):
        """Start playback of loaded route."""
        if not self.landmarks:
            print("[ERROR] No route loaded. Press 'p' to select a route.")
            return
        
        self.mode = "PLAYBACK"
        self.route_player.start_playback(self.route_type)
        
        if self.current_route_id:
            self.route_manager.set_active_route(self.current_route_id, self.current_landmark_idx)
        
        print(f"\n[PLAYBACK] Starting/Resuming at step {self.current_landmark_idx}.")
    
    def add_image_to_playback_step(self, step_idx: int, image: np.ndarray):
        """Add an image to a step during playback."""
        if not self.landmarks or not (0 <= step_idx < len(self.landmarks)):
            return
        
        crop = self._get_center_crop(image)
        timestamp = int(time.time())
        suffix = np.random.randint(0, 1000)
        filename = f"landmark_{timestamp}_{suffix}.png"
        path = os.path.join("templates/landmarks", filename)
        cv2.imwrite(path, crop)
        name = f"lm_{timestamp}_{suffix}"
        
        new_img = {"name": name, "filename": filename}
        self.landmarks[step_idx]['images'].append(new_img)
        self.vision.load_template(name, path)
        
        print(f"[PLAYBACK] Added new image to step {step_idx}.")
        self.route_manager.update_route_structure(self.current_route_id, self.landmarks)
    
    def delete_image_from_step(self, step_idx: int):
        """Delete an image from a step."""
        current_match_idx = self.route_player.get_current_match_img_index()
        
        still_has_images = self.route_manager.delete_image_from_step(
            self.landmarks, step_idx, self.current_landmark_idx,
            current_match_idx, None
        )
        
        if not still_has_images:
            self.delete_step(step_idx)
        else:
            # Update database based on route type
            if self.route_type == "HYBRID":
                self.route_manager.update_hybrid_route_structure(self.current_route_id, self.landmarks)
            else:
                self.route_manager.update_route_structure(self.current_route_id, self.landmarks)
    
    def delete_step(self, index: int):
        """Delete a step from the route."""
        result = self.route_manager.delete_step(self.landmarks, index)
        if not result['deleted']:
            return
        
        step_idx = result['step_idx']
        
        # Adjust current index
        if step_idx < self.current_landmark_idx:
            self.current_landmark_idx -= 1
        elif step_idx == self.current_landmark_idx:
            if self.current_landmark_idx >= len(self.landmarks):
                self.current_landmark_idx = 0
        
        # Update database based on route type
        if self.route_type == "HYBRID":
            self.route_manager.update_hybrid_route_structure(self.current_route_id, self.landmarks)
        else:
            self.route_manager.update_route_structure(self.current_route_id, self.landmarks)
        
        if not self.landmarks:
            print("[WARN] Route is now empty.")
            self.stop_all()
    
    def delete_next_node(self, next_idx: int):
        """
        Delete the next node/step from the route.
        
        Args:
            next_idx: Index of the next node to delete.
        """
        if not self.landmarks or not (0 <= next_idx < len(self.landmarks)):
            print(f"[DELETE] Cannot delete node {next_idx}: invalid index.")
            return
        
        if next_idx == self.current_landmark_idx:
            print("[DELETE] Cannot delete current node. Use delete_step for current node.")
            return
        
        result = self.route_manager.delete_step(self.landmarks, next_idx)
        if not result['deleted']:
            print(f"[DELETE] Failed to delete node {next_idx}.")
            return
        
        step_idx = result['step_idx']
        
        # Adjust current index if needed
        if step_idx < self.current_landmark_idx:
            self.current_landmark_idx -= 1
        
        # Update database based on route type
        if self.route_type == "HYBRID":
            success = self.route_manager.update_hybrid_route_structure(self.current_route_id, self.landmarks)
            if not success:
                print(f"[DELETE] Failed to update HYBRID route in database.")
                return
        else:
            success = self.route_manager.update_route_structure(self.current_route_id, self.landmarks)
            if not success:
                print(f"[DELETE] Failed to update route in database.")
                return
        
        print(f"[DELETE] Removed next node {step_idx} from route. Database updated.")
        
        if not self.landmarks:
            print("[WARN] Route is now empty.")
            self.stop_all()
    
    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """Get center crop of image."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_w, crop_h = 150, 150
        x = max(0, center_x - crop_w // 2)
        y = max(0, center_y - crop_h // 2)
        return image[y:y+crop_h, x:x+crop_w]
