"""
Movement state for handling navigation in the open world.

Main orchestrator that delegates to specialized components for recording,
playback, navigation, and visualization.
Updated to use new architecture (skills, sensors, filters, visualizers, UI).
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
from src.ui.input_handler import InputHandler
from src.ui.menu import MenuDefinition, MenuItem, MenuManager
from src.ui.text_input import TextInputDialog
from src.skills.movement import MovementSkill, CameraSkill
from src.visualizers.route import RoutePlaybackVisualizer, RecordingPreviewVisualizer
from src.sensors.minimap_state import MinimapStateSensor
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
    Now uses new architecture: skills, sensors, filters, visualizers, and UI components.
    """
    
    def __init__(self, manager):
        """Initialize the movement state."""
        super().__init__(manager)
        
        # Core components
        self.navigator = VisualNavigator(self.vision)
        self.route_manager = RouteManager(self.vision)
        self.nav_controller = NavigationController()
        self.seek_strategy = SeekStrategy()
        
        # Sensors
        self.minimap_state_sensor = MinimapStateSensor(manager.roi_cache)
        
        # New UI system
        self.input_handler = InputHandler()
        self.menu_manager = MenuManager(self.input_handler)
        self.text_input = TextInputDialog()
        
        # Set up menu manager and input handler in base class
        self.menu_manager = self.menu_manager
        self.input_handler = self.input_handler
        
        # Skills
        self.movement_skill = MovementSkill(self.controller)
        self.camera_skill = CameraSkill(self.controller)
        
        # Visualizers
        self.route_playback_viz = RoutePlaybackVisualizer()
        self.recording_preview_viz = RecordingPreviewVisualizer()
        # Keep old debug visualizer for now (for odometry/HSV debug that's integrated into navigator)
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
        self.selecting_recording_type = False
        
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
        """
        Check if movement state is active.
        
        Uses MinimapStateSensor to detect blue minimap frame (movement state).
        """
        # Enable sensor temporarily for detection
        self.minimap_state_sensor.enable()
        
        # Read minimap state
        state = self.minimap_state_sensor.read(image)
        
        # Disable sensor after reading (will be re-enabled in execute if needed)
        self.minimap_state_sensor.disable()
        
        return state == "movement"
    
    def get_menu(self) -> MenuDefinition:
        """
        Get menu definition for movement state.
        
        Returns:
            MenuDefinition with movement state controls
        """
        return MenuDefinition(
            title="Movement State",
            items=[
                MenuItem('r', 'Start New Recording (will prompt for type)', 'RECORD_MODE'),
                MenuItem('p', 'Playback Menu (0=Random Movement, 1-9=Select Route)', 'LIST_ROUTES'),
                MenuItem('u', 'Resume Current Loaded Route', 'PLAYBACK'),
                MenuItem('2', 'Delete Current Image (during playback)', 'DELETE_CURRENT_IMAGE'),
                MenuItem('3', 'Delete Next Image (during playback)', 'DELETE_NEXT_IMAGE'),
                MenuItem('4', 'Delete Next Node (during playback)', 'DELETE_NEXT_NODE'),
            ]
        )
    
    def on_enter(self):
        """Called when entering the state."""
        super().on_enter()  # This calls attach_input_listeners()
        print(f"\n--- Movement State ({self.mode}) ---")
        
        # Enable minimap state sensor
        self.minimap_state_sensor.enable()
        
        if self.mode == "RECORDING":
            print("  [RESUMED] Recording in progress.")
        elif self.mode == "PLAYBACK":
            print("  [RESUMED] Playback in progress.")
    
    def on_exit(self):
        """Called when exiting the state."""
        super().on_exit()  # This calls detach_input_listeners()
        
        # Disable minimap state sensor
        self.minimap_state_sensor.disable()
        
        # Stop movement and camera using skills
        self.movement_skill.stop()
        self.camera_skill.stop()
        self.controller.release_all()
        
        # Cleanup visualizers (only if they were shown)
        if self.route_playback_viz.is_visible():
            self.route_playback_viz.cleanup()
        if self.recording_preview_viz.is_visible():
            self.recording_preview_viz.cleanup()
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
        # Process input actions using new input handler
        actions = self.input_handler.get_pending_actions()
        
        for action in actions:
            if action == 'STOP':
                self.stop_all()
            elif action == 'RECORD_MODE':
                self.start_recording_dialog()
            elif action == 'SELECT_RECORDING_TYPE':
                data = self.input_handler.get_action_data('SELECT_RECORDING_TYPE')
                if data and 'key' in data:
                    try:
                        choice = data['key']
                        if self.selecting_recording_type:
                            if choice == '1':
                                self.start_recording("LANDMARK")
                                self.selecting_recording_type = False
                                # Restore normal menu bindings
                                if self.menu_manager:
                                    menu = self.get_menu()
                                    if menu:
                                        self.menu_manager.detach_menu()
                                        self.menu_manager.attach_menu(menu)
                            elif choice == '2':
                                self.start_recording("HYBRID")
                                self.selecting_recording_type = False
                                # Restore normal menu bindings
                                if self.menu_manager:
                                    menu = self.get_menu()
                                    if menu:
                                        self.menu_manager.detach_menu()
                                        self.menu_manager.attach_menu(menu)
                            elif choice == 'q':
                                print("Cancelled.")
                                self.selecting_recording_type = False
                                # Restore normal menu bindings
                                if self.menu_manager:
                                    menu = self.get_menu()
                                    if menu:
                                        self.menu_manager.detach_menu()
                                        self.menu_manager.attach_menu(menu)
                    except (ValueError, KeyError):
                        pass
            elif action == 'LIST_ROUTES':
                self.list_available_routes()
            elif action == 'SELECT_ROUTE':
                data = self.input_handler.get_action_data('SELECT_ROUTE')
                if data and 'key' in data:
                    try:
                        idx = int(data['key'])
                        if idx is not None and self.mode == "SELECTING":
                            if idx == 0:
                                self.start_random_movement()
                            else:
                                self.select_route(idx)
                    except ValueError:
                        pass
            elif action == 'RANDOM_MOVEMENT':
                self.start_random_movement()
            elif action == 'DELETE_CURRENT_IMAGE':
                if self.mode == "PLAYBACK":
                    self.delete_image_from_step(self.current_landmark_idx)
            elif action == 'DELETE_NEXT_IMAGE':
                if self.mode == "PLAYBACK" and self.landmarks:
                    next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                    self.delete_image_from_step(next_idx)
            elif action == 'DELETE_NEXT_NODE':
                if self.mode == "PLAYBACK" and self.landmarks:
                    next_idx = self.current_landmark_idx + 1
                    if next_idx < len(self.landmarks):
                        self.delete_next_node(next_idx)
                    else:
                        print("[DELETE] No next node to delete (at end of route).")
            elif action == 'PLAYBACK':
                self.start_playback()
            elif action == 'SAVE_EXIT':
                if self.mode == "RECORDING":
                    self.finish_recording()
            elif action == 'CAPTURE':
                if self.mode == "RECORDING":
                    self.route_recorder.capture_image(image)
                elif self.mode == "PLAYBACK":
                    self.add_image_to_playback_step(self.current_landmark_idx, image)
            elif action == 'NEXT_STEP':
                if self.mode == "RECORDING":
                    self.route_recorder.finish_step()
                elif self.mode == "PLAYBACK":
                    if self.landmarks:
                        next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
                        self.add_image_to_playback_step(next_idx, image)
            elif action == 'RETAKE':
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
            # LANDMARK mode - use new visualizer
            step_count = self.route_recorder.get_step_count()
            if self.recording_preview_viz.is_visible():
                preview_img = self.recording_preview_viz.render(image, {'step_count': step_count})
                cv2.imshow(self.recording_preview_viz.window_name, preview_img)
                cv2.waitKey(1)
            else:
                self.recording_preview_viz.show()
    
    def handle_random_movement(self, image: np.ndarray):
        """Handle random movement mode."""
        result = self.random_player.process_frame(image)
        
        # Apply controls using skills
        self.camera_skill.move(result['cam_x'], result['cam_y'])
        self.movement_skill.move(result['move_x'], result['move_y'])
    
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
            
            # Apply controls using skills
            self.camera_skill.move(result['cam_x'], result['cam_y'])
            self.movement_skill.move(result['move_x'], result['move_y'])
            
            # Show debug (still using old visualizer for odometry)
            # HSV debug is automatically shown when odometry debug is shown
            self.debug_visualizer.show_odometry_debug(
                image, result.get('target_mm'),
                self.controller.state,
                tracking_active=(result.get('status') == 'TRACKING'),
                status_msg=result.get('status', '')
            )
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
            
            # Apply controls using skills
            self.camera_skill.move(result['cam_x'], result['cam_y'])
            self.movement_skill.move(result['move_x'], result['move_y'])
            
            # Get next target filename for debug view
            next_idx = (self.current_landmark_idx + 1) % len(self.landmarks)
            next_filename = None
            if next_idx < len(self.landmarks):
                next_step = self.landmarks[next_idx]
                next_filename = next_step['images'][0]['filename'] if next_step.get('images') else None
            
            # Use new visualizer for playback debug
            if self.route_playback_viz.is_visible():
                viz_data = {
                    'match': result.get('match'),
                    'target_name': result.get('target_name', 'Unknown'),
                    'current_idx': self.current_landmark_idx,
                    'total_steps': len(self.landmarks),
                    'target_filename': result.get('target_filename'),
                    'next_target_filename': next_filename,
                    'seek_active': self.route_player.landmark_player.seek_active
                }
                viz_img = self.route_playback_viz.render(image, viz_data)
                cv2.imshow(self.route_playback_viz.window_name, viz_img)
                cv2.waitKey(1)
            else:
                self.route_playback_viz.show()
    
    def stop_all(self):
        """Stop all activities and return to IDLE."""
        print("[STOP] Stopping all activities. Returning to IDLE.")
        if self.mode == "RANDOM_MOVEMENT":
            self.random_player.stop()
        self.mode = "IDLE"
        self.movement_skill.stop()
        self.camera_skill.stop()
        self.controller.release_all()
        if self.route_playback_viz.is_visible():
            self.route_playback_viz.cleanup()
        if self.recording_preview_viz.is_visible():
            self.recording_preview_viz.cleanup()
        self.debug_visualizer.cleanup_windows()
        self.route_manager.clear_active_route()
    
    def start_recording_dialog(self):
        """Start recording with type selection dialog."""
        self.selecting_recording_type = True
        print("\nSelect Recording Type:")
        print("  1. Landmark Routing")
        print("  2. Hybrid Visual Odometry")
        print("  q. Cancel")
        print("Press number key (1, 2) or 'q' to select.")
        
        # Add temporary key bindings for recording type selection
        recording_type_bindings = {
            '1': 'SELECT_RECORDING_TYPE',
            '2': 'SELECT_RECORDING_TYPE',
            'q': 'SELECT_RECORDING_TYPE'
        }
        self.input_handler.add_key_bindings(recording_type_bindings)
    
    def start_recording(self, r_type: str):
        """Start recording a route."""
        self.mode = "RECORDING"
        self.route_type = r_type
        self.route_recorder.start_recording(r_type)
        self.recording_preview_viz.show()
    
    def finish_recording(self):
        """Finish recording and save route."""
        data, master_map_path = self.route_recorder.finish_recording()
        
        self.movement_skill.stop()
        self.camera_skill.stop()
        self.controller.release_all()
        self.recording_preview_viz.cleanup()
        self.debug_visualizer.cleanup_windows()
        
        # Use text input dialog for route name
        try:
            name = self.text_input.prompt("Enter name for this route", "")
            if not name:
                name = f"Route_{int(time.time())}"
            
            if self.route_manager.save_route(name, data, self.route_type, master_map_path):
                print(f"[SAVED] Route '{name}' saved to database.")
            else:
                print("[ERROR] Failed to save route.")
        except Exception as e:
            print(f"[ERROR] Failed to get route name: {e}")
            name = f"Route_{int(time.time())}"
            if self.route_manager.save_route(name, data, self.route_type, master_map_path):
                print(f"[SAVED] Route '{name}' saved to database (using default name).")
        
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
        
        # Add temporary key bindings for number keys when in SELECTING mode
        # These will work alongside the existing menu bindings
        route_selection_bindings = {str(i): 'SELECT_ROUTE' for i in range(10)}
        self.input_handler.add_key_bindings(route_selection_bindings)
    
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
            # Use text input dialog for resume question
            try:
                print(f"\n[RESUME] Route '{route_data['name']}' is in progress at step {active['current_idx']}.")
                ans = self.text_input.prompt("Resume from there? (y/n)", "y")
                if ans and ans.lower() == 'y':
                    start_idx = int(active['current_idx'])
                    print(f"Resuming at step {start_idx}.")
                else:
                    print("Starting from beginning.")
                    self.route_manager.clear_active_route()
            except Exception as e:
                print(f"[ERROR] Failed to get resume choice: {e}")
                print("Starting from beginning.")
                self.route_manager.clear_active_route()
        
        self.load_route_data(route_data)
        self.current_landmark_idx = start_idx
        self.current_route_id = route_id
        
        print(f"[SELECTED] Route '{route_data['name']}' loaded. Press 'u' to start.")
        self.mode = "IDLE"
        
        # Restore normal menu bindings by re-attaching the menu
        # This will reset bindings to the menu definition only
        if self.menu_manager:
            menu = self.get_menu()
            if menu:
                # Detach current menu and reattach to clear route selection bindings
                self.menu_manager.detach_menu()
                self.menu_manager.attach_menu(menu)
        
        self.route_playback_viz.show()
    
    def start_random_movement(self):
        """Start random movement mode."""
        self.mode = "RANDOM_MOVEMENT"
        self.random_player.start()
        print("\n[RANDOM] Random movement mode started. Press ESC to stop.")
        
        # Restore normal menu bindings by re-attaching the menu
        # This will reset bindings to the menu definition only
        if self.menu_manager:
            menu = self.get_menu()
            if menu:
                # Detach current menu and reattach to clear route selection bindings
                self.menu_manager.detach_menu()
                self.menu_manager.attach_menu(menu)
    
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
        self.route_playback_viz.show()
    
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
