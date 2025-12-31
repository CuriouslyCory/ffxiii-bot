"""
Route player for handling route playback in both LANDMARK and HYBRID modes.
"""

import time
import os
import cv2
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from .route_manager import LANDMARK_DIR
from .navigation_controller import NavigationController
from .seek_strategy import SeekStrategy
from .constants import (
    LOOKAHEAD_DEPTH, LANDMARK_SEEK_PAN_SPEED, LANDMARK_SEEK_TILT_SPEED,
    LANDMARK_TEMPLATE_THRESHOLD, LANDMARK_HIGH_CONFIDENCE_THRESHOLD,
    LANDMARK_DEADZONE, LANDMARK_MAX_OFFSET_X_FACTOR, LANDMARK_MAX_OFFSET_Y_FACTOR,
    LANDMARK_MAX_SPEED, LANDMARK_MOVE_THRESHOLD, LANDMARK_COAST_DURATION,
    LANDMARK_SEARCH_START_DELAY, LANDMARK_SEEK_TIMEOUT, LANDMARK_SEEK_TOGGLE_INTERVAL,
    ARRIVAL_BUFFER_SIZE,
    RANDOM_TURN_INTERVAL_MIN, RANDOM_TURN_INTERVAL_MAX,
    RANDOM_TURN_DURATION_MIN, RANDOM_TURN_DURATION_MAX,
    RANDOM_CAMERA_PAN_SPEED
)


class LandmarkPlayer:
    """
    Handles landmark-based route playback.
    
    Uses template matching to find landmarks and centers the camera on them.
    """
    
    def __init__(self, vision_engine, controller):
        """
        Initialize the landmark player.
        
        Args:
            vision_engine: Vision engine for template matching.
            controller: Controller for movement commands.
        """
        self.vision = vision_engine
        self.controller = controller
        self.current_match_img_index: Optional[int] = None
        self.search_start_time = 0.0
        self.last_seen_time = 0.0
        self.seek_active = False
        self.seek_start_time = 0.0
        self.seek_last_toggle_time = 0.0
        self.seek_vertical_direction = 1
        from .constants import LANDMARK_SEEK_PAN_SPEED, LANDMARK_SEEK_TILT_SPEED
        self.seek_pan_speed = LANDMARK_SEEK_PAN_SPEED
        self.seek_tilt_speed = LANDMARK_SEEK_TILT_SPEED
    
    def find_best_match_in_step(self, landmarks: List[Dict], step_idx: int, 
                                image: np.ndarray, threshold: float = 0.87,
                                preferred_img_idx: int = -1) -> Tuple[Optional[tuple], int]:
        """
        Find the best template match in a step.
        
        Args:
            landmarks: List of landmarks/steps.
            step_idx: Index of step to search.
            image: Current screen image.
            threshold: Template matching threshold.
            preferred_img_idx: Preferred image index to try first.
            
        Returns:
            Tuple of (match_result, image_index) or (None, -1) if no match.
        """
        if not landmarks or not (0 <= step_idx < len(landmarks)):
            return None, -1
        
        step = landmarks[step_idx]
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
                if conf > LANDMARK_HIGH_CONFIDENCE_THRESHOLD:
                    return match, idx
                
                if conf > best_conf:
                    best_conf = conf
                    best_match = match
                    best_img_idx = idx
        
        return best_match, best_img_idx
    
    def process_frame(self, landmarks: List[Dict], current_idx: int, 
                     image: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame during landmark playback.
        
        Args:
            landmarks: List of landmarks/steps.
            current_idx: Current landmark index.
            image: Current screen image.
            
        Returns:
            Dictionary with playback state and commands.
        """
        if not landmarks:
            return {'cam_x': 0.0, 'cam_y': 0.0, 'move_x': 0.0, 'move_y': 0.0, 
                   'match': None, 'target_name': None, 'target_filename': None}
        
        preferred_idx = self.current_match_img_index if self.current_match_img_index is not None else -1
        match, img_idx = self.find_best_match_in_step(
            landmarks, current_idx, image, threshold=LANDMARK_TEMPLATE_THRESHOLD, preferred_img_idx=preferred_idx
        )
        
        if match:
            self.current_match_img_index = img_idx
        
        # Lookahead: Check if we can skip ahead to next step
        max_lookahead = len(landmarks)
        for _ in range(max_lookahead):
            next_idx = (current_idx + 1) % len(landmarks)
            next_match, next_img_idx = self.find_best_match_in_step(
                landmarks, next_idx, image, threshold=LANDMARK_TEMPLATE_THRESHOLD
            )
            
            if next_match:
                print(f"[PLAYBACK] Found next step {next_idx}. Skipping ahead.")
                current_idx = next_idx
                match = next_match
                self.current_match_img_index = next_img_idx
                self.search_start_time = time.time()
                self.reset_seek_state()
                break
            else:
                break
        
        # Get target info for debug view
        if current_idx < len(landmarks):
            current_step = landmarks[current_idx]
            target_name = current_step['name']
            
            disp_img_idx = self.current_match_img_index if match and self.current_match_img_index is not None else 0
            if current_step['images']:
                if disp_img_idx >= len(current_step['images']):
                    disp_img_idx = 0
                target_filename = current_step['images'][disp_img_idx]['filename']
            else:
                target_filename = None
        else:
            target_name = "End of Route"
            target_filename = None
        
        # Compute controls
        cam_x, cam_y, move_x, move_y = 0.0, 0.0, 0.0, 0.0
        
        if match:
            self.reset_seek_state()
            self.last_seen_time = time.time()
            self.search_start_time = time.time()
            
            mx, my, _ = match
            mx += 75
            my += 75
            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            
            dx = mx - cx
            dy = my - cy
            deadzone = LANDMARK_DEADZONE
            
            max_offset_x = int(w * LANDMARK_MAX_OFFSET_X_FACTOR)
            max_offset_y = int(h * LANDMARK_MAX_OFFSET_Y_FACTOR)
            max_speed = LANDMARK_MAX_SPEED
            
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
            
            if abs(dx) < LANDMARK_MOVE_THRESHOLD:
                move_y = 1.0
            else:
                move_y = 0.0
        else:
            # Lost tracking - seek logic
            if time.time() - self.last_seen_time < LANDMARK_COAST_DURATION:
                move_y = 1.0
            else:
                move_y = 0.0
            
            if time.time() - self.search_start_time > LANDMARK_SEARCH_START_DELAY:
                cam_x, cam_y = self.execute_seek()
            else:
                cam_x, cam_y = 0.0, 0.0
        
        return {
            'cam_x': cam_x,
            'cam_y': cam_y,
            'move_x': move_x,
            'move_y': move_y,
            'match': match,
            'target_name': target_name,
            'target_filename': target_filename,
            'current_idx': current_idx
        }
    
    def execute_seek(self) -> Tuple[float, float]:
        """Execute seek behavior when landmark is lost."""
        current_time = time.time()
        if not self.seek_active:
            print("[SEEK] Lost landmark. Starting scan (pan right + alternating tilt).")
            self.seek_active = True
            self.seek_start_time = current_time
            self.seek_last_toggle_time = current_time
            self.seek_vertical_direction = -1
        
        if current_time - self.seek_start_time > LANDMARK_SEEK_TIMEOUT:
            print("[SEEK] FAILED. Timeout after 5 minutes. Stopping playback.")
            return 0.0, 0.0
        
        if current_time - self.seek_last_toggle_time >= LANDMARK_SEEK_TOGGLE_INTERVAL:
            self.seek_vertical_direction *= -1
            direction_name = "down" if self.seek_vertical_direction > 0 else "up"
            print(f"[SEEK] Toggling tilt direction: {direction_name}")
            self.seek_last_toggle_time = current_time
        
        cam_x = self.seek_pan_speed
        cam_y = self.seek_tilt_speed * self.seek_vertical_direction
        
        return cam_x, cam_y
    
    def reset_seek_state(self):
        """Reset seek state."""
        self.seek_active = False
        self.seek_start_time = 0.0
        self.seek_last_toggle_time = 0.0
        self.seek_vertical_direction = 1


class HybridPlayer:
    """
    Handles hybrid visual odometry route playback.
    
    Uses visual odometry to navigate between nodes with drift correction.
    """
    
    # Configuration (imported from constants)
    from .constants import LOOKAHEAD_DEPTH
    
    def __init__(self, navigator, controller, nav_controller, seek_strategy):
        """
        Initialize the hybrid player.
        
        Args:
            navigator: VisualNavigator instance.
            controller: Controller instance.
            nav_controller: NavigationController instance.
            seek_strategy: SeekStrategy instance.
        """
        self.navigator = navigator
        self.controller = controller
        self.nav_controller = nav_controller
        self.seek_strategy = seek_strategy
        self.arrival_buffer: List[Tuple[float, float, float]] = []
        self._minimap_cache: Dict[str, np.ndarray] = {}
        self.last_retry_attempt_time = 0.0
        self.retry_attempt_interval = 0.5  # Re-attempt drift computation every 0.5 seconds during retry
    
    def get_minimap_image(self, node: Dict, landmark_dir: str = LANDMARK_DIR) -> Optional[np.ndarray]:
        """
        Get minimap image for a node (with caching).
        
        Args:
            node: Node dictionary with minimap_path.
            landmark_dir: Directory containing landmark images.
            
        Returns:
            Minimap image or None if not found.
        """
        filename = node.get('minimap_path')
        if not filename:
            return None
        
        path = os.path.join(landmark_dir, filename)
        if not os.path.exists(path):
            return None
        
        # Cache lookup
        if path not in self._minimap_cache:
            img = cv2.imread(path)
            if img is not None:
                self._minimap_cache[path] = img
            else:
                return None
        
        return self._minimap_cache[path]
    
    def process_frame(self, landmarks: List[Dict], current_idx: int, 
                     image: np.ndarray, route_id: Optional[int] = None,
                     set_active_callback=None) -> Dict[str, Any]:
        """
        Process a frame during hybrid playback.
        
        Args:
            landmarks: List of nodes.
            current_idx: Current node index.
            image: Current screen image.
            route_id: Optional route ID for persistence.
            set_active_callback: Optional callback to set active route.
            
        Returns:
            Dictionary with playback state and commands.
        """
        if not landmarks:
            return {'cam_x': 0.0, 'cam_y': 0.0, 'move_x': 0.0, 'move_y': 0.0,
                   'drift': None, 'target_mm': None, 'status': 'NO_ROUTE'}
        
        # Check if route finished (loop back to start)
        if current_idx >= len(landmarks):
            print("[PLAYBACK] Route finished. Restarting at Node 0.")
            current_idx = 0
            self.arrival_buffer = []
            if route_id and set_active_callback:
                set_active_callback(route_id, current_idx)
        
        # Evaluate current node
        current_node = landmarks[current_idx]
        current_mm = self.get_minimap_image(current_node)
        
        target_mm = current_mm
        drift = None
        status = "TRACKING"
        
        if current_mm is not None:
            # Get target arrow angle for angle validation
            target_arrow_angle = current_node.get('arrow_angle')
            res = self.navigator.compute_drift(image, current_mm, target_arrow_angle)
            if res:
                dx, dy, angle, inliers, _ = res
                drift = (dx, dy, angle)
                
                # Maintain arrival rolling window
                if inliers >= self.navigator.MIN_INLIERS_FOR_TRACKING:
                    self.arrival_buffer.append((dx, dy, angle))
                    if len(self.arrival_buffer) > ARRIVAL_BUFFER_SIZE:
                        self.arrival_buffer.pop(0)
                
                # Check arrival
                has_arrived, dist, mean_angle = self.nav_controller.check_arrival(
                    dx, dy, angle, self.arrival_buffer
                )
                
                if has_arrived:
                    print(f"[PLAYBACK] Reached Node {current_idx} (Dist: {dist:.1f}, Ang: {mean_angle:.1f})")
                    current_idx += 1
                    self.arrival_buffer = []
                    if route_id and set_active_callback:
                        set_active_callback(route_id, current_idx)
                    # Return early to process next node next frame
                    return {'cam_x': 0.0, 'cam_y': 0.0, 'move_x': 0.0, 'move_y': 0.0,
                           'drift': None, 'target_mm': None, 'status': 'ARRIVED',
                           'new_idx': current_idx}
                
                # Track approach distance
                self.nav_controller.set_approach_dist(dist)
                
                # Check if tracking is good
                if inliers >= self.navigator.MIN_INLIERS_FOR_TRACKING:
                    # Valid tracking
                    pass
                else:
                    # Weak tracking - try lookahead recovery
                    candidates = [{'idx': current_idx, 'res': res}]
                    
                    for offset in range(1, LOOKAHEAD_DEPTH):
                        idx = current_idx + offset
                        if idx >= len(landmarks):
                            break
                        
                        node = landmarks[idx]
                        mm = self.get_minimap_image(node)
                        if mm is None:
                            continue
                        
                        node_arrow_angle = node.get('arrow_angle')
                        r = self.navigator.compute_drift(image, mm, node_arrow_angle)
                        if r:
                            candidates.append({'idx': idx, 'res': r})
                    
                    # Find best candidate
                    best = None
                    for c in candidates:
                        if c['res'][3] > self.navigator.MIN_INLIERS_FOR_RECOVERY:
                            if best is None or c['res'][3] > best['res'][3]:
                                best = c
                    
                    if best and best['idx'] > current_idx:
                        print(f"[PLAYBACK] Lost Node {current_idx}. Recovered at Node {best['idx']} "
                              f"(Inliers: {best['res'][3]})")
                        current_idx = best['idx']
                        if route_id and set_active_callback:
                            set_active_callback(route_id, current_idx)
                        
                        drift = (best['res'][0], best['res'][1], best['res'][2])
                        target_mm = self.get_minimap_image(landmarks[current_idx])
                        target_node = landmarks[current_idx]
                        target_arrow_angle = target_node.get('arrow_angle')
                        self.navigator.compute_drift(image, target_mm, target_arrow_angle)
            else:
                # compute_drift returned None (lost state - unrealistic offsets or no match)
                # Still attempt lookahead recovery before falling back to seek
                candidates = []
                
                for offset in range(1, LOOKAHEAD_DEPTH):
                    idx = current_idx + offset
                    if idx >= len(landmarks):
                        break
                    
                    node = landmarks[idx]
                    mm = self.get_minimap_image(node)
                    if mm is None:
                        continue
                    
                    node_arrow_angle = node.get('arrow_angle')
                    r = self.navigator.compute_drift(image, mm, node_arrow_angle)
                    if r:
                        candidates.append({'idx': idx, 'res': r})
                
                # Find best candidate
                best = None
                for c in candidates:
                    if c['res'][3] > self.navigator.MIN_INLIERS_FOR_RECOVERY:
                        if best is None or c['res'][3] > best['res'][3]:
                            best = c
                
                if best:
                    print(f"[PLAYBACK] Lost Node {current_idx} (unrealistic offset). Recovered at Node {best['idx']} "
                          f"(Inliers: {best['res'][3]})")
                    current_idx = best['idx']
                    if route_id and set_active_callback:
                        set_active_callback(route_id, current_idx)
                    
                    dx, dy, angle = best['res'][0], best['res'][1], best['res'][2]
                    drift = (dx, dy, angle)
                    target_mm = self.get_minimap_image(landmarks[current_idx])
                    self.navigator.compute_drift(image, target_mm)
                    
                    # Track approach distance for recovered node
                    dist = np.sqrt(dx*dx + dy*dy)
                    self.nav_controller.set_approach_dist(dist)
        
        # Handle lost tracking (if still None after recovery attempt)
        if drift is None:
            self.nav_controller.set_approach_dist(None)
            current_time = time.time()
            
            # During retry phase, periodically re-attempt drift computation
            # This allows us to re-acquire tracking as the camera moves
            time_since_last_seen = current_time - self.seek_strategy.last_seen_time
            from .constants import COAST_DURATION, COAST_DURATION_EXTENDED, COAST_TURNING_THRESHOLD, COAST_FORWARD_THRESHOLD, RETRY_DURATION, RETRY_ATTEMPTS
            
            # Determine if we're in retry phase (after coasting)
            last_ctrl = self.seek_strategy.last_valid_controls or self.controller.state
            last_rx = last_ctrl.get('rx', 0.0)
            last_ly = last_ctrl.get('ly', 0.0)
            coast_duration = COAST_DURATION
            if (abs(last_rx) > COAST_TURNING_THRESHOLD and 
                abs(last_ly) < COAST_FORWARD_THRESHOLD):
                coast_duration = COAST_DURATION_EXTENDED
            
            in_retry_phase = time_since_last_seen >= coast_duration
            
            # Re-attempt drift computation during retry phase
            if in_retry_phase and (current_time - self.last_retry_attempt_time) >= self.retry_attempt_interval:
                self.last_retry_attempt_time = current_time
                
                # Try current node first
                if current_mm is not None:
                    current_node_arrow_angle = current_node.get('arrow_angle')
                    res = self.navigator.compute_drift(image, current_mm, current_node_arrow_angle)
                    if res:
                        dx, dy, angle, inliers, _ = res
                        if inliers >= self.navigator.MIN_INLIERS_FOR_RECOVERY:
                            # Re-acquired tracking!
                            print(f"[RETRY] Re-acquired tracking on Node {current_idx} (Inliers: {inliers})")
                            drift = (dx, dy, angle)
                            target_mm = current_mm
                            self.seek_strategy.update_last_seen()
                            dist = np.sqrt(dx*dx + dy*dy)
                            self.nav_controller.set_approach_dist(dist)
                            status = "TRACKING"
                        else:
                            # Try lookahead recovery
                            for offset in range(1, LOOKAHEAD_DEPTH):
                                idx = current_idx + offset
                                if idx >= len(landmarks):
                                    break
                                
                                node = landmarks[idx]
                                mm = self.get_minimap_image(node)
                                if mm is None:
                                    continue
                                
                                r = self.navigator.compute_drift(image, mm)
                                if r and r[3] >= self.navigator.MIN_INLIERS_FOR_RECOVERY:
                                    print(f"[RETRY] Re-acquired tracking on Node {idx} (Inliers: {r[3]})")
                                    current_idx = idx
                                    if route_id and set_active_callback:
                                        set_active_callback(route_id, current_idx)
                                    drift = (r[0], r[1], r[2])
                                    target_mm = mm
                                    # Update target for next iteration
                                    target_node = landmarks[current_idx]
                                    target_arrow_angle = target_node.get('arrow_angle')
                                    self.navigator.compute_drift(image, target_mm, target_arrow_angle)
                                    self.seek_strategy.update_last_seen()
                                    dist = np.sqrt(r[0]*r[0] + r[1]*r[1])
                                    self.nav_controller.set_approach_dist(dist)
                                    status = "TRACKING"
                                    break
            
            # If still no drift, use seek strategy
            if drift is None:
                cam_x, cam_y, move_x, move_y, status_msg = self.seek_strategy.process_seek(
                    current_time, self.controller.state
                )
                status = status_msg
            else:
                # We re-acquired tracking, compute controls normally
                sdx, sdy, sangle = self.nav_controller.smooth_drift(drift[0], drift[1], drift[2])
                approach_dist = self.nav_controller.get_approach_dist()
                cam_x, cam_y, move_x, move_y = self.nav_controller.compute_controls(
                    sdx, sdy, sangle, approach_dist
                )
                self.seek_strategy.set_last_valid_controls(self.controller.state)
        else:
            # Valid drift - compute controls
            self.seek_strategy.update_last_seen()
            dx, dy, angle = drift
            
            # Smooth drift
            sdx, sdy, sangle = self.nav_controller.smooth_drift(dx, dy, angle)
            
            # Compute controls
            approach_dist = self.nav_controller.get_approach_dist()
            cam_x, cam_y, move_x, move_y = self.nav_controller.compute_controls(
                sdx, sdy, sangle, approach_dist
            )
            
            # Store valid controls for seek strategy
            self.seek_strategy.set_last_valid_controls(self.controller.state)
            status = "TRACKING"
        
        return {
            'cam_x': cam_x,
            'cam_y': cam_y,
            'move_x': move_x,
            'move_y': move_y,
            'drift': drift,
            'target_mm': target_mm,
            'status': status,
            'current_idx': current_idx
        }


class RandomMovementPlayer:
    """
    Handles semi-random movement playback.
    
    Continuously moves forward while randomly panning the camera left or right at intervals.
    """
    
    def __init__(self, controller):
        """
        Initialize the random movement player.
        
        Args:
            controller: Controller for movement commands.
        """
        self.controller = controller
        self.next_turn_time = 0.0
        self.turn_end_time = 0.0
        self.is_turning = False
        self.turn_direction = 1  # 1 for right, -1 for left
    
    def start(self):
        """Reset state and generate first random turn interval."""
        current_time = time.time()
        interval = random.uniform(RANDOM_TURN_INTERVAL_MIN, RANDOM_TURN_INTERVAL_MAX)
        self.next_turn_time = current_time + interval
        self.turn_end_time = 0.0
        self.is_turning = False
        print(f"[RANDOM] Starting random movement mode. First turn in {interval:.2f}s")
    
    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame during random movement playback.
        
        Args:
            image: Current screen image (unused but kept for interface consistency).
            
        Returns:
            Dictionary with movement commands.
        """
        current_time = time.time()
        
        # Always move forward
        move_x = 0.0
        move_y = -1.0  # Forward movement
        cam_x = 0.0
        cam_y = 0.0
        
        # Check if we should start a new turn
        if not self.is_turning and current_time >= self.next_turn_time:
            # Start a new turn
            self.is_turning = True
            # 1 in 5 chance (20%) to turn left, otherwise turn right
            self.turn_direction = -1 if random.random() < 0.2 else 1
            turn_duration = random.uniform(RANDOM_TURN_DURATION_MIN, RANDOM_TURN_DURATION_MAX)
            self.turn_end_time = current_time + turn_duration
            direction_str = "left" if self.turn_direction < 0 else "right"
            print(f"[RANDOM] Starting turn {direction_str} for {turn_duration:.2f}s")
        
        # Check if we should end the current turn
        if self.is_turning and current_time >= self.turn_end_time:
            # End the turn and schedule next one
            self.is_turning = False
            interval = random.uniform(RANDOM_TURN_INTERVAL_MIN, RANDOM_TURN_INTERVAL_MAX)
            self.next_turn_time = current_time + interval
            print(f"[RANDOM] Turn ended. Next turn in {interval:.2f}s")
        
        # Apply camera pan if turning (left or right based on turn_direction)
        if self.is_turning:
            cam_x = RANDOM_CAMERA_PAN_SPEED * self.turn_direction
        
        return {
            'cam_x': cam_x,
            'cam_y': cam_y,
            'move_x': move_x,
            'move_y': move_y
        }
    
    def stop(self):
        """Clean up state."""
        self.is_turning = False
        self.next_turn_time = 0.0
        self.turn_end_time = 0.0
        print("[RANDOM] Random movement mode stopped.")


class RoutePlayer:
    """
    Unified interface for route playback.
    
    Delegates to LandmarkPlayer or HybridPlayer based on route type.
    """
    
    def __init__(self, vision_engine, controller, navigator, 
                 nav_controller, seek_strategy):
        """
        Initialize the route player.
        
        Args:
            vision_engine: Vision engine instance.
            controller: Controller instance.
            navigator: VisualNavigator instance.
            nav_controller: NavigationController instance.
            seek_strategy: SeekStrategy instance.
        """
        self.vision = vision_engine
        self.controller = controller
        self.navigator = navigator
        self.nav_controller = nav_controller
        self.seek_strategy = seek_strategy
        self.landmark_player = LandmarkPlayer(vision_engine, controller)
        self.hybrid_player = HybridPlayer(navigator, controller, nav_controller, seek_strategy)
        self.route_type: Optional[str] = None
    
    def start_playback(self, route_type: str):
        """
        Start playback of a route.
        
        Args:
            route_type: "LANDMARK" or "HYBRID".
        """
        self.route_type = route_type
        if route_type == "HYBRID":
            self.nav_controller.reset()
            self.seek_strategy.reset()
    
    def process_frame(self, landmarks: List[Dict], current_idx: int, 
                     image: np.ndarray, route_id: Optional[int] = None,
                     set_active_callback=None) -> Dict[str, Any]:
        """
        Process a frame during playback.
        
        Args:
            landmarks: List of landmarks/nodes.
            current_idx: Current index.
            image: Current screen image.
            route_id: Optional route ID.
            set_active_callback: Optional callback for setting active route.
            
        Returns:
            Dictionary with playback state and commands.
        """
        if self.route_type == "HYBRID":
            return self.hybrid_player.process_frame(
                landmarks, current_idx, image, route_id, set_active_callback
            )
        else:  # LANDMARK
            result = self.landmark_player.process_frame(landmarks, current_idx, image)
            # Update current_idx if lookahead advanced
            if 'current_idx' in result:
                current_idx = result['current_idx']
            return result
    
    def get_current_match_img_index(self) -> Optional[int]:
        """Get current match image index (LANDMARK mode only)."""
        if self.route_type == "LANDMARK":
            return self.landmark_player.current_match_img_index
        return None
