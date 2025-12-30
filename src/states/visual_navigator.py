import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict

class VisualNavigator:
    """
    Handles Visual Odometry and Hybrid Navigation logic.
    Separated from MovementState to keep file sizes manageable.
    """
    
    def __init__(self, vision_engine):
        self.vision = vision_engine
        
        # Default Minimap configuration (Approximate for 1080p, needs calibration)
        self.minimap_roi = (1580, 50, 300, 300) # x, y, w, h
        self.minimap_center = (150, 150) # relative to ROI
        self.minimap_radius = 130
        
        # Initialize mask
        self.mask = np.zeros((300, 300), dtype=np.uint8)
        self._update_mask()
        
        # Odometry State
        self.last_dx = 0
        self.last_dy = 0
        self.last_angle = 0
        
        # Debugging
        self.debug_window_name = "Visual Odometry"
        
        # Attempt Auto-Calibration on Init if possible?
        # Probably better to do it explicitly or lazily
        self.calibrated = False

    def _update_mask(self):
        self.mask = np.zeros((self.minimap_roi[3], self.minimap_roi[2]), dtype=np.uint8)
        cv2.circle(self.mask, self.minimap_center, self.minimap_radius, (255), -1)

    def calibrate(self, image: np.ndarray) -> bool:
        """
        Attempts to auto-calibrate the minimap ROI using the minimap_outline template.
        """
        # Search in the top-right quadrant
        h, w = image.shape[:2]
        search_roi = (w // 2, 0, w // 2, h // 2)
        
        # Assuming 'minimap_outline' template is loaded in vision engine
        # If not, we might need to load it here or assume it's loaded.
        # Check if template exists in vision engine
        if "minimap_outline" not in self.vision.templates:
            # Try to load it if we know the path?
            # Or just fail gracefully.
            print("[CALIB] 'minimap_outline' template not loaded.")
            return False

        match = self.vision.find_template("minimap_outline", image, threshold=0.3, roi=search_roi)
        
        if match:
            mx, my, conf = match
            template = self.vision.templates["minimap_outline"]
            th, tw = template.shape[:2]
            
            # Center of the match
            center_x = mx + tw // 2
            center_y = my + th // 2
            
            # Define new ROI centered on this
            roi_w, roi_h = 300, 300
            new_x = max(0, center_x - roi_w // 2)
            new_y = max(0, center_y - roi_h // 2)
            
            # Ensure within bounds
            if new_x + roi_w > w: new_x = w - roi_w
            if new_y + roi_h > h: new_y = h - roi_h
            
            self.minimap_roi = (new_x, new_y, roi_w, roi_h)
            self.minimap_center = (roi_w // 2, roi_h // 2)
            # Radius: roughly fit inside the template ring? 
            # If template is the ring, radius should be slightly less than half width
            # Let's keep 130 as it seems reasonable for 300px ROI
            
            self._update_mask()
            self.calibrated = True
            print(f"[CALIB] Minimap ROI calibrated to: {self.minimap_roi}")
            return True
        else:
            print("[CALIB] Failed to find minimap_outline.")
            return False
            
    def get_minimap_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns the raw (unmasked) crop of the minimap ROI.
        Ensures calibration is performed if not already done.
        """
        if not self.calibrated:
             self.calibrate(image)
             self.calibrated = True 

        roi_img = self.vision.get_roi_slice(image, self.minimap_roi)
        
        # Safety check if ROI is valid for image
        if roi_img.shape[0] != self.minimap_roi[3] or roi_img.shape[1] != self.minimap_roi[2]:
            return None
            
        return roi_img

    def extract_minimap(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts and masks the minimap from the full screen image.
        """
        roi_img = self.get_minimap_crop(image)
        if roi_img is None:
            return None
            
        # Apply circular mask
        masked = cv2.bitwise_and(roi_img, roi_img, mask=self.mask)
        return masked

    def compute_drift(self, current_img: np.ndarray, target_minimap: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Computes the visual drift between current view and target minimap node.
        Returns (dx, dy, d_theta) or None if matching failed.
        """
        current_minimap = self.extract_minimap(current_img)
        if current_minimap is None or target_minimap is None:
            return None
            
        # Detect and Compute features
        kp1, des1 = self.vision.feature_matcher.detect_and_compute(target_minimap, self.mask)
        kp2, des2 = self.vision.feature_matcher.detect_and_compute(current_minimap, self.mask)
        
        # Match
        matches = self.vision.feature_matcher.match_features(des1, des2)
        
        # Compute Homography
        M, mask = self.vision.feature_matcher.compute_homography(kp1, kp2, matches)
        
        if M is None:
            return None
            
        # Decompose
        dx, dy, angle = self.vision.feature_matcher.decompose_homography(M)
        
        self.last_dx = dx
        self.last_dy = dy
        self.last_angle = angle
        
        return dx, dy, angle

    def show_debug_view(self, current_img: np.ndarray, target_minimap: np.ndarray, controller_state: Dict = None, tracking_active: bool = True, status_msg: str = ""):
        """
        Displays a debug window showing the visual odometry process and controller input.
        """
        current_minimap = self.extract_minimap(current_img)
        if current_minimap is None: return

        # Dimensions
        h, w = current_minimap.shape[:2]
        
        # Canvas Layout:
        # [Target Minimap] [Current Minimap]
        # [      Controller Overlay        ]
        
        # Controller overlay height
        ctrl_h = 150
        canvas_h = h + ctrl_h
        canvas_w = w * 2 + 20
        
        debug_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Target (Left)
        if target_minimap is not None:
             if target_minimap.shape[:2] != (h, w):
                 target_minimap = cv2.resize(target_minimap, (w, h))
             
             # Apply mask to Target for display consistency
             masked_target = cv2.bitwise_and(target_minimap, target_minimap, mask=self.mask)
             debug_canvas[0:h, 0:w] = masked_target
        
        # Current (Right)
        debug_canvas[0:h, w+20:w*2+20] = current_minimap
        
        # Overlay Info (Drift)
        info_text = f"dx: {self.last_dx:.2f}, dy: {self.last_dy:.2f}, ang: {self.last_angle:.2f}"
        cv2.putText(debug_canvas, info_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw flow arrow on Current
        cx, cy = w + 20 + w//2, h//2
        end_x = int(cx + self.last_dx)
        end_y = int(cy + self.last_dy)
        cv2.arrowedLine(debug_canvas, (cx, cy), (end_x, end_y), (0, 0, 255), 2)

        # Status Overlay
        if not tracking_active:
             msg = status_msg if status_msg else "LOST TRACKING"
             # Centered text
             text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
             tx = (canvas_w - text_size[0]) // 2
             ty = h // 2
             cv2.putText(debug_canvas, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # --- Controller Output Visualization ---
        ctrl_y_start = h
        
        # Background for controller area
        cv2.rectangle(debug_canvas, (0, ctrl_y_start), (canvas_w, canvas_h), (30, 30, 30), -1)
        
        if controller_state:
            # Layout: Left Stick | Buttons | Right Stick
            
            # Centers
            ls_center = (100, ctrl_y_start + 75)
            rs_center = (canvas_w - 100, ctrl_y_start + 75)
            btn_center = (canvas_w // 2, ctrl_y_start + 75)
            
            radius = 40
            
            # Left Stick (Movement)
            cv2.circle(debug_canvas, ls_center, radius, (100, 100, 100), 2)
            lx = controller_state.get('lx', 0.0)
            ly = controller_state.get('ly', 0.0) # Remember gamepad Y: -1 is Up
            # Map -1..1 to pixels
            ls_dot_x = int(ls_center[0] + lx * radius)
            ls_dot_y = int(ls_center[1] + ly * radius) # +ly because ly=-1 (Up) -> -radius (Top)
            cv2.circle(debug_canvas, (ls_dot_x, ls_dot_y), 8, (0, 255, 255), -1)
            cv2.putText(debug_canvas, "L-Stick (Move)", (ls_center[0]-50, ls_center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            
            # Right Stick (Camera)
            cv2.circle(debug_canvas, rs_center, radius, (100, 100, 100), 2)
            rx = controller_state.get('rx', 0.0)
            ry = controller_state.get('ry', 0.0)
            rs_dot_x = int(rs_center[0] + rx * radius)
            rs_dot_y = int(rs_center[1] + ry * radius) # +ry because ry=-1 (Up) -> -radius (Top)
            cv2.circle(debug_canvas, (rs_dot_x, rs_dot_y), 8, (0, 255, 255), -1)
            cv2.putText(debug_canvas, "R-Stick (Cam)", (rs_center[0]-50, rs_center[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            
            # Buttons (Diamond layout)
            btns = controller_state.get('buttons', set())
            # Y (North), B (East), A (South), X (West)
            # wait, Xbox: Y=North, B=East, A=South, X=West
            
            # Positions relative to btn_center
            b_radius = 12
            off = 35
            
            pos_y = (btn_center[0], btn_center[1] - off) # North (Y)
            pos_b = (btn_center[0] + off, btn_center[1]) # East (B)
            pos_a = (btn_center[0], btn_center[1] + off) # South (A)
            pos_x = (btn_center[0] - off, btn_center[1]) # West (X)
            
            # Draw
            # Y (Yellow)
            col_y = (0, 255, 255) if 'y' in btns else (50, 50, 0)
            cv2.circle(debug_canvas, pos_y, b_radius, col_y, -1)
            cv2.putText(debug_canvas, "Y", (pos_y[0]-5, pos_y[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            
            # B (Red)
            col_b = (0, 0, 255) if 'b' in btns else (50, 0, 0)
            cv2.circle(debug_canvas, pos_b, b_radius, col_b, -1)
            cv2.putText(debug_canvas, "B", (pos_b[0]-5, pos_b[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # A (Green)
            col_a = (0, 255, 0) if 'a' in btns else (0, 50, 0)
            cv2.circle(debug_canvas, pos_a, b_radius, col_a, -1)
            cv2.putText(debug_canvas, "A", (pos_a[0]-5, pos_a[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # X (Blue)
            col_x = (255, 0, 0) if 'x' in btns else (50, 0, 0)
            cv2.circle(debug_canvas, pos_x, b_radius, col_x, -1)
            cv2.putText(debug_canvas, "X", (pos_x[0]-5, pos_x[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        cv2.imshow(self.debug_window_name, debug_canvas)
        
        # Move window to right of game window on first show
        # Note: Some Window Managers might ignore this if called too fast.
        if not hasattr(self, '_debug_win_pos_set'):
            win_x, win_y = self.vision.window_offset if hasattr(self.vision, 'window_offset') else (0, 0)
            win_w, win_h = self.vision.resolution if hasattr(self.vision, 'resolution') else (1920, 1080)
            # Place at x + width + padding
            debug_x = win_x + win_w + 10
            debug_y = win_y + 100
            
            # Explicitly create window just in case imshow didn't fully register it for moveWindow
            cv2.namedWindow(self.debug_window_name)
            cv2.moveWindow(self.debug_window_name, debug_x, debug_y)
            
            print(f"[DEBUG UI] Moving Visual Odometry window to: ({debug_x}, {debug_y})")
            self._debug_win_pos_set = True
            
        cv2.waitKey(1)
