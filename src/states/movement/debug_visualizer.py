"""
Debug visualizer for handling all debug windows and visualization logic.
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
from .route_manager import LANDMARK_DIR


class DebugVisualizer:
    """
    Handles all debug windows and visualization logic.
    
    Responsibilities:
    - Landmark preview window (recording)
    - Visual odometry debug window (playback)
    - HSV filter debug window
    - Route playback debug view (target images, status)
    - Window lifecycle management
    """
    
    def __init__(self, navigator):
        """
        Initialize the debug visualizer.
        
        Args:
            navigator: VisualNavigator instance for odometry debug.
        """
        self.navigator = navigator
        self._debug_win_pos_set = False
    
    def show_recording_preview(self, image: np.ndarray, step_count: int):
        """
        Show recording preview window for landmark recording.
        
        Args:
            image: Current screen image.
            step_count: Number of images in current step.
        """
        crop = self._get_center_crop(image)
        preview = crop.copy()
        cv2.rectangle(preview, (0, 0), (149, 149), (0, 255, 0), 2)
        cv2.putText(preview, f"Step Imgs: {step_count}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Landmark Preview", preview)
        cv2.waitKey(1)
    
    def show_playback_debug(self, image: np.ndarray, match: Optional[Tuple],
                           target_name: str, current_idx: int, total_steps: int,
                           target_filename: Optional[str] = None,
                           next_target_filename: Optional[str] = None,
                           seek_active: bool = False):
        """
        Show debug view for landmark playback.
        
        Args:
            image: Current screen image.
            match: Template match result or None.
            target_name: Name of current target step.
            current_idx: Current step index.
            total_steps: Total number of steps.
            target_filename: Filename of current target image.
            next_target_filename: Filename of next target image.
            seek_active: Whether seek mode is active.
        """
        scale = 0.5
        h, w = image.shape[:2]
        debug_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        dh, dw = debug_img.shape[:2]
        
        cx, cy = dw // 2, dh // 2
        cv2.line(debug_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(debug_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        status = "TRACKING" if match else ("SEEKING" if seek_active else "SEARCHING")
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
        cv2.putText(debug_img, f"Step {current_idx + 1}/{total_steps}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        final_img = debug_img
        
        # Add target images if available
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
        
        win = "FFXIII Bot - Debug"
        cv2.imshow(win, final_img)
        if not self._debug_win_pos_set:
            cv2.moveWindow(win, 1930, 50)
            self._debug_win_pos_set = True
        cv2.waitKey(1)
    
    def show_odometry_debug(self, image: np.ndarray, target_mm: Optional[np.ndarray],
                           controller_state: dict, tracking_active: bool = True,
                           status_msg: str = ""):
        """
        Show visual odometry debug window (delegates to navigator).
        
        Args:
            image: Current screen image.
            target_mm: Target minimap image.
            controller_state: Current controller state.
            tracking_active: Whether tracking is active.
            status_msg: Status message to display.
        """
        self.navigator.show_debug_view(image, target_mm, controller_state,
                                      tracking_active=tracking_active,
                                      status_msg=status_msg)
    
    def show_hsv_debug(self, image: np.ndarray):
        """
        Show HSV filter debug window (delegates to navigator).
        
        Args:
            image: Current screen image.
        """
        if self.navigator.hsv_debug_enabled:
            self.navigator.show_hsv_debug(image)
    
    def cleanup_windows(self):
        """Clean up all debug windows."""
        try:
            cv2.destroyWindow("Landmark Preview")
        except:
            pass
        try:
            cv2.destroyWindow(self.navigator.debug_window_name)
        except:
            pass
        try:
            cv2.destroyWindow(self.navigator.hsv_debug_window_name)
        except:
            pass
    
    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """Get center crop of image."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_w, crop_h = 150, 150
        x = max(0, center_x - crop_w // 2)
        y = max(0, center_y - crop_h // 2)
        return image[y:y+crop_h, x:x+crop_w]
