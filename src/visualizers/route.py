"""
Route playback visualizer for displaying landmark playback debug information.
"""
import cv2
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
from .base import Visualizer


class RoutePlaybackVisualizer(Visualizer):
    """
    Visualizer for route playback debug view.
    
    Shows current screen, target images, and playback status.
    """
    
    def __init__(self, landmark_dir: str = "templates/landmarks"):
        """
        Initialize route playback visualizer.
        
        Args:
            landmark_dir: Directory containing landmark images
        """
        super().__init__("FFXIII Bot - Debug")
        self.landmark_dir = landmark_dir
        self._window_pos_set = False
    
    def render(
        self,
        image: np.ndarray,
        data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render the playback debug view.
        
        Data dictionary should contain:
        - 'match': Optional tuple (x, y, confidence) of template match
        - 'target_name': Name of current target step
        - 'current_idx': Current step index
        - 'total_steps': Total number of steps
        - 'target_filename': Optional filename of current target image
        - 'next_target_filename': Optional filename of next target image
        - 'seek_active': Whether seek mode is active
        """
        scale = 0.5
        h, w = image.shape[:2]
        debug_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        dh, dw = debug_img.shape[:2]
        
        # Draw center crosshair
        cx, cy = dw // 2, dh // 2
        cv2.line(debug_img, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(debug_img, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
        
        # Determine status
        match = data.get('match')
        seek_active = data.get('seek_active', False)
        status = "TRACKING" if match else ("SEEKING" if seek_active else "SEARCHING")
        color = (0, 255, 0) if match else (0, 165, 255)
        
        # Draw match if present
        if match:
            mx, my, conf = match
            bx, by = int(mx * scale), int(my * scale)
            bs = int(150 * scale)
            cv2.rectangle(debug_img, (bx, by), (bx + bs, by + bs), (0, 255, 0), 2)
            lcx, lcy = bx + bs // 2, by + bs // 2
            cv2.line(debug_img, (lcx, lcy), (cx, cy), (0, 255, 0), 1)
            cv2.putText(debug_img, f"{conf:.2f}", (bx, by - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw status and info
        target_name = data.get('target_name', 'Unknown')
        current_idx = data.get('current_idx', 0)
        total_steps = data.get('total_steps', 0)
        
        cv2.putText(debug_img, f"[{status}]", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(debug_img, f"Target: {target_name}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_img, f"Step {current_idx + 1}/{total_steps}",
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        final_img = debug_img
        
        # Add target images if available
        target_filename = data.get('target_filename')
        if target_filename:
            path = os.path.join(self.landmark_dir, target_filename)
            if os.path.exists(path):
                target_img = cv2.imread(path)
                
                next_target_filename = data.get('next_target_filename')
                next_target_img = None
                if next_target_filename:
                    next_path = os.path.join(self.landmark_dir, next_target_filename)
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
        
        return final_img
    
    def show(self):
        """Show the visualizer window."""
        super().show()
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            if not self._window_pos_set:
                cv2.moveWindow(self.window_name, 1930, 50)
                self._window_pos_set = True
        except cv2.error:
            # Window might already exist, which is fine
            pass


class RecordingPreviewVisualizer(Visualizer):
    """
    Visualizer for recording preview window.
    
    Shows center crop preview during landmark recording.
    """
    
    def __init__(self):
        """Initialize recording preview visualizer."""
        super().__init__("Landmark Preview")
    
    def render(self, image: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """
        Render the recording preview.
        
        Data dictionary should contain:
        - 'step_count': Number of images in current step
        """
        crop = self._get_center_crop(image)
        preview = crop.copy()
        cv2.rectangle(preview, (0, 0), (149, 149), (0, 255, 0), 2)
        
        step_count = data.get('step_count', 0)
        cv2.putText(preview, f"Step Imgs: {step_count}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return preview
    
    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """Get center crop of image."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_w, crop_h = 150, 150
        x = max(0, center_x - crop_w // 2)
        y = max(0, center_y - crop_h // 2)
        return image[y:y+crop_h, x:x+crop_w]
