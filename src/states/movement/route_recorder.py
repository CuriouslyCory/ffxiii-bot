"""
Route recorder for handling route recording in both LANDMARK and HYBRID modes.
"""

import time
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from .route_manager import LANDMARK_DIR
from .constants import HYBRID_NODE_SAMPLE_INTERVAL, PHASE_CORR_CONFIDENCE_THRESHOLD


class LandmarkRecorder:
    """
    Handles landmark-based route recording.
    
    Step-based image capture workflow where user manually captures images
    for each step of the route.
    """
    
    def __init__(self, vision_engine):
        """
        Initialize the landmark recorder.
        
        Args:
            vision_engine: Vision engine for loading templates.
        """
        self.vision = vision_engine
        self.landmarks: List[Dict[str, Any]] = []
        self.current_step_images: List[Dict[str, str]] = []
    
    def start_recording(self):
        """Start a new landmark recording session."""
        self.landmarks = []
        self.current_step_images = []
        print("\n[RECORDING LANDMARK] Started.")
        print("  't': Capture image for current step.")
        print("  'n': Finish step and move to next.")
        print("  'g': Undo last captured image.")
        print("  'y': Finish recording.")
    
    def capture_image(self, image: np.ndarray) -> bool:
        """
        Capture an image for the current step.
        
        Args:
            image: Current screen image.
            
        Returns:
            True if image was captured successfully.
        """
        crop = self._get_center_crop(image)
        timestamp = int(time.time())
        suffix = np.random.randint(0, 1000)
        filename = f"landmark_{timestamp}_{suffix}.png"
        path = os.path.join(LANDMARK_DIR, filename)
        cv2.imwrite(path, crop)
        name = f"lm_{timestamp}_{suffix}"
        
        self.current_step_images.append({"name": name, "filename": filename})
        self.vision.load_template(name, path)
        print(f"[RECORDING] Captured image {len(self.current_step_images)} for current step.")
        return True
    
    def retake_last_image(self) -> bool:
        """
        Remove the last captured image from current step.
        
        Returns:
            True if an image was removed.
        """
        if self.current_step_images:
            removed = self.current_step_images.pop()
            print(f"[RECORDING] Removed image: {removed['name']}")
            return True
        else:
            print("[RECORDING] No images in current step to remove.")
            return False
    
    def finish_step(self) -> bool:
        """
        Finish the current step and move to next.
        
        Returns:
            True if step was finished (had images).
        """
        if not self.current_step_images:
            print("[RECORDING] Cannot finish step: No images captured. Press 't' to capture.")
            return False
        
        step_name = f"Step_{len(self.landmarks)}"
        self.landmarks.append({
            "name": step_name,
            "images": list(self.current_step_images)
        })
        print(f"[RECORDING] Finished Step {len(self.landmarks)-1} with {len(self.current_step_images)} images.")
        self.current_step_images = []
        return True
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get the recorded landmarks data."""
        return self.landmarks
    
    def _get_center_crop(self, image: np.ndarray) -> np.ndarray:
        """Get center crop of image for landmark capture."""
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_w, crop_h = 150, 150
        x = max(0, center_x - crop_w // 2)
        y = max(0, center_y - crop_h // 2)
        return image[y:y+crop_h, x:x+crop_w]


class HybridRecorder:
    """
    Handles hybrid visual odometry route recording.
    
    Automatic node sampling with odometry tracking. Nodes are sampled
    at regular intervals and include minimap images and world-space offsets.
    """
    
    # Configuration (imported from constants at module level)
    
    def __init__(self, navigator):
        """
        Initialize the hybrid recorder.
        
        Args:
            navigator: VisualNavigator instance for odometry.
        """
        self.navigator = navigator
        self.recording_nodes: List[Dict[str, Any]] = []
        self.last_node_time = 0.0
        self.last_recorded_minimap: Optional[np.ndarray] = None
    
    def start_recording(self):
        """Start a new hybrid recording session."""
        self.recording_nodes = []
        self.last_node_time = time.time()
        self.last_recorded_minimap = None
        print("\n[RECORDING HYBRID] Started.")
        print("  Move naturally. Breadcrumbs are collected automatically.")
        print("  'y': Finish recording.")
    
    def process_frame(self, image: np.ndarray) -> bool:
        """
        Process a frame during recording.
        
        Samples nodes at configured intervals.
        
        Args:
            image: Current screen image.
            
        Returns:
            True if a node was recorded this frame.
        """
        current_time = time.time()
        if current_time - self.last_node_time >= HYBRID_NODE_SAMPLE_INTERVAL:
            self.record_node(image)
            self.last_node_time = current_time
            return True
        return False
    
    def record_node(self, image: np.ndarray) -> bool:
        """
        Record a hybrid node.
        
        Args:
            image: Current screen image.
            
        Returns:
            True if node was recorded successfully.
        """
        timestamp = int(time.time() * 1000)
        mm_filename = f"hybrid_mm_{timestamp}.png"
        mm_path = os.path.join(LANDMARK_DIR, mm_filename)
        
        # Save the UNMASKED minimap ROI (stretched 400x520)
        roi_img = self.navigator.get_minimap_crop(image)
        if roi_img is None:
            print(f"[REC HYBRID] Failed to extract minimap for node {len(self.recording_nodes)}")
            return False
        
        cv2.imwrite(mm_path, roi_img)
        
        # 1. Absolute Orientation (Compass)
        curr_arrow_angle, curr_pivot = self.navigator.get_gold_arrow_angle(roi_img)
        
        # 2. Translation Odometry (World Space)
        world_offset = None
        if self.last_recorded_minimap is not None:
            prev_arrow_angle = self.recording_nodes[-1].get('arrow_angle', 0.0)
            default_pivot = (self.navigator.MINIMAP_CENTER_STRETCHED_X, 
                           self.navigator.MINIMAP_CENTER_STRETCHED_Y)
            prev_pivot = self.recording_nodes[-1].get('arrow_pivot', default_pivot)
            
            dx, dy, conf = self.navigator.compute_drift_pc(
                roi_img, self.last_recorded_minimap,
                curr_arrow_angle, curr_pivot,
                prev_arrow_angle, prev_pivot
            )
            
            if conf > PHASE_CORR_CONFIDENCE_THRESHOLD:
                world_offset = {"dx": dx, "dy": dy, "conf": conf}
                print(f"[REC HYBRID] Node {len(self.recording_nodes)} | "
                      f"World Offset: dx={dx:.1f}, dy={dy:.1f} | Arrow: {curr_arrow_angle:.1f}")
            else:
                print(f"[REC HYBRID] Low confidence World Offset (conf: {conf:.2f}). Skipping.")
        
        node = {
            "id": len(self.recording_nodes),
            "timestamp": timestamp,
            "minimap_path": mm_filename,
            "main_view_path": None,  # Optional for now
            "inputs": [],
            "relative_offset": world_offset,
            "arrow_angle": curr_arrow_angle,
            "arrow_pivot": curr_pivot
        }
        self.recording_nodes.append(node)
        self.last_recorded_minimap = roi_img
        print(f"[REC HYBRID] Saved Node {node['id']}")
        return True
    
    def finish_recording(self, navigator) -> Optional[str]:
        """
        Finish recording and generate master map.
        
        Args:
            navigator: VisualNavigator instance for master map generation.
            
        Returns:
            Path to master map image if generated, None otherwise.
        """
        master_map_path = None
        if self.recording_nodes:
            print("[MASTER MAP] Generating composite route map...")
            master_map = navigator.generate_master_map(self.recording_nodes, LANDMARK_DIR)
            if master_map is not None:
                timestamp = int(time.time())
                filename = f"master_map_{timestamp}.png"
                master_map_path = os.path.join(LANDMARK_DIR, filename)
                cv2.imwrite(master_map_path, master_map)
                print(f"[MASTER MAP] Saved composite map to {master_map_path}")
            else:
                print("[MASTER MAP] Failed to generate composite map.")
        
        return master_map_path
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get the recorded nodes data."""
        return self.recording_nodes


class RouteRecorder:
    """
    Unified interface for route recording.
    
    Delegates to LandmarkRecorder or HybridRecorder based on route type.
    """
    
    def __init__(self, vision_engine, navigator):
        """
        Initialize the route recorder.
        
        Args:
            vision_engine: Vision engine instance.
            navigator: VisualNavigator instance.
        """
        self.vision = vision_engine
        self.navigator = navigator
        self.landmark_recorder = LandmarkRecorder(vision_engine)
        self.hybrid_recorder = HybridRecorder(navigator)
        self.route_type: Optional[str] = None
    
    def start_recording(self, route_type: str):
        """
        Start recording a route.
        
        Args:
            route_type: "LANDMARK" or "HYBRID".
        """
        self.route_type = route_type
        if route_type == "HYBRID":
            self.hybrid_recorder.start_recording()
        else:
            self.landmark_recorder.start_recording()
    
    def process_frame(self, image: np.ndarray):
        """
        Process a frame during recording.
        
        Args:
            image: Current screen image.
        """
        if self.route_type == "HYBRID":
            self.hybrid_recorder.process_frame(image)
    
    def capture_image(self, image: np.ndarray) -> bool:
        """Capture an image (LANDMARK mode only)."""
        if self.route_type == "LANDMARK":
            return self.landmark_recorder.capture_image(image)
        return False
    
    def retake_last_image(self) -> bool:
        """Retake last image (LANDMARK mode only)."""
        if self.route_type == "LANDMARK":
            return self.landmark_recorder.retake_last_image()
        return False
    
    def finish_step(self) -> bool:
        """Finish current step (LANDMARK mode only)."""
        if self.route_type == "LANDMARK":
            return self.landmark_recorder.finish_step()
        return False
    
    def finish_recording(self) -> tuple:
        """
        Finish recording and return data.
        
        Returns:
            Tuple of (data, master_map_path).
        """
        if self.route_type == "LANDMARK":
            # Finish any pending step
            if self.landmark_recorder.current_step_images:
                self.landmark_recorder.finish_step()
            data = self.landmark_recorder.get_data()
            return data, None
        else:  # HYBRID
            master_map_path = self.hybrid_recorder.finish_recording(self.navigator)
            data = self.hybrid_recorder.get_data()
            return data, master_map_path
    
    def get_step_count(self) -> int:
        """Get current step count (LANDMARK mode only)."""
        if self.route_type == "LANDMARK":
            return len(self.landmark_recorder.current_step_images)
        return 0
