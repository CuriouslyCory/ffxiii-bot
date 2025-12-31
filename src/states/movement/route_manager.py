"""
Route manager for handling route loading, selection, and database operations.
"""

import os
from typing import Optional, List, Tuple, Dict, Any
from src.db import (
    init_db, save_route, update_route_structure, update_hybrid_route_structure, load_route, 
    list_routes, set_active_route, get_active_route, clear_active_route
)

LANDMARK_DIR = "templates/landmarks"


class RouteManager:
    """
    Centralizes route loading, selection, and database operations.
    
    Handles:
    - Route listing and selection UI
    - Loading routes from database
    - Active route state persistence
    - Route structure updates (add/delete images, steps)
    - Template loading for vision engine
    """
    
    def __init__(self, vision_engine):
        """
        Initialize the route manager.
        
        Args:
            vision_engine: Vision engine instance for loading templates.
        """
        self.vision = vision_engine
        self.available_routes: List[Tuple] = []
        
        # Ensure landmark directory exists
        os.makedirs(LANDMARK_DIR, exist_ok=True)
        init_db()
    
    def list_available_routes(self) -> List[Tuple]:
        """
        List all available routes from the database.
        
        Returns:
            List of route tuples (id, name, created_at, type).
        """
        self.available_routes = list_routes()
        return self.available_routes
    
    def load_route(self, route_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a route from the database.
        
        Args:
            route_id: The route ID to load.
            
        Returns:
            Route data dictionary or None if not found.
        """
        route_data = load_route(route_id)
        if not route_data:
            return None
        
        # Load templates for landmark routes
        route_type = route_data.get('type', 'LANDMARK')
        if route_type == "LANDMARK":
            landmarks = route_data.get('landmarks', [])
            for step in landmarks:
                for img in step.get('images', []):
                    path = os.path.join(LANDMARK_DIR, img['filename'])
                    if os.path.exists(path):
                        self.vision.load_template(img['name'], path)
        
        return route_data
    
    def select_route(self, index: int, current_route_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Select a route by index from available routes.
        
        Args:
            index: Route index (1-9) from the list.
            current_route_id: Current active route ID for resume check.
            
        Returns:
            Tuple of (route_data, start_idx) or None if invalid selection.
        """
        if not (0 <= index - 1 < len(self.available_routes)):
            return None
        
        route_id = self.available_routes[index - 1][0]
        route_data = self.load_route(route_id)
        
        if not route_data:
            return None
        
        # Check if this route is currently in-progress
        start_idx = 0
        active = get_active_route()
        
        if active and str(active['route_id']) == str(route_id):
            # Route is in progress - caller should handle resume dialog
            start_idx = int(active['current_idx'])
        
        return {
            'route_data': route_data,
            'start_idx': start_idx,
            'route_id': route_id
        }
    
    def save_route(self, name: str, data: List[Dict], route_type: str, master_map_path: Optional[str] = None) -> bool:
        """
        Save a route to the database.
        
        Args:
            name: Route name.
            data: Route data (nodes or landmarks).
            route_type: Route type ("LANDMARK", "HYBRID", "KEYLOG").
            master_map_path: Optional path to master map image.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        return save_route(name, data, route_type, master_map_path=master_map_path)
    
    def update_route_structure(self, route_id: int, landmarks: List[Dict]) -> bool:
        """
        Update the structure of a landmark route.
        
        Args:
            route_id: Route ID to update.
            landmarks: Updated landmarks/steps data.
            
        Returns:
            True if updated successfully, False otherwise.
        """
        return update_route_structure(route_id, landmarks)
    
    def update_hybrid_route_structure(self, route_id: int, nodes: List[Dict]) -> bool:
        """
        Update the structure of a hybrid route.
        
        Args:
            route_id: Route ID to update.
            nodes: Updated nodes data.
            
        Returns:
            True if updated successfully, False otherwise.
        """
        return update_hybrid_route_structure(route_id, nodes)
    
    def set_active_route(self, route_id: int, current_idx: int):
        """Set the active route state."""
        set_active_route(route_id, current_idx)
    
    def get_active_route(self) -> Optional[Dict]:
        """Get the currently active route state."""
        return get_active_route()
    
    def clear_active_route(self):
        """Clear the active route state."""
        clear_active_route()
    
    def delete_image_from_step(self, landmarks: List[Dict], step_idx: int, 
                               current_landmark_idx: int, 
                               current_match_img_index: Optional[int] = None,
                               next_match_img_index: Optional[int] = None) -> bool:
        """
        Delete an image from a step.
        
        Args:
            landmarks: List of landmarks/steps.
            step_idx: Index of step to modify.
            current_landmark_idx: Current landmark index.
            current_match_img_index: Current matched image index.
            next_match_img_index: Next matched image index.
            
        Returns:
            True if image was deleted, False if step should be deleted instead.
        """
        if not landmarks or not (0 <= step_idx < len(landmarks)):
            return False
        
        step = landmarks[step_idx]
        images = step.get('images', [])
        
        if not images:
            return False  # Signal to delete step
        
        # Determine which image to delete
        target_img_idx = -1
        
        if step_idx == current_landmark_idx and current_match_img_index is not None:
            target_img_idx = current_match_img_index
        elif (next_match_img_index is not None and 
              step_idx == (current_landmark_idx + 1) % len(landmarks)):
            target_img_idx = next_match_img_index
        
        if target_img_idx != -1 and 0 <= target_img_idx < len(images):
            deleted = images.pop(target_img_idx)
            print(f"[DELETE] Removed image '{deleted['name']}' from step {step_idx}.")
        else:
            deleted = images.pop()
            print(f"[DELETE] Removed last image '{deleted['name']}' from step {step_idx} (fallback).")
        
        return len(images) > 0  # True if step still has images
    
    def delete_step(self, landmarks: List[Dict], step_idx: int) -> Dict[str, Any]:
        """
        Delete a step from landmarks.
        
        Args:
            landmarks: List of landmarks/steps.
            step_idx: Index of step to delete.
            
        Returns:
            Dictionary with update info: {'deleted': bool, 'new_current_idx': int}
        """
        if not (0 <= step_idx < len(landmarks)):
            return {'deleted': False, 'new_current_idx': None}
        
        deleted = landmarks.pop(step_idx)
        print(f"[DELETE] Removed step {step_idx}: {deleted.get('name', 'Unknown')}")
        
        # Calculate new current index
        new_current_idx = None
        # Note: Caller should handle current_idx adjustment
        
        return {
            'deleted': True,
            'new_current_idx': new_current_idx,
            'step_idx': step_idx
        }
