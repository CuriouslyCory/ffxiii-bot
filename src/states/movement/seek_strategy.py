"""
Seek strategy for handling recovery when visual tracking is lost.

Handles coasting logic, reverse rotation, retry attempts, and timeout.
"""

import time
from typing import Tuple, Optional, Dict, Any
from .constants import (
    COAST_DURATION, COAST_DURATION_EXTENDED, COAST_TURNING_THRESHOLD,
    COAST_FORWARD_THRESHOLD, RETRY_DURATION, RETRY_ATTEMPTS,
    RETRY_SCALE_START, RETRY_SCALE_DECREMENT, RETRY_SCALE_MIN
)


class SeekStrategy:
    """
    Handles seek/recovery logic when tracking is lost.
    
    Responsibilities:
    - Coasting logic (hold last valid input)
    - Reverse rotation phase
    - Retry attempts with scaled controls
    - Timeout handling
    """
    
    # Configuration parameters (imported from constants at module level)
    # These are used as class attributes
    
    def __init__(self):
        """Initialize the seek strategy."""
        self.last_seen_time = 0.0
        self.last_valid_controls: Optional[Dict[str, float]] = None
    
    def update_last_seen(self):
        """Update the last seen timestamp."""
        self.last_seen_time = time.time()
    
    def set_last_valid_controls(self, controls: Dict[str, float]):
        """Store the last valid control state."""
        self.last_valid_controls = controls.copy()
    
    def process_seek(self, current_time: float, 
                    controller_state: Dict[str, Any]) -> Tuple[float, float, float, float, str]:
        """
        Process seek/recovery logic when tracking is lost.
        
        Args:
            current_time: Current timestamp.
            controller_state: Current controller state (fallback if no last_valid_controls).
            
        Returns:
            Tuple of (cam_x, cam_y, move_x, move_y, status_msg).
        """
        time_since_last_seen = current_time - self.last_seen_time
        
        # Retrieve last valid controls (or current if none stored)
        last_ctrl = self.last_valid_controls or controller_state
        last_rx = last_ctrl.get('rx', 0.0)
        last_ly = last_ctrl.get('ly', 0.0)
        last_ry = last_ctrl.get('ry', 0.0)
        last_lx = last_ctrl.get('lx', 0.0)
        
        # Determine coast duration
        coast_duration = COAST_DURATION
        if (abs(last_rx) > COAST_TURNING_THRESHOLD and 
            abs(last_ly) < COAST_FORWARD_THRESHOLD):
            coast_duration = COAST_DURATION_EXTENDED
        
        retry_duration = RETRY_DURATION
        
        if time_since_last_seen < coast_duration:
            # Phase 1: Reverse Rotation, Hold Forward
            # Reverse camera pan to return towards last known readable orientation
            cam_x = -last_rx
            cam_y = last_ry
            
            # Hold character movement
            move_x = last_lx
            move_y = last_ly
            
            return cam_x, cam_y, move_x, move_y, "REVERSING"
            
        elif time_since_last_seen < coast_duration + retry_duration * RETRY_ATTEMPTS:
            # Phase 2: Recovery / Retry (Slower Spin)
            retry_time = time_since_last_seen - coast_duration
            attempt = int(retry_time // retry_duration) + 1  # 1, 2, 3
            
            # Slow down factor: progressively slower
            scale = max(RETRY_SCALE_MIN, 
                       RETRY_SCALE_START - (RETRY_SCALE_DECREMENT * attempt))
            
            # Apply scaled controls based on LAST VALID inputs
            cam_x = last_rx * scale
            cam_y = last_ry * scale
            
            # Stop character movement to avoid walking blindly
            move_x = 0.0
            move_y = 0.0
            
            return cam_x, cam_y, move_x, move_y, f"RETRY {attempt}"
            
        else:
            # Phase 3: Give Up
            if time_since_last_seen < (coast_duration + retry_duration * RETRY_ATTEMPTS + 0.5):
                print(f"[PLAYBACK] Lost visual tracking for {time_since_last_seen:.2f}s. Stopping.")
            
            return 0.0, 0.0, 0.0, 0.0, "LOST"
    
    def should_give_up(self, current_time: float) -> bool:
        """
        Check if we should give up seeking.
        
        Args:
            current_time: Current timestamp.
            
        Returns:
            True if we should give up.
        """
        time_since_last_seen = current_time - self.last_seen_time
        coast_duration = COAST_DURATION
        retry_duration = RETRY_DURATION
        
        return time_since_last_seen >= (coast_duration + retry_duration * RETRY_ATTEMPTS)
    
    def reset(self):
        """Reset seek state."""
        self.last_seen_time = 0.0
        self.last_valid_controls = None
