"""
Navigation controller for implementing navigation control algorithms.

Handles drift smoothing, PID-like control, movement control, and arrival detection.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
from .constants import (
    EMA_ALPHA_DX_DY, EMA_WINDOW_SIZE_DX_DY, EMA_ALPHA_ANGLE, EMA_WINDOW_SIZE_ANGLE,
    CAMERA_ROTATION_KP, CAMERA_ROTATION_MAX,
    CAMERA_ROTATION_MIN_THRESHOLD, CAMERA_ROTATION_MIN_BOOST,
    STRAFE_KP, STRAFE_MAX, APPROACH_SLOWDOWN_DISTANCE, APPROACH_SLOWDOWN_MIN,
    ARRIVAL_DISTANCE_THRESHOLD, ARRIVAL_ANGLE_THRESHOLD, ARRIVAL_BUFFER_SIZE
)


class NavigationController:
    """
    Implements navigation control algorithms for visual odometry navigation.
    
    Responsibilities:
    - Drift smoothing (EMA with circular mean for angles)
    - PID-like control for camera rotation
    - Movement control (forward/back, strafe)
    - Arrival detection (distance + angle thresholds)
    - Approach distance tracking for slowdown
    """
    
    # Configuration parameters (imported from constants at module level)
    # These are used as class attributes
    
    def __init__(self):
        """Initialize the navigation controller."""
        # Rolling window buffers for EMA
        self._dx_buffer: deque = deque(maxlen=EMA_WINDOW_SIZE_DX_DY)
        self._dy_buffer: deque = deque(maxlen=EMA_WINDOW_SIZE_DX_DY)
        self._angle_buffer: deque = deque(maxlen=EMA_WINDOW_SIZE_ANGLE)
        
        # Current smoothed values
        self._smoothed_dx: Optional[float] = None
        self._smoothed_dy: Optional[float] = None
        self._smoothed_angle: Optional[float] = None
        
        self._approach_dist: Optional[float] = None
    
    def smooth_drift(self, dx: float, dy: float, angle: float) -> Tuple[float, float, float]:
        """
        Apply EMA smoothing to drift values using rolling window buffers.
        
        Uses exponential moving average for dx/dy and circular mean for angle
        to handle +/- 180 wrap-around. Each uses separate alpha and window size.
        The EMA is calculated only over the values in the rolling window buffer.
        
        Args:
            dx: Raw x drift.
            dy: Raw y drift.
            angle: Raw angle drift in degrees.
            
        Returns:
            Tuple of (smoothed_dx, smoothed_dy, smoothed_angle).
        """
        # Add new values to buffers (automatically drops oldest when full)
        self._dx_buffer.append(dx)
        self._dy_buffer.append(dy)
        self._angle_buffer.append(angle)
        
        # Calculate EMA for dx over the window
        if len(self._dx_buffer) == 1:
            self._smoothed_dx = dx
        else:
            # Recalculate EMA over the entire window for accuracy
            # This ensures we only use values within the window
            alpha_dx = EMA_ALPHA_DX_DY
            smoothed = self._dx_buffer[0]
            for val in list(self._dx_buffer)[1:]:
                smoothed = alpha_dx * val + (1 - alpha_dx) * smoothed
            self._smoothed_dx = smoothed
        
        # Calculate EMA for dy over the window
        if len(self._dy_buffer) == 1:
            self._smoothed_dy = dy
        else:
            alpha_dy = EMA_ALPHA_DX_DY
            smoothed = self._dy_buffer[0]
            for val in list(self._dy_buffer)[1:]:
                smoothed = alpha_dy * val + (1 - alpha_dy) * smoothed
            self._smoothed_dy = smoothed
        
        # Calculate EMA for angle using circular mean over the window
        if len(self._angle_buffer) == 1:
            self._smoothed_angle = angle
        else:
            alpha_angle = EMA_ALPHA_ANGLE
            # Start with first angle
            smoothed_angle = self._angle_buffer[0]
            
            # Apply EMA to each subsequent angle in the buffer
            for ang in list(self._angle_buffer)[1:]:
                a_rad = np.radians(ang)
                prev_a_rad = np.radians(smoothed_angle)
                
                # EMA on sin/cos components
                s_sin = alpha_angle * np.sin(a_rad) + (1 - alpha_angle) * np.sin(prev_a_rad)
                s_cos = alpha_angle * np.cos(a_rad) + (1 - alpha_angle) * np.cos(prev_a_rad)
                
                # Reconstruct angle
                smoothed_angle = np.degrees(np.arctan2(s_sin, s_cos))
            
            self._smoothed_angle = smoothed_angle
        
        return (self._smoothed_dx, self._smoothed_dy, self._smoothed_angle)
    
    def check_arrival(self, dx: float, dy: float, angle: float, 
                     arrival_buffer: Optional[list] = None) -> Tuple[bool, float, float]:
        """
        Check if we have arrived at the target node.
        
        Uses averaged values from arrival buffer if provided to avoid jumpy triggers.
        
        Args:
            dx: X drift.
            dy: Y drift.
            angle: Angle drift in degrees.
            arrival_buffer: Optional list of (dx, dy, angle) tuples for averaging.
            
        Returns:
            Tuple of (has_arrived, distance, mean_angle).
        """
        # Use averaged values if buffer provided
        mean_dx = dx
        mean_dy = dy
        mean_angle = angle
        
        if arrival_buffer:
            n = len(arrival_buffer)
            mean_dx = sum(v[0] for v in arrival_buffer) / n
            mean_dy = sum(v[1] for v in arrival_buffer) / n
            # Circular mean for angle
            s_sin = sum(np.sin(np.radians(v[2])) for v in arrival_buffer) / n
            s_cos = sum(np.cos(np.radians(v[2])) for v in arrival_buffer) / n
            mean_angle = np.degrees(np.arctan2(s_sin, s_cos))
        
        # Calculate distance
        dist = np.sqrt(mean_dx * mean_dx + mean_dy * mean_dy)
        
        # Check thresholds
        has_arrived = (abs(dist) < ARRIVAL_DISTANCE_THRESHOLD and 
                      abs(mean_angle) < ARRIVAL_ANGLE_THRESHOLD)
        
        return has_arrived, dist, mean_angle
    
    def compute_controls(self, sdx: float, sdy: float, sangle: float, 
                       approach_dist: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Compute control commands from smoothed drift values.
        
        Args:
            sdx: Smoothed x drift.
            sdy: Smoothed y drift.
            sangle: Smoothed angle drift in degrees.
            approach_dist: Optional approach distance for slowdown.
            
        Returns:
            Tuple of (cam_x, cam_y, move_x, move_y).
        """
        # 1. Camera Control (Pan Left/Right to fix Angle)
        # Positive angle means current is rotated CW relative to target
        # To correct, we rotate CCW (camera left = negative X)
        cam_x = sangle * CAMERA_ROTATION_KP
        
        # Clamp to max speed
        cam_x = max(-CAMERA_ROTATION_MAX, min(CAMERA_ROTATION_MAX, cam_x))
        
        # Min speed boost: If absolute value is too small but not zero, boost it
        if abs(cam_x) < CAMERA_ROTATION_MIN_THRESHOLD:
            cam_x = 0.0
        elif abs(cam_x) < CAMERA_ROTATION_MIN_BOOST:
            cam_x = CAMERA_ROTATION_MIN_BOOST if cam_x > 0 else -CAMERA_ROTATION_MIN_BOOST
        
        cam_y = 0.0  # No vertical camera movement for now
        
        # 2. Movement Control (Forward/Back + Strafe)
        move_x = 0.0
        move_y = 0.0
        
        if abs(sangle) < 180.0:  # Allow movement while still correcting
            # Forward Speed (y)
            # Drive stick forward when reasonably aligned
            if abs(sdy) > 0:
                move_y = 1.0 * (sdy / abs(sdy))
            else:
                move_y = 1.0  # Default forward
            
            # Slow down as we get very close to the node
            if approach_dist is not None and approach_dist < APPROACH_SLOWDOWN_DISTANCE:
                slowdown = max(APPROACH_SLOWDOWN_MIN, 
                             approach_dist / APPROACH_SLOWDOWN_DISTANCE)
                move_y *= slowdown
            
            # Strafe correction
            move_x = sdx * STRAFE_KP
            move_x = max(-STRAFE_MAX, min(STRAFE_MAX, move_x))
        else:
            # Angle too large, stop moving and just rotate
            move_y = 0.0
            move_x = 0.0
        
        return cam_x, cam_y, move_x, move_y
    
    def set_approach_dist(self, dist: Optional[float]):
        """Set the current approach distance."""
        self._approach_dist = dist
    
    def get_approach_dist(self) -> Optional[float]:
        """Get the current approach distance."""
        return self._approach_dist
    
    def reset(self):
        """Reset internal state."""
        self._dx_buffer.clear()
        self._dy_buffer.clear()
        self._angle_buffer.clear()
        self._smoothed_dx = None
        self._smoothed_dy = None
        self._smoothed_angle = None
        self._approach_dist = None
