"""
Hostile detected state for combat engagement.
"""
import math
import time
import cv2
import numpy as np
from src.states.base import State
from src.sensors.minimap_state import MinimapStateSensor
from src.sensors.enemy_position import EnemyPositionSensor
from src.sensors.player_direction import PlayerDirectionSensor
from src.skills.movement import MovementSkill
from src.skills.button import ButtonTapSkill


class HostileDetectedState(State):
    """
    HostileDetectedState handles logic when enemies are detected nearby.
    
    Detected when the minimap frame is red, indicating enemies are nearby.
    Orients the player, turns toward the nearest enemy, and engages in combat.
    """
    
    # Configuration constants
    ACTION_ANGLE_THRESHOLD = 15.0  # Degrees threshold for pressing action button
    MOVEMENT_ANGLE_THRESHOLD = 5.0  # Minimum angle difference to apply movement
    ORIENTATION_FORWARD_DURATION = 0.1  # Duration to press forward for orientation
    ORIENTATION_STABILIZE_TIME = 0.2  # Wait time after orientation
    ACTION_COOLDOWN = 1.0  # Minimum time between action button presses (attack animation duration)
    ROTATION_CAMERA_PAN = 0.55  # Camera pan amount for rotation (~0.45)
    ROTATION_FORWARD_AMOUNT = -0.75  # Forward movement amount during rotation (~-0.35)
    ROTATION_FORWARD_DURATION = 0.25  # Duration to move forward during rotation (0.25 seconds)
    ROTATION_FORWARD_INTERVAL = 0.5  # Interval between forward movement pulses (0.5 seconds)
    
    def __init__(self, manager):
        """Initialize hostile detected state."""
        super().__init__(manager)
        
        # Initialize sensors
        self.minimap_state_sensor = MinimapStateSensor(manager.roi_cache)
        self.enemy_position_sensor = EnemyPositionSensor(manager.roi_cache)
        self.player_direction_sensor = PlayerDirectionSensor(manager.roi_cache)
        
        # Initialize skills
        self.movement_skill = MovementSkill(self.controller)
        self.tap_skill = ButtonTapSkill(self.controller)
        
        # State tracking
        self._oriented = False
        self._last_orientation_time = 0.0
        self._last_action_time = 0.0
        self._rotation_start_time = 0.0
        self._rotation_active = False
        self._rotation_camera_direction = 0.0  # Camera pan direction (-1.0 to 1.0)
        self._last_forward_pulse_time = 0.0  # Time of last forward movement pulse start
    
    def is_active(self, image) -> bool:
        """
        Checks if hostile_detected state is active.
        
        Uses both MinimapStateSensor (red minimap frame) and hostile clock template
        matching as detection methods. Returns True if either method detects hostile state.
        
        :param image: Current screen capture.
        :return: True if hostile_detected state is detected.
        """
        # Method 1: Check minimap state sensor (red minimap frame)
        self.minimap_state_sensor.enable()
        state = self.minimap_state_sensor.read(image)
        self.minimap_state_sensor.disable()
        
        if state == "hostile_detected":
            return True
        
        # Method 2: Check for hostile clock template in bottom center ROI
        if self.manager.roi_cache.has_roi("hostile_clock"):
            hostile_clock_roi = self.manager.roi_cache.get_roi("hostile_clock", image)
            if hostile_clock_roi is not None:
                # Check if hostile_clock template is available and match it
                if hasattr(self.manager.vision, 'templates') and "hostile_clock" in self.manager.vision.templates:
                    template = self.manager.vision.templates["hostile_clock"]
                    
                    # Convert ROI to grayscale (pre-filter)
                    if len(hostile_clock_roi.shape) == 3:
                        roi_gray = cv2.cvtColor(hostile_clock_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        roi_gray = hostile_clock_roi
                    
                    # Convert template to grayscale if needed
                    if len(template.shape) == 3:
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    else:
                        template_gray = template
                    
                    # Check if template is larger than ROI
                    if template_gray.shape[0] > roi_gray.shape[0] or template_gray.shape[1] > roi_gray.shape[1]:
                        return False
                    
                    # Perform template matching on grayscale images
                    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # Check if match exceeds threshold
                    if max_val >= 0.45:
                        return True
        
        return False
    
    def execute(self, image):
        """
        Executes hostile detected state instructions.
        
        Orients player with camera, then turns toward nearest enemy and engages.
        """
        current_time = time.time()
        
        # Enable sensors
        self.minimap_state_sensor.enable()
        self.enemy_position_sensor.enable()
        self.player_direction_sensor.enable()
        
        # Orientation phase: align player arrow with camera direction
        if not self._oriented:
            # Press forward briefly to orient player with camera
            # Use movement skill to press forward (y > 0)
            print("[HOSTILE] Starting orientation phase - pressing forward")
            self.movement_skill.move(0.0, -1.0)  # Forward
            self._last_orientation_time = current_time
            self._oriented = True
            return  # Wait for next frame to stabilize
        
        # Wait for orientation to stabilize, then release forward
        elapsed = current_time - self._last_orientation_time
        if elapsed < self.ORIENTATION_STABILIZE_TIME:
            # Keep forward pressed during stabilization
            print(f"[HOSTILE] Orientation stabilizing... {elapsed:.2f}s / {self.ORIENTATION_STABILIZE_TIME:.2f}s")
            return  # Still stabilizing
        elif elapsed < self.ORIENTATION_STABILIZE_TIME + 0.1:
            # Release forward movement after stabilization
            print("[HOSTILE] Orientation complete - releasing forward")
            self.movement_skill.stop()
            return
        
        # Read sensors
        enemy_pos = self.enemy_position_sensor.read(image)
        player_dir = self.player_direction_sensor.read(image)
        
        # If either sensor returns None, skip this frame
        if enemy_pos is None:
            print("[HOSTILE] Enemy position sensor returned None")
            return
        if player_dir is None:
            print("[HOSTILE] Player direction sensor returned None")
            return
        
        dx, dy = enemy_pos
        
        # Calculate target angle from enemy position
        # In minimap coordinates (north-up):
        # - dx > 0 = East (90°)
        # - dx < 0 = West (270°)
        # - dy < 0 = North (0°) - negative because image Y increases down
        # - dy > 0 = South (180°)
        # Use atan2(dx, -dy) to get angle where 0° = North, 90° = East
        target_angle_rad = math.atan2(dx, -dy)
        target_angle = math.degrees(target_angle_rad)
        
        # Normalize to [0, 360)
        if target_angle < 0:
            target_angle += 360
        
        # Calculate angle difference
        angle_diff = target_angle - player_dir
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        # Log enemy position and angle diff each frame
        print(f"[HOSTILE] Enemy pos: dx={dx}, dy={dy}, Player dir: {player_dir:.1f}°, Target: {target_angle:.1f}°, Angle diff: {angle_diff:.1f}°")
        
        # Movement control: rotate towards nearest enemy using camera panning + forward movement
        if abs(angle_diff) > self.ACTION_ANGLE_THRESHOLD:
            # Need to rotate toward enemy
            # Determine camera pan direction: positive angle_diff = turn right (pan camera right/+)
            # Negative angle_diff = turn left (pan camera left/-)
            camera_direction = 1.0 if angle_diff > 0 else -1.0
            camera_pan = self.ROTATION_CAMERA_PAN * camera_direction
            
            # Start rotation if not already active or if direction changed
            if not self._rotation_active or self._rotation_camera_direction != camera_direction:
                print(f"[HOSTILE] Starting rotation - Camera pan: {camera_pan:.2f}, Angle diff: {angle_diff:.1f}°")
                self._rotation_active = True
                self._rotation_start_time = current_time
                self._rotation_camera_direction = camera_direction
                self._last_forward_pulse_time = current_time
                # Start camera panning
                self.controller.move_camera(camera_pan, 0.0)
                # Start forward movement
                self.movement_skill.move(0.0, self.ROTATION_FORWARD_AMOUNT)
            
            # Continue panning camera continuously
            self.controller.move_camera(camera_pan, 0.0)
            
            # Check if it's time for a new forward movement pulse
            time_since_last_pulse = current_time - self._last_forward_pulse_time
            
            # Start a new forward pulse if interval has elapsed
            if time_since_last_pulse >= self.ROTATION_FORWARD_INTERVAL:
                print(f"[HOSTILE] Starting forward pulse - Camera pan: {camera_pan:.2f}, Angle diff: {angle_diff:.1f}°")
                self._last_forward_pulse_time = current_time
                time_since_last_pulse = 0.0
            
            # Calculate time in current pulse
            time_in_current_pulse = current_time - self._last_forward_pulse_time
            
            # Check if we're still in the current forward pulse duration
            if time_in_current_pulse < self.ROTATION_FORWARD_DURATION:
                # Still in forward movement pulse
                self.movement_skill.move(0.0, self.ROTATION_FORWARD_AMOUNT)
                print(f"[HOSTILE] Rotation in progress - Camera pan: {camera_pan:.2f}, Forward: {self.ROTATION_FORWARD_AMOUNT:.2f}, Pulse elapsed: {time_in_current_pulse:.2f}s, Angle diff: {angle_diff:.1f}°")
            else:
                # Forward pulse duration elapsed, stop forward movement but keep camera panning
                self.movement_skill.stop()
                print(f"[HOSTILE] Rotation in progress - Camera pan: {camera_pan:.2f}, Forward stopped, Angle diff: {angle_diff:.1f}°")
        else:
            # Within action angle threshold - stop rotation and move forward towards enemy
            if self._rotation_active:
                print(f"[HOSTILE] Rotation complete - aligned with enemy (Angle diff: {angle_diff:.1f}°)")
                self.controller.move_camera(0.0, 0.0)
                self.movement_skill.stop()
                self._rotation_active = False
                self._rotation_camera_direction = 0.0
            
            # Move forward towards enemy
            self.movement_skill.move(0.0, -1.0)  # Move forward
        
        # Attack when facing enemy (within threshold) and cooldown has passed
        if abs(angle_diff) <= self.ACTION_ANGLE_THRESHOLD:
            time_since_last_action = current_time - self._last_action_time
            if time_since_last_action >= self.ACTION_COOLDOWN:
                print(f"[HOSTILE] ATTACK! Time since last: {time_since_last_action:.2f}s, Angle diff: {angle_diff:.1f}°")
                self.tap_skill.tap("gamepad_a", duration=0.2)
                self._last_action_time = current_time
            else:
                print(f"[HOSTILE] Attack on cooldown. Time since last: {time_since_last_action:.2f}s (need {self.ACTION_COOLDOWN:.1f}s)")
    
    def on_enter(self):
        """Called when entering hostile detected state."""
        super().on_enter()
        print("--- Hostile Detected State ---")
        self.minimap_state_sensor.enable()
        self.enemy_position_sensor.enable()
        self.player_direction_sensor.enable()
    
    def on_exit(self):
        """Called when exiting hostile detected state."""
        super().on_exit()
        
        # Disable sensors
        self.minimap_state_sensor.disable()
        self.enemy_position_sensor.disable()
        self.player_direction_sensor.disable()
        
        # Stop all movement and camera
        self.movement_skill.stop()
        self.controller.move_camera(0.0, 0.0)
        self.controller.move_character(0.0, 0.0)
        
        # Release all buttons
        self.controller.release_all()
        
        # Reset state tracking
        self._oriented = False
        self._last_orientation_time = 0.0
        self._last_action_time = 0.0
        self._rotation_start_time = 0.0
        self._rotation_active = False
        self._rotation_camera_direction = 0.0
        self._last_forward_pulse_time = 0.0
