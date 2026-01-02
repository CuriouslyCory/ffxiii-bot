"""
Hostile detected state for combat engagement.
"""
import math
import time
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
    ROTATION_STRAFE_AMOUNT = 0.5  # Strafe amount for rotation (0.0-1.0)
    ROTATION_FORWARD_AMOUNT = 0.35  # Forward amount during rotation
    
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
    
    def is_active(self, image) -> bool:
        """
        Checks if hostile_detected state is active.
        
        Uses MinimapStateSensor to detect red minimap frame.
        
        :param image: Current screen capture.
        :return: True if hostile_detected state is detected.
        """
        # Enable sensor temporarily for detection
        self.minimap_state_sensor.enable()
        
        # Read minimap state
        state = self.minimap_state_sensor.read(image)
        
        # Disable sensor after reading (will be re-enabled in execute if needed)
        self.minimap_state_sensor.disable()
        
        return state == "hostile_detected"
    
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
        
        # Movement control: continuously turn towards nearest enemy
        if abs(angle_diff) > self.ACTION_ANGLE_THRESHOLD:
            # Need to rotate toward enemy
            # Calculate movement vector to rotate
            # Forward + Strafe Right = Rotate clockwise (positive angle_diff)
            # Forward + Strafe Left = Rotate counter-clockwise (negative angle_diff)
            
            # Scale movement based on angle difference (smaller diff = slower rotation)
            # Use proportion of angle_diff to max rotation needed (180 degrees)
            rotation_speed = min(1.0, abs(angle_diff) / 90.0)  # Scale to 0-1.0
            
            forward = self.ROTATION_FORWARD_AMOUNT * rotation_speed
            
            if angle_diff > 0:
                # Need to turn right (clockwise) - strafe right + forward
                strafe = self.ROTATION_STRAFE_AMOUNT * rotation_speed
                self.movement_skill.move(strafe, forward)
            else:
                # Need to turn left (counter-clockwise) - strafe left + forward
                strafe = -self.ROTATION_STRAFE_AMOUNT * rotation_speed
                self.movement_skill.move(strafe, forward)

            print(f"[HOSTILE] Rotation speed: {rotation_speed:.2f}, Forward: {forward:.2f}, Strafe: {strafe:.2f}")
        else:
            # Within action angle threshold - move forward towards enemy
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
