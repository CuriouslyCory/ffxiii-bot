"""
Constants and configuration values for movement state.
"""

# Directory paths
LANDMARK_DIR = "templates/landmarks"

# Hybrid Recording
HYBRID_NODE_SAMPLE_INTERVAL = 0.25  # seconds
PHASE_CORR_CONFIDENCE_THRESHOLD = 0.05

# Navigation Control - EMA Smoothing
# Separate configuration for dx/dy vs angle
EMA_ALPHA_DX_DY = 0.75  # EMA alpha for dx and dy smoothing
EMA_WINDOW_SIZE_DX_DY = 30  # Number of samples in rolling window for dx/dy
EMA_ALPHA_ANGLE = 0.25  # EMA alpha for angle smoothing
EMA_WINDOW_SIZE_ANGLE = 30  # Number of samples in rolling window for angle

# Legacy constant (deprecated, use EMA_ALPHA_DX_DY)
SMOOTHING_ALPHA = 0.25

CAMERA_ROTATION_KP = 0.03
CAMERA_ROTATION_MAX = 0.35
CAMERA_ROTATION_MIN_THRESHOLD = 0.05
CAMERA_ROTATION_MIN_BOOST = 0.2
STRAFE_KP = 0.01
STRAFE_MAX = 0.35
APPROACH_SLOWDOWN_DISTANCE = 40
APPROACH_SLOWDOWN_MIN = 0.25

# Seek/Recovery
COAST_DURATION = 1.0
COAST_DURATION_EXTENDED = 1.5  # When turning
COAST_TURNING_THRESHOLD = 0.3
COAST_FORWARD_THRESHOLD = 0.2
RETRY_DURATION = 1.0
RETRY_ATTEMPTS = 3
RETRY_SCALE_START = 0.8
RETRY_SCALE_DECREMENT = 0.2
RETRY_SCALE_MIN = 0.2

# Lookahead
LOOKAHEAD_DEPTH = 3

# Arrival detection (shared with navigator)
ARRIVAL_DISTANCE_THRESHOLD = 15
ARRIVAL_ANGLE_THRESHOLD = 45
ARRIVAL_BUFFER_SIZE = 8

# Landmark Playback
LANDMARK_SEEK_TIMEOUT = 300.0  # 5 minutes
LANDMARK_SEEK_TOGGLE_INTERVAL = 4.5
LANDMARK_SEEK_PAN_SPEED = 0.15
LANDMARK_SEEK_TILT_SPEED = 0.15
LANDMARK_TEMPLATE_THRESHOLD = 0.87
LANDMARK_HIGH_CONFIDENCE_THRESHOLD = 0.95
LANDMARK_CENTER_CROP_SIZE = 150
LANDMARK_DEADZONE = 50
LANDMARK_MAX_OFFSET_X_FACTOR = 0.25  # fraction of width
LANDMARK_MAX_OFFSET_Y_FACTOR = 0.33  # fraction of height
LANDMARK_MAX_SPEED = 0.35
LANDMARK_MOVE_THRESHOLD = 200  # pixels
LANDMARK_COAST_DURATION = 2.0  # seconds
LANDMARK_SEARCH_START_DELAY = 3.0  # seconds

# Advanced Map Stitching Configuration
# Enable/disable advanced stitching features
ENABLE_ADVANCED_STITCHING = True  # Set to False to use legacy ORB-based method
ENABLE_MESH_WARPING = False  # Enable APAP mesh-warping for parallax handling

# SuperPoint Feature Extraction
SUPERPOINT_N_FEATURES = 512  # Maximum features per image
SUPERPOINT_DEVICE = None  # None for auto-detect, 'cuda' or 'cpu' to force

# LightGlue Feature Matching
LIGHTGLUE_FILTER_THRESHOLD = 0.2  # Minimum match confidence (0-1)
LIGHTGLUE_DEVICE = None  # None for auto-detect, 'cuda' or 'cpu' to force

# MAGSAC++ Robust Estimation
MAGSAC_METHOD = 'homography'  # 'homography' or 'affine'
MAGSAC_THRESHOLD = 5.0  # Inlier threshold in pixels
MAGSAC_CONFIDENCE = 0.99  # Required confidence level (0-1)
MAGSAC_MAX_ITERS = 2000  # Maximum RANSAC iterations

# Bundle Adjustment
BUNDLE_ADJUSTMENT_MAX_ITERS = 100  # Maximum optimization iterations
BUNDLE_ADJUSTMENT_FTOL = 1e-6  # Function tolerance
BUNDLE_ADJUSTMENT_XTOL = 1e-8  # Parameter tolerance
BUNDLE_ADJUSTMENT_GTOL = 1e-6  # Gradient tolerance

# Mesh Warping (APAP)
MESH_WARPING_GRID_SIZE = 10  # Grid cells per dimension for mesh warping
MESH_WARPING_BLEND_RADIUS = 5  # Blending radius in pixels

# Random Movement Mode
RANDOM_TURN_INTERVAL_MIN = 1.0  # Minimum seconds between turns
RANDOM_TURN_INTERVAL_MAX = 10.0  # Maximum seconds between turns
RANDOM_TURN_DURATION_MIN = 0.5  # Minimum seconds for a turn
RANDOM_TURN_DURATION_MAX = 1.0  # Maximum seconds for a turn
RANDOM_CAMERA_PAN_SPEED = 0.75  # Camera pan speed when turning right
