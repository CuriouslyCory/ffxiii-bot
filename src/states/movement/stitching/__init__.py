"""
Advanced map stitching pipeline components.

This module provides state-of-the-art stitching capabilities using:
- SuperPoint + LightGlue for feature detection and matching
- MAGSAC++ for robust estimation
- Bundle Adjustment for global optimization
- APAP/Mesh-warping for parallax-tolerant stitching
"""

try:
    from .feature_extractor import SuperPointExtractor
except ImportError:
    SuperPointExtractor = None

try:
    from .feature_matcher import LightGlueMatcher
except ImportError:
    LightGlueMatcher = None

try:
    from .robust_estimator import MAGSACEstimator
except ImportError:
    MAGSACEstimator = None

try:
    from .bundle_adjustment import BundleAdjustment
except ImportError:
    BundleAdjustment = None

try:
    from .mesh_warping import APAPWarper
except ImportError:
    APAPWarper = None

__all__ = [
    'SuperPointExtractor',
    'LightGlueMatcher',
    'MAGSACEstimator',
    'BundleAdjustment',
    'APAPWarper',
]
