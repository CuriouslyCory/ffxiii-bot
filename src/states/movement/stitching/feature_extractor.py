"""
SuperPoint feature extractor for high-precision keypoint detection and description.

Uses LightGlue's SuperPoint implementation (preferred) or falls back to kornia if available.
LightGlue's SuperPoint is actively maintained and works seamlessly with LightGlue matching.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict

_IMPORT_ERROR = None
try:
    import torch
    # Try LightGlue's SuperPoint first (preferred)
    try:
        from lightglue import SuperPoint
        SUPERPOINT_SOURCE = 'lightglue'
        SUPERPOINT_AVAILABLE = True
    except ImportError:
        # Fallback to kornia if available
        try:
            from kornia.feature import SuperPoint
            SUPERPOINT_SOURCE = 'kornia'
            SUPERPOINT_AVAILABLE = True
        except ImportError:
            SUPERPOINT_AVAILABLE = False
            SuperPoint = None
            SUPERPOINT_SOURCE = None
            _IMPORT_ERROR = "Neither lightglue nor kornia SuperPoint available"
except ImportError as e:
    SUPERPOINT_AVAILABLE = False
    torch = None
    SuperPoint = None
    SUPERPOINT_SOURCE = None
    _IMPORT_ERROR = str(e)
except Exception as e:
    # Catch other potential errors (e.g., version incompatibility)
    SUPERPOINT_AVAILABLE = False
    torch = None
    SuperPoint = None
    SUPERPOINT_SOURCE = None
    _IMPORT_ERROR = str(e)


class SuperPointExtractor:
    """
    Wrapper for SuperPoint feature extraction.
    
    Extracts keypoints and descriptors from minimap images using a pre-trained
    SuperPoint neural network. Provides rotation-invariant feature detection
    suitable for map stitching.
    """
    
    def __init__(self, n_features: int = 512, device: Optional[str] = None):
        """
        Initialize the SuperPoint extractor.
        
        Args:
            n_features: Maximum number of features to extract per image.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).
        """
        if not SUPERPOINT_AVAILABLE:
            error_msg = "SuperPoint is required. Install with: uv add 'lightglue @ git+https://github.com/cvg/LightGlue.git'"
            if _IMPORT_ERROR:
                error_msg += f"\nImport error: {_IMPORT_ERROR}"
            raise ImportError(error_msg)
        
        self.n_features = n_features
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize SuperPoint model
        # LightGlue's SuperPoint uses different initialization
        if SUPERPOINT_SOURCE == 'lightglue':
            # LightGlue SuperPoint takes max_num_keypoints as a config parameter
            self.model = SuperPoint(max_num_keypoints=n_features).to(self.device).eval()
        else:
            # kornia's SuperPoint
            self.model = SuperPoint(pretrained=True).to(self.device).eval()
        
        print(f"[SuperPoint] Initialized from {SUPERPOINT_SOURCE} on device: {self.device} (max_features={n_features})")
    
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract keypoints and descriptors from an image.
        
        Args:
            image: Input image as numpy array (BGR format, uint8).
            
        Returns:
            Dictionary with keys:
                - 'keypoints': (N, 2) array of keypoint coordinates (x, y)
                - 'descriptors': (N, 256) array of descriptor vectors
                - 'scores': (N,) array of keypoint confidence scores
        """
        # Convert BGR to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            if SUPERPOINT_SOURCE == 'lightglue':
                # LightGlue's SuperPoint expects {'image': tensor} format
                features = self.model({'image': img_tensor})
                # LightGlue returns batched results, extract first batch
                keypoints = features['keypoints'][0].cpu().numpy()  # (N, 2)
                descriptors = features['descriptors'][0].cpu().numpy()  # (N, 256)
                scores = features['keypoint_scores'][0].cpu().numpy()  # (N,)
            else:
                # kornia's SuperPoint
                features = self.model(img_tensor)
                keypoints = features['keypoints'][0].cpu().numpy()  # (N, 2)
                descriptors = features['descriptors'][0].cpu().numpy()  # (N, 256)
                scores = features['keypoint_scores'][0].cpu().numpy()  # (N,)
        
        # Limit to top N features by score
        if len(keypoints) > self.n_features:
            top_indices = np.argsort(scores)[-self.n_features:][::-1]
            keypoints = keypoints[top_indices]
            descriptors = descriptors[top_indices]
            scores = scores[top_indices]
        
        # Normalize descriptors (L2 normalization)
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8)
        
        return {
            'keypoints': keypoints.astype(np.float32),
            'descriptors': descriptors.astype(np.float32),
            'scores': scores.astype(np.float32)
        }
    
    def extract_batch(self, images: list) -> list:
        """
        Extract features from multiple images.
        
        Args:
            images: List of numpy arrays (BGR format).
            
        Returns:
            List of feature dictionaries (same format as extract()).
        """
        return [self.extract(img) for img in images]
