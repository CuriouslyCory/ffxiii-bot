"""
LightGlue feature matcher for high-accuracy feature correspondence.

Uses attention-based matching to find correspondences between feature sets
from different images, providing better accuracy than brute-force matching.

Note: LightGlue is optional. If not available, falls back to brute-force
matching with Lowe's ratio test. To install LightGlue (optional):
    pip install git+https://github.com/cvg/LightGlue.git
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    # Fallback will be used automatically


class LightGlueMatcher:
    """
    Wrapper for LightGlue attention-based feature matching.
    
    Matches features between image pairs using an attention mechanism,
    providing better accuracy than brute-force matching, especially for
    repetitive patterns.
    """
    
    def __init__(self, device: Optional[str] = None, filter_threshold: float = 0.2):
        """
        Initialize the LightGlue matcher.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect).
            filter_threshold: Minimum confidence threshold for matches.
        """
        self.filter_threshold = filter_threshold
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize LightGlue model if available
        if LIGHTGLUE_AVAILABLE:
            self.matcher = LightGlue(pretrained='superpoint').to(self.device).eval()
            print(f"[LightGlue] Initialized on device: {self.device}")
        else:
            self.matcher = None
            print("[LightGlue] Using fallback brute-force matching")
    
    def match(self, 
              features0: Dict[str, np.ndarray],
              features1: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Match features between two feature sets.
        
        Args:
            features0: First feature set with keys 'keypoints', 'descriptors', 'scores'.
            features1: Second feature set with same structure.
            
        Returns:
            Dictionary with keys:
                - 'matches': (M, 2) array of match indices [idx0, idx1]
                - 'match_confidence': (M,) array of match confidence scores
                - 'inliers': (M,) boolean array indicating inlier matches
        """
        if not LIGHTGLUE_AVAILABLE or self.matcher is None:
            return self._fallback_match(features0, features1)
        
        # Convert to tensors and add batch dimension
        # LightGlue expects shape (batch, num_keypoints, ...)
        kpts0 = torch.from_numpy(features0['keypoints']).float().unsqueeze(0).to(self.device)  # (1, N, 2)
        desc0 = torch.from_numpy(features0['descriptors']).float().unsqueeze(0).to(self.device)  # (1, N, 256)
        scores0 = torch.from_numpy(features0['scores']).float().unsqueeze(0).to(self.device)  # (1, N)
        
        kpts1 = torch.from_numpy(features1['keypoints']).float().unsqueeze(0).to(self.device)  # (1, M, 2)
        desc1 = torch.from_numpy(features1['descriptors']).float().unsqueeze(0).to(self.device)  # (1, M, 256)
        scores1 = torch.from_numpy(features1['scores']).float().unsqueeze(0).to(self.device)  # (1, M)
        
        # Match features
        # LightGlue expects a single dictionary with 'image0' and 'image1' keys
        with torch.no_grad():
            data = {
                'image0': {
                    'keypoints': kpts0,
                    'descriptors': desc0,
                    'keypoint_scores': scores0
                },
                'image1': {
                    'keypoints': kpts1,
                    'descriptors': desc1,
                    'keypoint_scores': scores1
                }
            }
            matches01 = self.matcher(data)
        
        # Extract matches
        # LightGlue returns matches as a dictionary with 'matches' and 'scores'
        # Remove batch dimension
        matches = matches01['matches'][0].cpu().numpy()  # (M, 2) - indices [idx0, idx1]
        mconf = matches01['scores'][0].cpu().numpy()  # (M,) - matching confidence scores
        
        # Filter by confidence threshold
        valid = mconf >= self.filter_threshold
        matches = matches[valid]
        mconf = mconf[valid]
        
        # All matches are considered inliers initially (MAGSAC will filter further)
        inliers = np.ones(len(matches), dtype=bool)
        
        return {
            'matches': matches.astype(np.int32),
            'match_confidence': mconf.astype(np.float32),
            'inliers': inliers
        }
    
    def _fallback_match(self,
                       features0: Dict[str, np.ndarray],
                       features1: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fallback to brute-force matching if LightGlue is not available.
        
        Uses L2 distance matching with ratio test.
        """
        desc0 = features0['descriptors']
        desc1 = features1['descriptors']
        kpts0 = features0['keypoints']
        kpts1 = features1['keypoints']
        
        # Compute pairwise distances
        distances = np.linalg.norm(desc0[:, None, :] - desc1[None, :, :], axis=2)
        
        # Find best and second-best matches
        sorted_indices = np.argsort(distances, axis=1)
        best_matches = sorted_indices[:, 0]
        second_best_matches = sorted_indices[:, 1]
        
        # Lowe's ratio test
        ratio_threshold = 0.75
        ratios = distances[np.arange(len(desc0)), best_matches] / (
            distances[np.arange(len(desc0)), second_best_matches] + 1e-8
        )
        valid = ratios < ratio_threshold
        
        # Build match array
        matches = np.column_stack([
            np.arange(len(desc0))[valid],
            best_matches[valid]
        ])
        
        # Compute confidence as inverse distance (normalized)
        match_distances = distances[valid, best_matches[valid]]
        max_dist = match_distances.max() if len(match_distances) > 0 else 1.0
        match_confidence = 1.0 - (match_distances / (max_dist + 1e-8))
        
        inliers = np.ones(len(matches), dtype=bool)
        
        return {
            'matches': matches.astype(np.int32),
            'match_confidence': match_confidence.astype(np.float32),
            'inliers': inliers
        }
    
    def match_all_pairs(self, 
                       all_features: list) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
        """
        Match features between all pairs of images.
        
        Args:
            all_features: List of feature dictionaries (one per image).
            
        Returns:
            Dictionary mapping (i, j) pairs to match results.
        """
        matches_dict = {}
        n = len(all_features)
        
        for i in range(n):
            for j in range(i + 1, n):
                matches = self.match(all_features[i], all_features[j])
                matches_dict[(i, j)] = matches
        
        return matches_dict
