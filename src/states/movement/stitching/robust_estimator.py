"""
MAGSAC++ robust estimator for outlier rejection and transformation estimation.

Provides better accuracy than RANSAC in noisy environments by using
more sophisticated probability models to distinguish true matches from outliers.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class MAGSACEstimator:
    """
    Wrapper for MAGSAC++ robust estimation.
    
    Estimates homography/affine transformations between matched feature points
    with improved outlier rejection compared to standard RANSAC.
    """
    
    def __init__(self, 
                 method: str = 'homography',
                 threshold: float = 5.0,
                 confidence: float = 0.99,
                 max_iters: int = 2000):
        """
        Initialize the MAGSAC estimator.
        
        Args:
            method: Transformation type ('homography' or 'affine').
            threshold: Inlier threshold in pixels.
            confidence: Required confidence level (0-1).
            max_iters: Maximum number of iterations.
        """
        self.method = method
        self.threshold = threshold
        self.confidence = confidence
        self.max_iters = max_iters
        
        # Map method to OpenCV constant
        if method == 'homography':
            self.method_flag = cv2.USAC_MAGSAC
        elif method == 'affine':
            self.method_flag = cv2.USAC_MAGSAC
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate(self,
                 points0: np.ndarray,
                 points1: np.ndarray,
                 matches: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """
        Estimate transformation between two point sets.
        
        Args:
            points0: (N, 2) array of points from first image.
            points1: (N, 2) array of points from second image.
            matches: Optional (M, 2) array of match indices. If None, assumes
                     points0 and points1 are already aligned.
            
        Returns:
            Tuple of:
                - Transformation matrix (3x3 for homography, 2x3 for affine) or None
                - Inlier mask (boolean array)
                - Confidence score (0-1)
        """
        # Handle matches if provided
        if matches is not None:
            src_pts = points0[matches[:, 0]]
            dst_pts = points1[matches[:, 1]]
        else:
            if len(points0) != len(points1):
                raise ValueError("points0 and points1 must have same length if matches not provided")
            src_pts = points0
            dst_pts = points1
        
        if len(src_pts) < 4:  # Minimum points for homography
            return None, np.array([], dtype=bool), 0.0
        
        # Reshape for OpenCV (N, 1, 2)
        src_pts = src_pts.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = dst_pts.reshape(-1, 1, 2).astype(np.float32)
        
        # Estimate transformation using MAGSAC
        try:
            if self.method == 'homography':
                M, inlier_mask = cv2.findHomography(
                    src_pts, dst_pts,
                    method=self.method_flag,
                    ransacReprojThreshold=self.threshold,
                    confidence=self.confidence,
                    maxIters=self.max_iters
                )
            else:  # affine
                M, inlier_mask = cv2.estimateAffine2D(
                    src_pts, dst_pts,
                    method=self.method_flag,
                    ransacReprojThreshold=self.threshold,
                    confidence=self.confidence,
                    maxIters=self.max_iters
                )
                # Convert to 3x3 for consistency
                if M is not None:
                    M_3x3 = np.eye(3)
                    M_3x3[:2, :] = M
                    M = M_3x3
            
            # Compute confidence based on inlier ratio
            n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
            n_total = len(src_pts)
            confidence_score = n_inliers / n_total if n_total > 0 else 0.0
            
            # Convert inlier_mask to boolean array
            if inlier_mask is not None:
                inlier_mask = inlier_mask.ravel().astype(bool)
            else:
                inlier_mask = np.zeros(len(src_pts), dtype=bool)
            
            return M, inlier_mask, confidence_score
            
        except cv2.error as e:
            print(f"[MAGSAC] Error during estimation: {e}")
            return None, np.zeros(len(src_pts), dtype=bool), 0.0
    
    def estimate_from_features(self,
                              features0: Dict[str, np.ndarray],
                              features1: Dict[str, np.ndarray],
                              matches: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        """
        Estimate transformation from feature matches.
        
        Args:
            features0: First feature set with 'keypoints'.
            features1: Second feature set with 'keypoints'.
            matches: (M, 2) array of match indices.
            
        Returns:
            Tuple of (transformation, inlier_mask, confidence).
        """
        kpts0 = features0['keypoints']
        kpts1 = features1['keypoints']
        
        return self.estimate(kpts0, kpts1, matches)
    
    def filter_matches(self,
                      matches: np.ndarray,
                      inlier_mask: np.ndarray) -> np.ndarray:
        """
        Filter matches to keep only inliers.
        
        Args:
            matches: (M, 2) array of match indices.
            inlier_mask: (M,) boolean array indicating inliers.
            
        Returns:
            Filtered matches array.
        """
        return matches[inlier_mask]
