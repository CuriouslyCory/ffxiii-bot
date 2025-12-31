"""
Bundle Adjustment for global optimization of node positions.

Jointly optimizes all node positions and orientations to minimize
reprojection error across all feature matches, reducing cumulative drift.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
import cv2


class BundleAdjustment:
    """
    Bundle Adjustment solver for global optimization.
    
    Optimizes node positions (x, y) and orientations (θ) simultaneously
    to minimize reprojection error across all feature correspondences.
    """
    
    def __init__(self,
                 max_iterations: int = 100,
                 ftol: float = 1e-6,
                 xtol: float = 1e-8,
                 gtol: float = 1e-6):
        """
        Initialize Bundle Adjustment solver.
        
        Args:
            max_iterations: Maximum optimization iterations.
            ftol: Function tolerance for convergence.
            xtol: Parameter tolerance for convergence.
            gtol: Gradient tolerance for convergence.
        """
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
    
    def optimize(self,
                 initial_positions: np.ndarray,
                 all_features: List[Dict[str, np.ndarray]],
                 all_matches: Dict[Tuple[int, int], Dict[str, np.ndarray]],
                 pairwise_transforms: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
        """
        Optimize node positions using Bundle Adjustment.
        
        Args:
            initial_positions: (N, 3) array of initial [x, y, θ] for each node.
            all_features: List of feature dictionaries (one per node).
            all_matches: Dictionary mapping (i, j) pairs to match results.
            pairwise_transforms: Dictionary mapping (i, j) pairs to transformation matrices.
            
        Returns:
            (N, 3) array of optimized [x, y, θ] positions.
        """
        n_nodes = len(initial_positions)
        
        # Flatten initial parameters [x0, y0, θ0, x1, y1, θ1, ...]
        x0 = initial_positions.flatten()
        
        # Build observation list: (node_i, node_j, match_idx, point_in_i, point_in_j)
        observations = []
        for (i, j), match_data in all_matches.items():
            if (i, j) not in pairwise_transforms:
                continue
            
            matches = match_data['matches']
            inliers = match_data.get('inliers', np.ones(len(matches), dtype=bool))
            
            # Only use inlier matches
            valid_matches = matches[inliers]
            
            kpts_i = all_features[i]['keypoints']
            kpts_j = all_features[j]['keypoints']
            
            for match in valid_matches:
                idx_i, idx_j = match
                pt_i = kpts_i[idx_i]
                pt_j = kpts_j[idx_j]
                
                observations.append((i, j, pt_i, pt_j))
        
        if len(observations) == 0:
            print("[Bundle Adjustment] No valid observations, returning initial positions")
            return initial_positions
        
        # Define residual function
        def residuals(params):
            """Compute reprojection residuals."""
            # Reshape parameters to (N, 3)
            positions = params.reshape(n_nodes, 3)
            
            residuals_list = []
            for i, j, pt_i, pt_j in observations:
                # Get node positions
                x_i, y_i, theta_i = positions[i]
                x_j, y_j, theta_j = positions[j]
                
                # Transform point from node i to world coordinates
                # First rotate by theta_i, then translate
                cos_i = np.cos(np.deg2rad(theta_i))
                sin_i = np.sin(np.deg2rad(theta_i))
                
                # Point in world coordinates (from node i)
                world_x_i = x_i + pt_i[0] * cos_i - pt_i[1] * sin_i
                world_y_i = y_i + pt_i[0] * sin_i + pt_i[1] * cos_i
                
                # Transform point from node j to world coordinates
                cos_j = np.cos(np.deg2rad(theta_j))
                sin_j = np.sin(np.deg2rad(theta_j))
                
                world_x_j = x_j + pt_j[0] * cos_j - pt_j[1] * sin_j
                world_y_j = y_j + pt_j[0] * sin_j + pt_j[1] * cos_j
                
                # Residual is the difference in world coordinates
                # (they should match if transformation is correct)
                residuals_list.append(world_x_i - world_x_j)
                residuals_list.append(world_y_i - world_y_j)
            
            return np.array(residuals_list)
        
        # Optimize using Levenberg-Marquardt
        print(f"[Bundle Adjustment] Optimizing {len(observations)} observations across {n_nodes} nodes...")
        
        result = least_squares(
            residuals,
            x0,
            method='lm',  # Levenberg-Marquardt
            max_nfev=self.max_iterations,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            verbose=1
        )
        
        # Reshape optimized parameters
        optimized_positions = result.x.reshape(n_nodes, 3)
        
        # Compute final error
        final_residuals = residuals(result.x)
        final_error = np.sqrt(np.mean(final_residuals**2))
        
        print(f"[Bundle Adjustment] Optimization complete. Final RMS error: {final_error:.2f} pixels")
        
        return optimized_positions
    
    def optimize_simple(self,
                       initial_positions: np.ndarray,
                       pairwise_transforms: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
        """
        Simplified optimization using only pairwise transformations.
        
        This version uses the pairwise transformations directly without
        feature correspondences, which is faster but less accurate.
        
        Args:
            initial_positions: (N, 3) array of initial [x, y, θ].
            pairwise_transforms: Dictionary mapping (i, j) to transformation matrices.
            
        Returns:
            (N, 3) array of optimized positions.
        """
        n_nodes = len(initial_positions)
        
        # Build constraint graph from transformations
        # For each transformation T_ij, we have: pos_j ≈ T_ij(pos_i)
        x0 = initial_positions.flatten()
        
        def residuals(params):
            """Compute residuals from pairwise transformations."""
            positions = params.reshape(n_nodes, 3)
            residuals_list = []
            
            for (i, j), T in pairwise_transforms.items():
                if T is None:
                    continue
                
                # Get positions
                x_i, y_i, theta_i = positions[i]
                x_j, y_j, theta_j = positions[j]
                
                # Transform point (0, 0) from node i using T
                # T is a 3x3 homography matrix
                pt_i = np.array([0.0, 0.0, 1.0])  # Origin in node i
                pt_transformed = T @ pt_i
                pt_transformed = pt_transformed[:2] / pt_transformed[2]  # Normalize
                
                # Expected position of origin in node j's frame
                # Should be at (x_j - x_i, y_j - y_i) in world coordinates
                # But we need to account for rotation
                cos_i = np.cos(np.deg2rad(theta_i))
                sin_i = np.sin(np.deg2rad(theta_i))
                
                # Transform origin through node i's frame
                world_x = x_i + pt_transformed[0] * cos_i - pt_transformed[1] * sin_i
                world_y = y_i + pt_transformed[0] * sin_i + pt_transformed[1] * cos_i
                
                # Expected position in node j's frame
                cos_j = np.cos(np.deg2rad(theta_j))
                sin_j = np.sin(np.deg2rad(theta_j))
                
                # Inverse transform to node j's frame
                rel_x = world_x - x_j
                rel_y = world_y - y_j
                
                # Rotate back to node j's local frame
                local_x = rel_x * cos_j + rel_y * sin_j
                local_y = -rel_x * sin_j + rel_y * cos_j
                
                # Residual: should be close to (0, 0) in node j's frame
                residuals_list.append(local_x)
                residuals_list.append(local_y)
            
            return np.array(residuals_list)
        
        print(f"[Bundle Adjustment] Optimizing {len(pairwise_transforms)} pairwise constraints...")
        
        result = least_squares(
            residuals,
            x0,
            method='lm',
            max_nfev=self.max_iterations,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            verbose=1
        )
        
        optimized_positions = result.x.reshape(n_nodes, 3)
        
        final_residuals = residuals(result.x)
        final_error = np.sqrt(np.mean(final_residuals**2))
        
        print(f"[Bundle Adjustment] Optimization complete. Final RMS error: {final_error:.2f}")
        
        return optimized_positions
