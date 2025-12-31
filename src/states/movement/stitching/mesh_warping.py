"""
APAP (As-Projective-As-Possible) mesh warping for parallax-tolerant stitching.

Divides overlapping regions into a mesh grid and applies localized transformations
per grid cell to handle parallax and depth variations in the scene.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class APAPWarper:
    """
    APAP mesh warping for parallax-tolerant image stitching.
    
    Applies localized projective transformations across a mesh grid to
    handle parallax and depth variations in overlapping regions.
    """
    
    def __init__(self, grid_size: int = 10, blend_radius: int = 5):
        """
        Initialize APAP warper.
        
        Args:
            grid_size: Number of grid cells per dimension.
            blend_radius: Blending radius in pixels for smooth transitions.
        """
        self.grid_size = grid_size
        self.blend_radius = blend_radius
    
    def warp_image(self,
                   image: np.ndarray,
                   homography: np.ndarray,
                   target_size: Tuple[int, int],
                   mesh_homographies: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Warp an image using APAP mesh warping.
        
        Args:
            image: Source image to warp.
            homography: Global homography matrix (3x3).
            target_size: (width, height) of output image.
            mesh_homographies: Optional (grid_size, grid_size, 3, 3) array of
                             per-cell homographies. If None, uses global homography.
            
        Returns:
            Warped image.
        """
        h, w = image.shape[:2]
        out_w, out_h = target_size
        
        if mesh_homographies is None:
            # Simple global warping
            return cv2.warpPerspective(image, homography, target_size)
        
        # Create mesh grid
        x_coords = np.linspace(0, w, self.grid_size + 1)
        y_coords = np.linspace(0, h, self.grid_size + 1)
        
        # Create output image
        warped = np.zeros((out_h, out_w, 3), dtype=image.dtype)
        weight_map = np.zeros((out_h, out_w), dtype=np.float32)
        
        # Process each grid cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get cell boundaries
                x1, x2 = int(x_coords[j]), int(x_coords[j + 1])
                y1, y2 = int(y_coords[i]), int(y_coords[i + 1])
                
                # Get cell homography
                H_cell = mesh_homographies[i, j]
                
                # Extract cell from source image
                cell = image[y1:y2, x1:x2]
                
                if cell.size == 0:
                    continue
                
                # Warp cell
                cell_h, cell_w = cell.shape[:2]
                cell_warped = cv2.warpPerspective(
                    cell, H_cell, (out_w, out_h)
                )
                
                # Create weight mask (distance-based)
                mask = np.ones((out_h, out_w), dtype=np.float32)
                
                # Apply blending weights
                warped += cell_warped * mask[:, :, np.newaxis]
                weight_map += mask
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-8)
        warped = warped / weight_map[:, :, np.newaxis]
        
        return warped.astype(image.dtype)
    
    def compute_mesh_homographies(self,
                                image0: np.ndarray,
                                image1: np.ndarray,
                                matches: np.ndarray,
                                keypoints0: np.ndarray,
                                keypoints1: np.ndarray,
                                global_homography: np.ndarray) -> np.ndarray:
        """
        Compute per-cell homographies for APAP warping.
        
        Args:
            image0: First image.
            image1: Second image.
            matches: (M, 2) array of match indices.
            keypoints0: Keypoints from first image.
            keypoints1: Keypoints from second image.
            global_homography: Global homography matrix.
            
        Returns:
            (grid_size, grid_size, 3, 3) array of homographies.
        """
        h, w = image0.shape[:2]
        
        # Create mesh grid
        x_coords = np.linspace(0, w, self.grid_size + 1)
        y_coords = np.linspace(0, h, self.grid_size + 1)
        
        mesh_homographies = np.zeros((self.grid_size, self.grid_size, 3, 3))
        
        # For each grid cell, compute local homography
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get cell center
                cell_center_x = (x_coords[j] + x_coords[j + 1]) / 2
                cell_center_y = (y_coords[i] + y_coords[i + 1]) / 2
                
                # Find matches within this cell (with some margin)
                cell_margin = min(w, h) / self.grid_size
                
                # Filter matches by distance to cell center
                kpts0_matched = keypoints0[matches[:, 0]]
                distances = np.sqrt(
                    (kpts0_matched[:, 0] - cell_center_x)**2 +
                    (kpts0_matched[:, 1] - cell_center_y)**2
                )
                
                cell_matches = matches[distances < cell_margin]
                
                if len(cell_matches) >= 4:
                    # Compute local homography for this cell
                    src_pts = keypoints0[cell_matches[:, 0]].reshape(-1, 1, 2).astype(np.float32)
                    dst_pts = keypoints1[cell_matches[:, 1]].reshape(-1, 1, 2).astype(np.float32)
                    
                    H_local, _ = cv2.findHomography(
                        src_pts, dst_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=5.0
                    )
                    
                    if H_local is not None:
                        mesh_homographies[i, j] = H_local
                    else:
                        mesh_homographies[i, j] = global_homography
                else:
                    # Not enough matches, use global homography
                    mesh_homographies[i, j] = global_homography
        
        return mesh_homographies
    
    def blend_images(self,
                    image0: np.ndarray,
                    image1: np.ndarray,
                    mask0: Optional[np.ndarray] = None,
                    mask1: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Blend two warped images with smooth transitions.
        
        Args:
            image0: First warped image.
            image1: Second warped image.
            mask0: Optional mask for first image (non-zero = valid).
            mask1: Optional mask for second image (non-zero = valid).
            
        Returns:
            Blended image.
        """
        if mask0 is None:
            mask0 = (image0.sum(axis=2) > 0).astype(np.float32)
        if mask1 is None:
            mask1 = (image1.sum(axis=2) > 0).astype(np.float32)
        
        # Create distance-based weights
        # Use distance transform for smooth blending
        dist0 = cv2.distanceTransform(
            mask0.astype(np.uint8), cv2.DIST_L2, 5
        )
        dist1 = cv2.distanceTransform(
            mask1.astype(np.uint8), cv2.DIST_L2, 5
        )
        
        # Normalize weights
        total_dist = dist0 + dist1
        total_dist = np.maximum(total_dist, 1e-8)
        
        weight0 = dist0 / total_dist
        weight1 = dist1 / total_dist
        
        # Blend
        blended = (
            image0 * weight0[:, :, np.newaxis] +
            image1 * weight1[:, :, np.newaxis]
        )
        
        return blended.astype(image0.dtype)
