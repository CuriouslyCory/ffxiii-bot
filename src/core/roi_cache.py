"""
ROI Cache for managing and caching region of interest extractions.
"""
from typing import Dict, Optional, Tuple
import numpy as np
from src.core.vision import VisionEngine


class ROICache:
    """
    Cache for Region of Interest (ROI) extractions.
    
    Allows labeled ROIs to be registered and extracted, with caching
    per frame to avoid redundant extractions when multiple sensors
    need the same ROI.
    """
    
    def __init__(self, vision_engine: VisionEngine):
        """
        Initialize the ROI cache.
        
        Args:
            vision_engine: VisionEngine instance for ROI extraction
        """
        self.vision = vision_engine
        self._roi_definitions: Dict[str, Tuple[int, int, int, int]] = {}
        self._cache: Dict[str, np.ndarray] = {}
    
    def register_roi(self, label: str, coords: Tuple[int, int, int, int]):
        """
        Register an ROI definition.
        
        Args:
            label: Unique label for this ROI
            coords: ROI coordinates as (x, y, width, height)
        """
        self._roi_definitions[label] = coords
    
    def has_roi(self, label: str) -> bool:
        """
        Check if an ROI with the given label is registered.
        
        Args:
            label: ROI label to check
            
        Returns:
            True if ROI is registered, False otherwise
        """
        return label in self._roi_definitions
    
    def get_roi(self, label: str, full_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get ROI extraction, using cache if available.
        
        Args:
            label: ROI label
            full_image: Full screen capture image
            
        Returns:
            Extracted ROI image, or None if label not registered
        """
        if label not in self._roi_definitions:
            return None
        
        # Return cached extraction if available
        if label in self._cache:
            return self._cache[label]
        
        # Extract and cache
        coords = self._roi_definitions[label]
        roi_image = self.vision.get_roi_slice(full_image, coords)
        self._cache[label] = roi_image
        return roi_image
    
    def clear_cache(self):
        """Clear the cache (should be called each frame)."""
        self._cache.clear()
    
    def get_coords(self, label: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get ROI coordinates for a registered label.
        
        Args:
            label: ROI label
            
        Returns:
            ROI coordinates as (x, y, width, height), or None if not registered
        """
        return self._roi_definitions.get(label)
