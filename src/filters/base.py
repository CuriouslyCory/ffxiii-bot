"""
Base filter interface for image processing filters.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Filter(ABC):
    """
    Abstract base class for all image filters.
    
    Filters transform input images to produce filtered output images.
    They can be composed together using CompositeFilter.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the filter.
        
        Args:
            name: Human-readable name for this filter
            description: Optional description of what the filter does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the filter to an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Filtered image (BGR format)
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        return f"{self.__class__.__name__}(name='{self.name}')"
