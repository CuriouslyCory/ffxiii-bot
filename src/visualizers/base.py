"""
Base visualizer interface for debug dashboards and visualization panels.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import cv2


class Visualizer(ABC):
    """
    Abstract base class for all visualizers.
    
    Visualizers render debug information and state dashboards.
    """
    
    def __init__(self, window_name: str):
        """
        Initialize the visualizer.
        
        Args:
            window_name: Name of the OpenCV window for this visualizer
        """
        self.window_name = window_name
        self._visible = False
    
    @abstractmethod
    def render(self, image: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """
        Render the visualizer content.
        
        Args:
            image: Current screen capture or base image
            data: Dictionary containing data to visualize
            
        Returns:
            Rendered image
        """
        pass
    
    def show(self):
        """Show the visualizer window."""
        self._visible = True
    
    def hide(self):
        """Hide the visualizer window."""
        self._visible = False
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            # Window doesn't exist, which is fine
            pass
    
    def is_visible(self) -> bool:
        """Check if the visualizer is currently visible."""
        return self._visible
    
    def cleanup(self):
        """Clean up resources (close windows, etc.)."""
        self.hide()
