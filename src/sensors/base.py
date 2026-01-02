"""
Base sensor interface for game state detection.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import numpy as np
from src.filters.base import Filter


class Sensor(ABC):
    """
    Abstract base class for all sensors.
    
    Sensors detect and extract information from game screens.
    They can be enabled/disabled to control processing (lazy evaluation).
    
    Sensors can register filters for debugging and reuse, and register debug outputs
    at different processing stages for visualization.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the sensor.
        
        Args:
            name: Human-readable name for this sensor
            description: Optional description of what the sensor detects
        """
        self.name = name
        self.description = description
        self._enabled = False
        self._registered_filters: Dict[str, Filter] = {}
        self._debug_outputs: Dict[str, np.ndarray] = {}
    
    @property
    def is_enabled(self) -> bool:
        """Check if the sensor is currently enabled."""
        return self._enabled
    
    def enable(self):
        """Enable the sensor (will process when read() is called)."""
        self._enabled = True
    
    def disable(self):
        """Disable the sensor (will not process when read() is called)."""
        self._enabled = False
    
    @abstractmethod
    def read(self, image: np.ndarray) -> Any:
        """
        Read sensor data from the current image.
        
        This method should check is_enabled before processing.
        If disabled, it may return None or a default value.
        
        Args:
            image: Current screen capture (BGR format)
            
        Returns:
            Sensor data (type depends on sensor implementation)
        """
        pass
    
    def is_available(self, image: np.ndarray) -> bool:
        """
        Check if the sensor can provide data for the current image.
        
        This is optional - sensors can override this to check if the
        required UI elements are present before attempting to read.
        
        Args:
            image: Current screen capture (BGR format)
            
        Returns:
            True if sensor can provide data, False otherwise
        """
        return True
    
    def register_filter(self, name: str, filter_obj: Filter) -> None:
        """
        Register a filter with this sensor.
        
        Registered filters can be accessed by debug tools for parameter adjustment.
        
        Args:
            name: Name for the filter (used for identification)
            filter_obj: Filter instance to register
        """
        if not name:
            raise ValueError("Filter name cannot be empty")
        if filter_obj is None:
            raise ValueError("Filter object cannot be None")
        self._registered_filters[name] = filter_obj
    
    def get_registered_filters(self) -> Dict[str, Filter]:
        """
        Get all registered filters for this sensor.
        
        Returns:
            Dictionary mapping filter names to Filter instances
        """
        return self._registered_filters.copy()
    
    def register_debug_output(self, label: str, image: np.ndarray) -> None:
        """
        Register a debug image output at a processing stage.
        
        Debug outputs are collected during sensor processing and can be
        displayed by debug tools for inspection.
        
        Args:
            label: Label/name for this debug output
            image: Image to register (will be copied)
        """
        if not label:
            raise ValueError("Debug output label cannot be empty")
        if image is None:
            raise ValueError("Debug output image cannot be None")
        # Store a copy to avoid issues with image modification
        self._debug_outputs[label] = image.copy()
    
    def get_debug_outputs(self) -> Dict[str, np.ndarray]:
        """
        Get all registered debug outputs.
        
        Returns:
            Dictionary mapping labels to debug images
        """
        return self._debug_outputs.copy()
    
    def clear_debug_outputs(self) -> None:
        """
        Clear all registered debug outputs.
        
        Should be called at the start of each read() call to ensure
        debug outputs only contain data from the current frame.
        """
        self._debug_outputs.clear()
    
    def __repr__(self) -> str:
        """String representation of the sensor."""
        status = "enabled" if self._enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
