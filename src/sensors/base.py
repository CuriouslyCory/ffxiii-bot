"""
Base sensor interface for game state detection.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Sensor(ABC):
    """
    Abstract base class for all sensors.
    
    Sensors detect and extract information from game screens.
    They can be enabled/disabled to control processing (lazy evaluation).
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
    
    def __repr__(self) -> str:
        """String representation of the sensor."""
        status = "enabled" if self._enabled else "disabled"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
