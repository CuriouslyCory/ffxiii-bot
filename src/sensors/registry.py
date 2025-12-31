"""
Sensor registry for managing state-specific sensor availability.
"""
from typing import Dict, List, Type, Set
from .base import Sensor


class SensorRegistry:
    """
    Registry for managing which sensors are available to which states.
    
    States can register sensors that are available in their context.
    The registry provides methods to query available sensors and manage
    sensor lifecycle (enable/disable) based on current state.
    """
    
    def __init__(self):
        """Initialize the sensor registry."""
        # Map state class -> set of sensors
        self._state_sensors: Dict[Type, Set[Sensor]] = {}
        # Map sensor -> set of state classes that use it
        self._sensor_states: Dict[Sensor, Set[Type]] = {}
    
    def register(self, state_class: Type, sensor: Sensor):
        """
        Register a sensor as available for a specific state class.
        
        Args:
            state_class: The state class that can use this sensor
            sensor: The sensor instance to register
        """
        if state_class not in self._state_sensors:
            self._state_sensors[state_class] = set()
        
        self._state_sensors[state_class].add(sensor)
        
        if sensor not in self._sensor_states:
            self._sensor_states[sensor] = set()
        self._sensor_states[sensor].add(state_class)
    
    def unregister(self, state_class: Type, sensor: Sensor):
        """
        Unregister a sensor from a state class.
        
        Args:
            state_class: The state class to unregister from
            sensor: The sensor instance to unregister
        """
        if state_class in self._state_sensors:
            self._state_sensors[state_class].discard(sensor)
        
        if sensor in self._sensor_states:
            self._sensor_states[sensor].discard(state_class)
            if not self._sensor_states[sensor]:
                del self._sensor_states[sensor]
    
    def get_available(self, state_class: Type) -> List[Sensor]:
        """
        Get all sensors available for a state class.
        
        Args:
            state_class: The state class to query
            
        Returns:
            List of sensors available to this state
        """
        return list(self._state_sensors.get(state_class, set()))
    
    def enable_for_state(self, state_class: Type):
        """
        Enable all sensors registered for a state class.
        
        Args:
            state_class: The state class to enable sensors for
        """
        for sensor in self.get_available(state_class):
            sensor.enable()
    
    def disable_for_state(self, state_class: Type):
        """
        Disable all sensors registered for a state class.
        
        Args:
            state_class: The state class to disable sensors for
        """
        for sensor in self.get_available(state_class):
            sensor.disable()
    
    def disable_all(self):
        """Disable all registered sensors."""
        for sensor in self._sensor_states.keys():
            sensor.disable()
    
    def get_sensor_by_name(self, state_class: Type, name: str) -> Sensor:
        """
        Get a sensor by name for a specific state class.
        
        Args:
            state_class: The state class to search in
            name: Name of the sensor to find
            
        Returns:
            The sensor with matching name
            
        Raises:
            KeyError: If no sensor with that name is found
        """
        for sensor in self.get_available(state_class):
            if sensor.name == name:
                return sensor
        raise KeyError(f"No sensor named '{name}' found for state {state_class.__name__}")
