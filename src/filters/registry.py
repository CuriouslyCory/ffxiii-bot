"""
Global filter registry for reusable filter pipelines.
"""
from typing import Dict, Optional
from .base import Filter


class FilterRegistry:
    """
    Singleton registry for storing and retrieving named filter pipelines.
    
    Allows filters to be registered globally and reused across multiple sensors.
    For example, "minimap_frame" can be registered once and used by multiple sensors.
    """
    
    _instance: Optional['FilterRegistry'] = None
    _filters: Dict[str, Filter] = {}
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._filters = {}
        return cls._instance
    
    @classmethod
    def register_filter(cls, name: str, filter_obj: Filter) -> None:
        """
        Register a named filter pipeline.
        
        Args:
            name: Unique name for the filter pipeline
            filter_obj: Filter instance to register (can be CompositeFilter or single filter)
            
        Raises:
            ValueError: If name is empty or filter_obj is None
        """
        if not name:
            raise ValueError("Filter name cannot be empty")
        if filter_obj is None:
            raise ValueError("Filter object cannot be None")
        
        instance = cls()
        instance._filters[name] = filter_obj
    
    @classmethod
    def get_filter(cls, name: str) -> Optional[Filter]:
        """
        Retrieve a registered filter by name.
        
        Args:
            name: Name of the filter to retrieve
            
        Returns:
            Filter instance if found, None otherwise
        """
        instance = cls()
        return instance._filters.get(name)
    
    @classmethod
    def has_filter(cls, name: str) -> bool:
        """
        Check if a filter is registered.
        
        Args:
            name: Name of the filter to check
            
        Returns:
            True if filter is registered, False otherwise
        """
        instance = cls()
        return name in instance._filters
    
    @classmethod
    def list_filters(cls) -> list[str]:
        """
        Get list of all registered filter names.
        
        Returns:
            List of registered filter names
        """
        instance = cls()
        return list(instance._filters.keys())
    
    @classmethod
    def unregister_filter(cls, name: str) -> bool:
        """
        Unregister a filter by name.
        
        Args:
            name: Name of the filter to unregister
            
        Returns:
            True if filter was removed, False if it didn't exist
        """
        instance = cls()
        if name in instance._filters:
            del instance._filters[name]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered filters."""
        instance = cls()
        instance._filters.clear()
