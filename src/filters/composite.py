"""
Composite filters for chaining multiple filters together.
"""
import cv2
import numpy as np
from typing import List, Literal
from .base import Filter


class CompositeFilter(Filter):
    """
    Composite filter that chains multiple filters together.
    
    Supports two composition modes:
    - Additive: Combines filter results (union) - e.g., blue + gold = both colors
    - Progressive: Sequential application (intersection) - e.g., blue then gold = only gold pixels that were blue
    """
    
    def __init__(
        self,
        filters: List[Filter],
        mode: Literal["additive", "progressive"] = "additive",
        name: str = "Composite Filter",
        description: str = ""
    ):
        """
        Initialize composite filter.
        
        Args:
            filters: List of filters to compose
            mode: Composition mode ("additive" or "progressive")
            name: Filter name
            description: Filter description
        """
        if not filters:
            raise ValueError("CompositeFilter requires at least one filter")
        
        if mode not in ("additive", "progressive"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'additive' or 'progressive'")
        
        super().__init__(name, description)
        self.filters = filters
        self.mode = mode
        
        # Generate description if not provided
        if not description:
            filter_names = ", ".join(f.name for f in filters)
            self.description = f"Composite filter ({mode}): {filter_names}"
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply composite filter to the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Filtered BGR image
        """
        if self.mode == "additive":
            return self._apply_additive(image)
        else:  # progressive
            return self._apply_progressive(image)
    
    def _apply_additive(self, image: np.ndarray) -> np.ndarray:
        """
        Apply filters additively (combines results with OR operation).
        
        Example: blue + gold = image with both blue and gold pixels visible
        """
        results = []
        for filter_obj in self.filters:
            filtered = filter_obj.apply(image)
            results.append(filtered)
        
        # Combine all results using bitwise OR
        combined = results[0]
        for result in results[1:]:
            combined = cv2.bitwise_or(combined, result)
        
        return combined
    
    def _apply_progressive(self, image: np.ndarray) -> np.ndarray:
        """
        Apply filters progressively (sequential application).
        
        Example: blue then gold = only gold pixels that remain after blue filtering
        """
        result = image
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result
