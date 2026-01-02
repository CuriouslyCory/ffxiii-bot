"""
Mask filters for geometric shape-based image masking.
"""
import cv2
import numpy as np
from typing import Tuple, Literal
from .base import Filter


class ShapeFilter(Filter):
    """
    Base class for geometric shape-based mask filters.
    
    All shape filters support:
    - Configurable shape parameters (center, dimensions)
    - Inside/outside masking flag
    - Automatic image dimension handling
    """
    
    def __init__(
        self,
        mask_inside: bool = True,
        name: str = "Shape Filter",
        description: str = ""
    ):
        """
        Initialize shape filter.
        
        Args:
            mask_inside: If True, keep pixels inside shape (mask out outside).
                        If False, keep pixels outside shape (mask out inside).
            name: Filter name
            description: Filter description
        """
        super().__init__(name, description)
        self.mask_inside = mask_inside
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create the binary mask for the shape.
        
        This should be implemented by subclasses to create the appropriate shape mask.
        
        Args:
            image: Input image to determine dimensions
            
        Returns:
            Binary mask (0/255) where 255 indicates pixels to keep
        """
        raise NotImplementedError("Subclasses must implement _create_mask")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply shape mask to the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Masked BGR image (masked regions are blacked out)
        """
        # Create binary mask
        binary_mask = self._create_mask(image)
        
        # If mask_inside=False, invert the mask
        if not self.mask_inside:
            binary_mask = cv2.bitwise_not(binary_mask)
        
        # Apply mask to image
        result = cv2.bitwise_and(image, image, mask=binary_mask)
        return result


class EllipseMaskFilter(ShapeFilter):
    """
    Elliptical mask filter with configurable center, radii, and inside/outside flag.
    """
    
    def __init__(
        self,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        angle: float = 0.0,
        mask_inside: bool = True,
        name: str = "Ellipse Mask Filter",
        description: str = ""
    ):
        """
        Initialize ellipse mask filter.
        
        Args:
            center: Center point of ellipse (x, y)
            axes: Half-lengths of the ellipse axes (radius_x, radius_y)
            angle: Rotation angle of ellipse in degrees (default: 0.0)
            mask_inside: If True, keep pixels inside ellipse. If False, keep outside.
            name: Filter name
            description: Filter description
        """
        super().__init__(mask_inside, name, description)
        self.center = center
        self.axes = axes
        self.angle = angle
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create elliptical binary mask.
        
        Args:
            image: Input image to determine dimensions
            
        Returns:
            Binary mask with ellipse filled (255 inside, 0 outside)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, self.center, self.axes, self.angle, 0, 360, 255, -1)
        return mask


class RectangleMaskFilter(ShapeFilter):
    """
    Rectangular mask filter with configurable bounds and inside/outside flag.
    """
    
    def __init__(
        self,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        mask_inside: bool = True,
        name: str = "Rectangle Mask Filter",
        description: str = ""
    ):
        """
        Initialize rectangle mask filter.
        
        Args:
            top_left: Top-left corner (x, y)
            bottom_right: Bottom-right corner (x, y)
            mask_inside: If True, keep pixels inside rectangle. If False, keep outside.
            name: Filter name
            description: Filter description
        """
        super().__init__(mask_inside, name, description)
        self.top_left = top_left
        self.bottom_right = bottom_right
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create rectangular binary mask.
        
        Args:
            image: Input image to determine dimensions
            
        Returns:
            Binary mask with rectangle filled (255 inside, 0 outside)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, self.top_left, self.bottom_right, 255, -1)
        return mask


class CircleMaskFilter(ShapeFilter):
    """
    Circular mask filter with configurable center, radius, and inside/outside flag.
    """
    
    def __init__(
        self,
        center: Tuple[int, int],
        radius: int,
        mask_inside: bool = True,
        name: str = "Circle Mask Filter",
        description: str = ""
    ):
        """
        Initialize circle mask filter.
        
        Args:
            center: Center point of circle (x, y)
            radius: Radius of circle
            mask_inside: If True, keep pixels inside circle. If False, keep outside.
            name: Filter name
            description: Filter description
        """
        super().__init__(mask_inside, name, description)
        self.center = center
        self.radius = radius
    
    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create circular binary mask.
        
        Args:
            image: Input image to determine dimensions
            
        Returns:
            Binary mask with circle filled (255 inside, 0 outside)
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, self.center, self.radius, 255, -1)
        return mask
