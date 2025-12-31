"""
Base skill interface for bot interaction capabilities.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Skill(ABC):
    """
    Abstract base class for all skills.
    
    Skills represent things the bot can do to interact with the game,
    such as moving, pressing buttons, or executing macros.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the skill.
        
        Args:
            name: Human-readable name for this skill
            description: Optional description of what the skill does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute the skill with the given context.
        
        Args:
            context: Context dictionary containing required parameters
        """
        pass
    
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """
        Check if the skill can be executed with the given context.
        
        Override this method to add validation logic.
        
        Args:
            context: Context dictionary containing required parameters
            
        Returns:
            True if the skill can be executed, False otherwise
        """
        return True
    
    def __repr__(self) -> str:
        """String representation of the skill."""
        return f"{self.__class__.__name__}(name='{self.name}')"
