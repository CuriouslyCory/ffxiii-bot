"""
Menu system for managing state menus and hotkeys.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from .input_handler import InputHandler


@dataclass
class MenuItem:
    """Represents a single menu item with key binding and description."""
    key: str  # Key character (e.g., 'r', 'p')
    description: str  # Human-readable description
    action: str  # Action identifier
    enabled: bool = True  # Whether this menu item is enabled


@dataclass
class MenuDefinition:
    """Represents a complete menu definition for a state."""
    items: List[MenuItem] = field(default_factory=list)
    title: str = "Menu"
    
    def to_key_bindings(self) -> Dict[str, str]:
        """
        Convert menu definition to key bindings dictionary.
        
        Returns:
            Dictionary mapping keys to action identifiers
        """
        bindings = {}
        for item in self.items:
            if item.enabled:
                bindings[item.key] = item.action
        return bindings


class MenuManager:
    """
    Manages state menus and hotkeys.
    
    Handles attach/detach of keyboard listeners for states,
    supports sub-state menu layering, and manages menu display.
    """
    
    def __init__(self, input_handler: InputHandler):
        """
        Initialize menu manager.
        
        Args:
            input_handler: InputHandler instance to use for key bindings
        """
        self.input_handler = input_handler
        self._current_menus: List[MenuDefinition] = []  # Stack for sub-state menus
    
    def attach_menu(self, menu: MenuDefinition):
        """
        Attach a menu to the input handler.
        
        If a menu is already attached, this adds to the stack (for sub-states).
        
        Args:
            menu: Menu definition to attach
        """
        self._current_menus.append(menu)
        self._update_key_bindings()
    
    def detach_menu(self):
        """
        Detach the most recently attached menu.
        
        If multiple menus are stacked (sub-state), removes the top one
        and restores the parent menu bindings.
        """
        if self._current_menus:
            self._current_menus.pop()
            self._update_key_bindings()
    
    def detach_all_menus(self):
        """Detach all menus and clear key bindings."""
        self._current_menus.clear()
        self.input_handler.clear_key_bindings()
    
    def _update_key_bindings(self):
        """Update input handler key bindings based on current menu stack."""
        if not self._current_menus:
            self.input_handler.clear_key_bindings()
            return
        
        # Merge all menus in the stack (later menus override earlier ones)
        combined_bindings = {}
        for menu in self._current_menus:
            menu_bindings = menu.to_key_bindings()
            combined_bindings.update(menu_bindings)
        
        self.input_handler.set_key_bindings(combined_bindings)
    
    def print_menu(self):
        """Print the current menu to console."""
        if not self._current_menus:
            return
        
        # Print all menus in stack (most recent first)
        for i, menu in enumerate(reversed(self._current_menus)):
            if i > 0:
                print(f"\n--- Sub-menu ---")
            else:
                print(f"\n--- {menu.title} ---")
            
            for item in menu.items:
                if item.enabled:
                    print(f"  '{item.key}': {item.description}")
        
        # Always show ESC for stop
        print("  'ESC': Stop/Cancel")
    
    def get_current_menu(self) -> Optional[MenuDefinition]:
        """
        Get the most recently attached menu.
        
        Returns:
            Current menu definition, or None if no menu attached
        """
        return self._current_menus[-1] if self._current_menus else None
