"""
Text input dialog component using OpenCV.
"""
import cv2
import numpy as np
from typing import Optional


class TextInputDialog:
    """
    OpenCV-based text input dialog for user text input.
    
    Provides a visual text input window that blocks until the user
    submits or cancels the input.
    """
    
    def __init__(self, window_name: str = "Text Input"):
        """
        Initialize text input dialog.
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        self._input_text: str = ""
        self._cursor_pos: int = 0
        self._result: Optional[str] = None
        self._cancelled: bool = False
        self._active: bool = False
    
    def prompt(self, title: str = "Enter Text", default: str = "") -> Optional[str]:
        """
        Show text input dialog and return user input.
        
        Args:
            title: Dialog title text
            default: Default text value
            
        Returns:
            User input string, or None if cancelled
        """
        self._input_text = default
        self._cursor_pos = len(default)
        self._result = None
        self._cancelled = False
        self._active = True
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # Set mouse callback for window focus
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        
        while self._active:
            self._draw(title)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self._cancelled = True
                self._active = False
            elif key == 13:  # Enter
                self._active = False
            elif key == 8:  # Backspace
                if self._cursor_pos > 0:
                    self._input_text = (
                        self._input_text[:self._cursor_pos - 1] +
                        self._input_text[self._cursor_pos:]
                    )
                    self._cursor_pos -= 1
            elif key == 255:  # Delete (may need adjustment based on system)
                pass  # Not handling delete for now
            elif 32 <= key <= 126:  # Printable ASCII
                char = chr(key)
                self._input_text = (
                    self._input_text[:self._cursor_pos] +
                    char +
                    self._input_text[self._cursor_pos:]
                )
                self._cursor_pos += 1
            elif key == 81:  # Left arrow
                if self._cursor_pos > 0:
                    self._cursor_pos -= 1
            elif key == 83:  # Right arrow
                if self._cursor_pos < len(self._input_text):
                    self._cursor_pos += 1
        
        cv2.destroyWindow(self.window_name)
        
        if self._cancelled:
            return None
        
        return self._input_text
    
    def _draw(self, title: str):
        """
        Draw the dialog window.
        
        Args:
            title: Title text to display
        """
        # Window dimensions
        width = 600
        height = 200
        
        # Create canvas
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # Draw border
        cv2.rectangle(canvas, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
        
        # Draw title
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(title, font, font_scale, thickness)
        title_x = (width - text_width) // 2
        cv2.putText(canvas, title, (title_x, 40), font, font_scale, (0, 0, 0), thickness)
        
        # Draw input box
        box_y = 80
        box_height = 40
        box_x1 = 30
        box_x2 = width - 30
        cv2.rectangle(canvas, (box_x1, box_y), (box_x2, box_y + box_height), (255, 255, 255), -1)
        cv2.rectangle(canvas, (box_x1, box_y), (box_x2, box_y + box_height), (0, 0, 0), 2)
        
        # Draw text
        if self._input_text:
            text_y = box_y + box_height - 10
            cv2.putText(canvas, self._input_text, (box_x1 + 5, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw cursor
        if self._active:
            cursor_x = box_x1 + 5
            if self._input_text:
                # Calculate cursor position based on text
                prefix = self._input_text[:self._cursor_pos]
                (cursor_width, _), _ = cv2.getTextSize(prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cursor_x = box_x1 + 5 + cursor_width
            
            cursor_y1 = box_y + 5
            cursor_y2 = box_y + box_height - 5
            cv2.line(canvas, (cursor_x, cursor_y1), (cursor_x, cursor_y2), (0, 0, 0), 2)
        
        # Draw instructions
        instruction = "Press Enter to confirm, ESC to cancel"
        (inst_width, inst_height), _ = cv2.getTextSize(instruction, font, 0.5, 1)
        inst_x = (width - inst_width) // 2
        cv2.putText(canvas, instruction, (inst_x, height - 20),
                   font, 0.5, (100, 100, 100), 1)
        
        cv2.imshow(self.window_name, canvas)
    
    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback for window interaction."""
        pass  # Can be extended for click-to-focus functionality
