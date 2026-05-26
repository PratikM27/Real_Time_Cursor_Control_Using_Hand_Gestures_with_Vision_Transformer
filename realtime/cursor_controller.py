"""
cursor_controller.py — PyAutoGUI Cursor Control
==================================================
Maps gesture predictions to OS-level cursor actions
using PyAutoGUI.
"""

import pyautogui

# Safety: enable fail-safe (move mouse to top-left corner to abort)
pyautogui.FAILSAFE = True
# Disable pause between PyAutoGUI calls for lower latency
pyautogui.PAUSE = 0.01


class CursorController:
    """
    Controls the system cursor based on gesture predictions.
    
    Maps gesture class names to cursor actions:
        open_palm         → move cursor
        index_point       → left click
        two_fingers_up    → right click
        fist              → neutral (no action)
        pinch             → drag (hold left button)
        three_fingers_up  → scroll up
        three_fingers_down→ scroll down
    """
    
    def __init__(self, scroll_amount=3):
        """
        Args:
            scroll_amount: Number of scroll units per trigger
        """
        self.screen_w, self.screen_h = pyautogui.size()
        self.scroll_amount = scroll_amount
        self.is_dragging = False
    
    def move_cursor(self, norm_x, norm_y):
        """
        Move cursor to position mapped from hand coordinates.
        
        Args:
            norm_x, norm_y: Normalized hand position (0-1)
        """
        # Flip x because webcam is mirrored
        screen_x = int(norm_x * self.screen_w)
        screen_y = int(norm_y * self.screen_h)
        
        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, self.screen_w - 1))
        screen_y = max(0, min(screen_y, self.screen_h - 1))
        
        # Stop drag if we were dragging
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        
        pyautogui.moveTo(screen_x, screen_y, _pause=False)
    
    def left_click(self, norm_x=None, norm_y=None):
        """Perform left click at current position."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        pyautogui.click()
    
    def right_click(self, norm_x=None, norm_y=None):
        """Perform right click at current position."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
        pyautogui.rightClick()
    
    def start_drag(self, norm_x=None, norm_y=None):
        """Start or continue dragging — holds mouse button + moves cursor."""
        if not self.is_dragging:
            pyautogui.mouseDown()
            self.is_dragging = True
        
        # Move cursor while dragging
        if norm_x is not None and norm_y is not None:
            screen_x = int(norm_x * self.screen_w)
            screen_y = int(norm_y * self.screen_h)
            screen_x = max(0, min(screen_x, self.screen_w - 1))
            screen_y = max(0, min(screen_y, self.screen_h - 1))
            pyautogui.moveTo(screen_x, screen_y, _pause=False)
    
    def scroll_up(self, norm_x=None, norm_y=None):
        """Scroll up."""
        pyautogui.scroll(self.scroll_amount)
    
    def scroll_down(self, norm_x=None, norm_y=None):
        """Scroll down."""
        pyautogui.scroll(-self.scroll_amount)
    
    def neutral(self, norm_x=None, norm_y=None):
        """No action (fist gesture)."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
    
    def execute_action(self, action_type, norm_x=0.5, norm_y=0.5):
        """
        Execute a cursor action.
        
        Args:
            action_type: One of: 'move_cursor', 'left_click', 'right_click',
                        'neutral', 'drag', 'scroll_up', 'scroll_down'
            norm_x, norm_y: Normalized hand position (0-1)
        """
        action_map = {
            'move_cursor': self.move_cursor,
            'left_click':  self.left_click,
            'right_click': self.right_click,
            'neutral':     self.neutral,
            'drag':        self.start_drag,
            'scroll_up':   self.scroll_up,
            'scroll_down': self.scroll_down,
        }
        
        action_fn = action_map.get(action_type)
        if action_fn:
            if action_type in ('move_cursor', 'drag'):
                action_fn(norm_x, norm_y)
            else:
                action_fn()
    
    def cleanup(self):
        """Release any held buttons."""
        if self.is_dragging:
            pyautogui.mouseUp()
            self.is_dragging = False
