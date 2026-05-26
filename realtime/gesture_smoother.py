"""
gesture_smoother.py — Prediction Smoothing & Debouncing
=========================================================
Smooths jittery cursor movements and prevents accidental
gesture triggers through debouncing logic.
"""

import time
from collections import deque
import numpy as np


class CursorSmoother:
    """
    Moving average filter for smooth cursor movement.
    Reduces jitter from hand tremors.
    """
    
    def __init__(self, window_size=5):
        """
        Args:
            window_size: Number of positions to average
        """
        self.window_size = window_size
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
    
    def smooth(self, x, y):
        """
        Add new position and return smoothed coordinates.
        
        Args:
            x, y: Raw position (normalized 0-1 or pixel coordinates)
        
        Returns:
            smooth_x, smooth_y: Smoothed position
        """
        self.x_history.append(x)
        self.y_history.append(y)
        
        smooth_x = np.mean(self.x_history)
        smooth_y = np.mean(self.y_history)
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Clear position history."""
        self.x_history.clear()
        self.y_history.clear()


class GestureDebouncer:
    """
    Debounces gesture predictions to prevent accidental triggers.
    
    Requires N consecutive frames of the same gesture before
    triggering an action. Implements cooldown for click-type actions.
    """
    
    def __init__(self, debounce_frames=3, click_cooldown_ms=500,
                 confidence_threshold=0.6):
        """
        Args:
            debounce_frames: Required consecutive same-predictions
            click_cooldown_ms: Minimum time between click actions
            confidence_threshold: Minimum softmax confidence to accept
        """
        self.debounce_frames = debounce_frames
        self.click_cooldown_ms = click_cooldown_ms
        self.confidence_threshold = confidence_threshold
        
        self.gesture_counts = {}
        self.last_gesture = None
        self.last_action_time = {}
        
        # Actions that need cooldown (click-type)
        self.cooldown_actions = {
            'left_click', 'right_click', 'drag',
            'scroll_up', 'scroll_down'
        }
    
    def process(self, predicted_class, confidence, action_type):
        """
        Process a prediction and decide whether to trigger action.
        
        Args:
            predicted_class: Predicted gesture class name
            confidence: Softmax confidence (0-1)
            action_type: Mapped action type string
        
        Returns:
            should_trigger: True if action should be executed
            stable_gesture: The stabilized gesture class name
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.gesture_counts = {}
            return False, None
        
        # Count consecutive predictions
        if predicted_class == self.last_gesture:
            self.gesture_counts[predicted_class] = \
                self.gesture_counts.get(predicted_class, 0) + 1
        else:
            self.gesture_counts = {predicted_class: 1}
            self.last_gesture = predicted_class
        
        count = self.gesture_counts.get(predicted_class, 0)
        
        # Move cursor doesn't need debounce
        if action_type == 'move_cursor':
            return True, predicted_class
        
        # Neutral doesn't trigger any action
        if action_type == 'neutral':
            return False, predicted_class
        
        # Drag: trigger after debounce, keep triggering continuously
        if action_type == 'drag':
            return count >= self.debounce_frames, predicted_class
        
        # Check debounce
        if count < self.debounce_frames:
            return False, predicted_class
        
        # Scroll actions: trigger repeatedly with cooldown (no counter reset)
        if action_type in ('scroll_up', 'scroll_down'):
            current_time = time.time() * 1000
            last_time = self.last_action_time.get(action_type, 0)
            
            if current_time - last_time < self.click_cooldown_ms:
                return False, predicted_class
            
            self.last_action_time[action_type] = current_time
            # Don't reset counter — keeps triggering while gesture is held
            return True, predicted_class
        
        # Click actions (left_click, right_click): trigger once, then cooldown
        if action_type in ('left_click', 'right_click'):
            current_time = time.time() * 1000
            last_time = self.last_action_time.get(action_type, 0)
            
            if current_time - last_time < self.click_cooldown_ms:
                return False, predicted_class
            
            self.last_action_time[action_type] = current_time
            # Reset only this gesture's count so it needs fresh debounce
            self.gesture_counts[predicted_class] = 0
            return True, predicted_class
        
        return True, predicted_class
    
    def reset(self):
        """Reset all state."""
        self.gesture_counts = {}
        self.last_gesture = None
        self.last_action_time = {}


class FPSCounter:
    """Tracks frames per second."""
    
    def __init__(self, avg_window=30):
        self.times = deque(maxlen=avg_window)
        self.last_time = time.perf_counter()
    
    def tick(self):
        """Call once per frame."""
        current = time.perf_counter()
        self.times.append(current - self.last_time)
        self.last_time = current
    
    @property
    def fps(self):
        """Get current FPS."""
        if len(self.times) == 0:
            return 0.0
        return 1.0 / max(np.mean(self.times), 1e-6)
    
    @property
    def latency_ms(self):
        """Get average latency in ms."""
        if len(self.times) == 0:
            return 0.0
        return np.mean(self.times) * 1000
