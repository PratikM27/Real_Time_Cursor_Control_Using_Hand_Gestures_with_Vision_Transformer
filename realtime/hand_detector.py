"""
hand_detector.py — MediaPipe Hand Detection Module
=====================================================
Wraps MediaPipe Hands for real-time hand detection,
landmark extraction, and ROI cropping.
"""

import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    """
    MediaPipe-based hand detector.
    
    Detects a single hand, extracts landmarks, computes bounding box,
    and crops the Region of Interest (ROI).
    """
    
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7,
                 min_tracking_confidence=0.5, roi_padding=20):
        """
        Args:
            max_num_hands: Maximum hands to detect
            min_detection_confidence: Detection threshold
            min_tracking_confidence: Tracking threshold
            roi_padding: Extra pixels around bounding box
        """
        self.roi_padding = roi_padding
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        # Key landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
    
    def detect(self, frame):
        """
        Detect hand in frame.
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            result: Dict with keys:
                - 'detected': bool
                - 'landmarks': MediaPipe landmarks object
                - 'bbox': (x_min, y_min, x_max, y_max) in pixels
                - 'center': (cx, cy) normalized (0-1)
                - 'index_tip': (x, y) normalized position of index finger
            Returns None if no hand detected.
        """
        h, w, _ = frame.shape
        
        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        landmarks = results.multi_hand_landmarks[0]
        
        # Compute bounding box
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min = max(0, int(min(x_coords)) - self.roi_padding)
        y_min = max(0, int(min(y_coords)) - self.roi_padding)
        x_max = min(w, int(max(x_coords)) + self.roi_padding)
        y_max = min(h, int(max(y_coords)) + self.roi_padding)
        
        # Hand center (normalized 0-1)
        cx = np.mean(x_coords) / w
        cy = np.mean(y_coords) / h
        
        # Index fingertip position (for cursor control)
        index_tip = landmarks.landmark[self.INDEX_TIP]
        
        return {
            'detected': True,
            'landmarks': landmarks,
            'bbox': (x_min, y_min, x_max, y_max),
            'center': (cx, cy),
            'index_tip': (index_tip.x, index_tip.y),
        }
    
    def extract_roi(self, frame, bbox):
        """
        Crop the hand Region of Interest from the frame.
        
        Args:
            frame: BGR image
            bbox: (x_min, y_min, x_max, y_max)
        
        Returns:
            roi: Cropped BGR image, or None if too small
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Validate size
        if x_max - x_min < 20 or y_max - y_min < 20:
            return None
        
        roi = frame[y_min:y_max, x_min:x_max].copy()
        return roi
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks and connections on the frame.
        
        Args:
            frame: BGR image (modified in place)
            landmarks: MediaPipe hand landmarks
        
        Returns:
            frame: Annotated frame
        """
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        return frame
    
    def draw_bbox(self, frame, bbox, label="", color=(0, 255, 0)):
        """Draw bounding box on frame."""
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        if label:
            cv2.putText(frame, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
