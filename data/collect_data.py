"""
collect_data.py — Webcam-Based Gesture Data Collection
=======================================================
Captures hand gesture images via webcam using MediaPipe hand detection.
Images are saved as cropped ROI patches per gesture class.

Usage:
    python data/collect_data.py

Controls:
    0-6  : Select gesture class
    SPACE: Toggle auto-capture mode
    S    : Save single frame
    Q    : Quit
"""

import os
import sys
import cv2
import time
import mediapipe as mp
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GESTURE_CLASSES, GESTURE_LABELS, DATA_COLLECTION,
    MEDIAPIPE, PATHS
)


def create_class_directories(base_dir):
    """Create subdirectories for each gesture class."""
    for class_name in GESTURE_CLASSES.values():
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    return base_dir


def count_images(base_dir):
    """Count images per class in the given directory."""
    counts = {}
    for class_name in GESTURE_CLASSES.values():
        class_dir = os.path.join(base_dir, class_name)
        if os.path.exists(class_dir):
            counts[class_name] = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
        else:
            counts[class_name] = 0
    return counts


def extract_hand_roi(frame, hand_landmarks, padding=20):
    """
    Extract the hand Region of Interest from the frame.
    
    Args:
        frame: BGR image (H, W, 3)
        hand_landmarks: MediaPipe hand landmarks
        padding: Extra pixels around the bounding box
    
    Returns:
        roi: Cropped hand image, or None if invalid
        bbox: (x_min, y_min, x_max, y_max)
    """
    h, w, _ = frame.shape
    
    # Get bounding box from landmarks
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Validate bounding box
    if x_max - x_min < 20 or y_max - y_min < 20:
        return None, None
    
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)


def draw_ui(frame, current_class, auto_capture, counts, fps, hand_detected):
    """Draw the collection UI overlay on the frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, "GESTURE DATA COLLECTOR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "-" * 35, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Current class
    class_name = GESTURE_CLASSES.get(current_class, "None")
    label = GESTURE_LABELS.get(class_name, class_name)
    color = (0, 255, 0) if auto_capture else (0, 200, 255)
    cv2.putText(frame, f"Class [{current_class}]: {label}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Auto-capture status
    status = "AUTO-CAPTURE ON" if auto_capture else "Manual Mode"
    status_color = (0, 255, 0) if auto_capture else (100, 100, 255)
    cv2.putText(frame, f"Mode: {status}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Hand detection status
    hand_status = "HAND DETECTED" if hand_detected else "No Hand"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, hand_status, (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Class counts
    cv2.putText(frame, "Image Counts:", (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    target = DATA_COLLECTION["images_per_class"]
    for i, (class_id, class_name) in enumerate(GESTURE_CLASSES.items()):
        count = counts.get(class_name, 0)
        progress = min(count / target, 1.0)
        label_text = GESTURE_LABELS.get(class_name, class_name)
        
        # Color based on progress
        if progress >= 1.0:
            c = (0, 255, 0)
        elif progress >= 0.5:
            c = (0, 200, 255)
        else:
            c = (0, 100, 255)
        
        # Highlight current class
        prefix = ">" if class_id == current_class else " "
        text = f"{prefix}[{class_id}] {label_text}: {count}/{target}"
        cv2.putText(frame, text, (10, 210 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        
        # Progress bar
        bar_x = 10
        bar_y = 218 + i * 25
        bar_w = int(progress * 100)
        cv2.rectangle(frame, (bar_x + 200, bar_y), (bar_x + 300, bar_y + 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x + 200, bar_y), (bar_x + 200 + bar_w, bar_y + 5), c, -1)
    
    # Instructions
    y_start = 400
    instructions = [
        "Controls:",
        "  0-6  : Select class",
        "  SPACE: Toggle auto-capture",
        "  S    : Save single frame",
        "  Q    : Quit",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, y_start + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    return frame


def main():
    """Main data collection loop."""
    print("=" * 60)
    print("  GESTURE DATA COLLECTION TOOL")
    print("=" * 60)
    print()
    
    # Setup
    save_dir = PATHS["raw_data"]
    create_class_directories(save_dir)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MEDIAPIPE["max_num_hands"],
        min_detection_confidence=MEDIAPIPE["min_detection_confidence"],
        min_tracking_confidence=MEDIAPIPE["min_tracking_confidence"],
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(DATA_COLLECTION["camera_id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DATA_COLLECTION["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DATA_COLLECTION["camera_height"])
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    print(f"Webcam opened. Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"Saving to: {save_dir}")
    print(f"Target: {DATA_COLLECTION['images_per_class']} images per class")
    print()
    
    # State
    current_class = 0
    auto_capture = False
    last_capture_time = 0
    capture_delay = DATA_COLLECTION["capture_delay_ms"] / 1000.0
    frame_count = 0
    fps = 0.0
    fps_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame!")
            break
        
        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # FPS calculation
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        hand_detected = False
        roi = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_detected = True
            
            # Keep a clean copy before drawing landmarks (for saving clean ROI)
            clean_frame = frame.copy()
            
            # Draw landmarks on frame (for display only)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # Extract ROI from clean frame (no landmarks)
            roi, bbox = extract_hand_roi(clean_frame, hand_landmarks, MEDIAPIPE["roi_padding"])
            
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Count current images
        counts = count_images(save_dir)
        
        # Auto-capture logic
        if auto_capture and hand_detected and roi is not None:
            current_time = time.time()
            if current_time - last_capture_time >= capture_delay:
                class_name = GESTURE_CLASSES[current_class]
                class_dir = os.path.join(save_dir, class_name)
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, roi)
                last_capture_time = current_time
                
                # Flash effect
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                             (0, 255, 0), 5)
        
        # Draw UI
        frame = draw_ui(frame, current_class, auto_capture, counts, fps, hand_detected)
        
        # Show frame
        cv2.imshow("Gesture Data Collector", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):  # Space - toggle auto-capture
            auto_capture = not auto_capture
            mode = "ON" if auto_capture else "OFF"
            print(f"Auto-capture: {mode}")
        elif key == ord('s') or key == ord('S'):  # Save single frame
            if hand_detected and roi is not None:
                class_name = GESTURE_CLASSES[current_class]
                class_dir = os.path.join(save_dir, class_name)
                timestamp = int(time.time() * 1000)
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, roi)
                print(f"Saved: {filename}")
        elif ord('0') <= key <= ord('6'):
            current_class = key - ord('0')
            class_name = GESTURE_CLASSES[current_class]
            label = GESTURE_LABELS.get(class_name, class_name)
            print(f"Selected class [{current_class}]: {label}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    # Print final counts
    print("\n" + "=" * 40)
    print("  FINAL IMAGE COUNTS")
    print("=" * 40)
    counts = count_images(save_dir)
    total = 0
    for class_id, class_name in GESTURE_CLASSES.items():
        count = counts.get(class_name, 0)
        label = GESTURE_LABELS.get(class_name, class_name)
        print(f"  [{class_id}] {label:20s}: {count}")
        total += count
    print(f"\n  TOTAL: {total}")
    print("=" * 40)


if __name__ == "__main__":
    main()
