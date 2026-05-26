"""
gesture_control.py — Real-Time ViT Gesture Control System
===========================================================
Integrates webcam, hand detection, ViT model inference, and cursor
control into a single real-time loop.

Usage:
    python realtime/gesture_control.py
    python realtime/gesture_control.py --no-cursor   # Debug mode (no cursor movement)
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIT_CONFIG, PATHS, REALTIME, MEDIAPIPE,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES,
    GESTURE_CLASSES, GESTURE_LABELS, ACTION_MAP
)
from models.vit_model import build_vit_model
from realtime.hand_detector import HandDetector
from realtime.cursor_controller import CursorController
from realtime.gesture_smoother import CursorSmoother, GestureDebouncer, FPSCounter


class GestureControlSystem:
    """
    Real-time Vision Transformer gesture control system.

    Pipeline:
        Webcam → Hand Detection → ROI → Preprocess → ViT → Gesture → Cursor Action
    """

    def __init__(self, checkpoint_path=None, enable_cursor=True, show_debug=True):
        """
        Args:
            checkpoint_path: Path to ViT checkpoint (default: auto-detect)
            enable_cursor:   Whether to actually move the cursor
            show_debug:      Show debug visualization window
        """
        self.enable_cursor = enable_cursor
        self.show_debug    = show_debug

        self.config     = VIT_CONFIG
        self.input_size = self.config["input_size"]

        self.device = self._setup_device()
        self.model  = self._load_model(checkpoint_path)

        # Class names sorted to match ImageFolder ordering
        self.class_names = sorted(GESTURE_CLASSES.values())

        self.hand_detector = HandDetector(
            max_num_hands=MEDIAPIPE["max_num_hands"],
            min_detection_confidence=MEDIAPIPE["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE["min_tracking_confidence"],
            roi_padding=MEDIAPIPE["roi_padding"],
        )

        self.cursor_controller = CursorController(
            scroll_amount=REALTIME["scroll_amount"]
        ) if enable_cursor else None

        self.cursor_smoother   = CursorSmoother(window_size=REALTIME["smoothing_window"])
        self.gesture_debouncer = GestureDebouncer(
            debounce_frames=REALTIME["debounce_frames"],
            click_cooldown_ms=REALTIME["click_cooldown_ms"],
            confidence_threshold=REALTIME["confidence_threshold"],
        )
        self.fps_counter = FPSCounter()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])

        print(f"\n  ViT Gesture Control System initialized on {self.device}")
        if not enable_cursor:
            print("  ⚠ Cursor control DISABLED (debug mode)")

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("  Device: CPU")
        return device

    def _load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PATHS["checkpoints"], "best_vit_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train the model first: python training/train.py"
            )

        print(f"  Loading ViT model from: {checkpoint_path}")

        model = build_vit_model(
            model_name=self.config["model_name"],
            num_classes=NUM_CLASSES,
            pretrained=False,
            dropout=self.config["dropout"],
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', 0):.2f}%)")

        return model

    def preprocess_roi(self, roi):
        """Preprocess the hand ROI for ViT inference."""
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor  = self.transform(rgb_roi)
        tensor  = tensor.unsqueeze(0).to(self.device)
        return tensor

    def predict(self, tensor):
        """Run ViT inference and return (class_name, confidence)."""
        with torch.no_grad():
            outputs    = self.model(tensor)
            probs      = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probs.max(1)

        class_name = self.class_names[predicted_idx.item()]
        return class_name, confidence.item()

    def draw_debug_ui(self, frame, gesture_name, confidence, action_type,
                      fps, latency_ms, hand_detected):
        """Draw debug overlay on the frame."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 300, 0), (w, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        x0 = w - 290
        cv2.putText(frame, "Model: ViT (Vision Transformer)", (x0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (x0, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (x0, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        hand_txt   = "Hand: DETECTED" if hand_detected else "Hand: Not Found"
        hand_color = (0, 255, 0)      if hand_detected else (0, 0, 255)
        cv2.putText(frame, hand_txt, (x0, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        if gesture_name:
            label      = GESTURE_LABELS.get(gesture_name, gesture_name)
            conf_color = (0, 255, 0) if confidence > 0.8 else (0, 200, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.putText(frame, f"Gesture: {label}", (x0, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x0, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            cv2.putText(frame, f"Action: {action_type}", (x0, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        cv2.putText(frame, "Press 'Q' to quit | Move mouse to top-left for fail-safe",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return frame

    def run(self):
        """Main real-time loop."""
        print("\n" + "=" * 60)
        print("  REAL-TIME ViT GESTURE CONTROL")
        print("=" * 60)
        print("  Press 'Q' to quit")
        print("  Move mouse to top-left corner for emergency stop")
        print()

        cap = cv2.VideoCapture(REALTIME["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REALTIME["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REALTIME["camera_height"])

        if not cap.isOpened():
            print("ERROR: Cannot open webcam!")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                self.fps_counter.tick()

                detection    = self.hand_detector.detect(frame)
                gesture_name = None
                confidence   = 0.0
                action_type  = "none"

                if detection is not None:
                    # Extract ROI FIRST from clean frame (no landmarks)
                    roi = self.hand_detector.extract_roi(frame, detection['bbox'])

                    # Then draw landmarks for display only
                    if self.show_debug:
                        self.hand_detector.draw_landmarks(frame, detection['landmarks'])
                        self.hand_detector.draw_bbox(frame, detection['bbox'])

                    if roi is not None and roi.size > 0:
                        tensor       = self.preprocess_roi(roi)
                        gesture_name, confidence = self.predict(tensor)
                        action_type  = ACTION_MAP.get(gesture_name, "neutral")

                        should_trigger, _ = self.gesture_debouncer.process(
                            gesture_name, confidence, action_type
                        )

                        if should_trigger and self.cursor_controller:
                            raw_x, raw_y     = detection['index_tip']
                            smooth_x, smooth_y = self.cursor_smoother.smooth(raw_x, raw_y)
                            self.cursor_controller.execute_action(
                                action_type, smooth_x, smooth_y
                            )
                else:
                    self.cursor_smoother.reset()

                if self.show_debug:
                    frame = self.draw_debug_ui(
                        frame,
                        gesture_name, confidence, action_type,
                        self.fps_counter.fps,
                        self.fps_counter.latency_ms,
                        detection is not None,
                    )
                    cv2.imshow("ViT Gesture Control", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            if self.cursor_controller:
                self.cursor_controller.cleanup()
            print("\n  System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-time ViT gesture control")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to ViT checkpoint (default: checkpoints/best_vit_model.pth)')
    parser.add_argument('--no-cursor', action='store_true',
                        help='Disable cursor control (debug mode)')
    parser.add_argument('--no-debug', action='store_true',
                        help='Hide debug visualization window')

    args = parser.parse_args()

    system = GestureControlSystem(
        checkpoint_path=args.checkpoint,
        enable_cursor=not args.no_cursor,
        show_debug=not args.no_debug,
    )
    system.run()


if __name__ == "__main__":
    main()
