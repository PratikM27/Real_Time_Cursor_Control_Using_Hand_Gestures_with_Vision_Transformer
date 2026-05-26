"""
realtime_evaluate.py — Live Real-Time Accuracy Evaluation
==========================================================
Prompts you to perform each gesture in front of the webcam,
records model predictions vs ground truth, and calculates
real-time accuracy metrics with a confusion matrix.

Usage:
    python training/realtime_evaluate.py
    python training/realtime_evaluate.py --rounds 5 --hold 5
    python training/realtime_evaluate.py --checkpoint checkpoints/best_vit_model.pth

Controls during session:
    SPACE : Skip current gesture early
    Q     : Abort entire session
"""

import os
import sys
import argparse
import json
import time
import random

import cv2
import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIT_CONFIG, PATHS, REALTIME, MEDIAPIPE,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES,
    GESTURE_CLASSES, GESTURE_LABELS, SEED
)
from models.vit_model import build_vit_model
from realtime.hand_detector import HandDetector


# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────
COLOR_BG       = (30, 30, 30)
COLOR_WHITE    = (255, 255, 255)
COLOR_GREEN    = (0, 220, 100)
COLOR_YELLOW   = (0, 220, 255)
COLOR_RED      = (0, 0, 255)
COLOR_CYAN     = (255, 220, 0)
COLOR_GRAY     = (150, 150, 150)
COLOR_DARKGRAY = (80, 80, 80)


class RealTimeEvaluator:
    """
    Live evaluation system that prompts gestures and records predictions.
    
    Flow:
        1. COUNTDOWN (3 sec)  — "Get ready to show: OPEN PALM"
        2. RECORDING (N sec)  — Records predictions every frame
        3. TRANSITION (2 sec) — Brief rest before next gesture
        4. Repeat for all gestures × rounds
        5. Calculate & save metrics
    """

    def __init__(self, checkpoint_path=None, rounds=3, hold_seconds=5,
                 countdown_seconds=3, transition_seconds=2):
        self.rounds = rounds
        self.hold_seconds = hold_seconds
        self.countdown_seconds = countdown_seconds
        self.transition_seconds = transition_seconds

        self.config = VIT_CONFIG
        self.input_size = self.config["input_size"]

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            print("  Device: CPU")

        # Load model
        self._load_model(checkpoint_path)

        # Class names sorted to match ImageFolder ordering
        self.class_names = sorted(GESTURE_CLASSES.values())

        # Hand detector
        self.hand_detector = HandDetector(
            max_num_hands=MEDIAPIPE["max_num_hands"],
            min_detection_confidence=MEDIAPIPE["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE["min_tracking_confidence"],
            roi_padding=MEDIAPIPE["roi_padding"],
        )

        # Preprocessing (same as gesture_control.py)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])

        # Results storage
        self.all_true_labels = []
        self.all_pred_labels = []
        self.all_confidences = []
        self.per_trial_results = []

    def _load_model(self, checkpoint_path=None):
        """Load the ViT model from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PATHS["checkpoints"], "best_vit_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train the model first: python training/train.py"
            )

        print(f"  Loading model: {checkpoint_path}")

        self.model = build_vit_model(
            model_name=self.config["model_name"],
            num_classes=NUM_CLASSES,
            pretrained=False,
            dropout=self.config["dropout"],
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"  Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', 0):.2f}%)")

    def preprocess_roi(self, roi):
        """Preprocess hand ROI for inference (same as gesture_control.py)."""
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_roi)
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor

    def predict(self, tensor):
        """Run ViT inference → (class_name, confidence)."""
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = probs.max(1)
        class_name = self.class_names[predicted_idx.item()]
        return class_name, confidence.item()

    def _draw_overlay(self, frame, alpha=0.7):
        """Draw semi-transparent dark overlay on full frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), COLOR_BG, -1)
        cv2.addWeighted(overlay, 1 - alpha, frame, alpha, 0, frame)
        return frame

    def _draw_progress_bar(self, frame, x, y, width, height, progress, color):
        """Draw a progress bar."""
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_DARKGRAY, -1)
        fill_width = int(width * min(progress, 1.0))
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR_GRAY, 1)

    def _put_centered_text(self, frame, text, y, font_scale, color, thickness=2):
        """Put text centered horizontally."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def _draw_countdown_screen(self, frame, gesture_label, seconds_left,
                                trial_num, total_trials):
        """Draw the 'Get Ready' countdown screen."""
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - 250, 30), (w//2 + 250, 250), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        self._put_centered_text(frame, "GET READY", 80, 0.9, COLOR_YELLOW, 2)
        self._put_centered_text(frame, f"Show: {gesture_label}", 130, 1.2, COLOR_GREEN, 3)
        self._put_centered_text(frame, f"Starting in {seconds_left}...", 180, 0.8, COLOR_WHITE, 2)
        self._put_centered_text(frame, f"Trial {trial_num}/{total_trials}", 220, 0.6, COLOR_GRAY, 1)

        return frame

    def _draw_recording_screen(self, frame, gesture_label, predicted_label,
                                 confidence, time_left, frames_recorded,
                                 correct_frames, trial_num, total_trials,
                                 hand_detected):
        """Draw the recording screen with live prediction feedback."""
        h, w = frame.shape[:2]

        # Info panel at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Recording indicator (blinking red dot)
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(frame, (25, 25), 10, COLOR_RED, -1)
        cv2.putText(frame, "RECORDING", (42, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

        # Trial info
        cv2.putText(frame, f"Trial {trial_num}/{total_trials}", (w - 180, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GRAY, 1)

        # Expected gesture
        cv2.putText(frame, f"Expected: {gesture_label}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2)

        # Predicted gesture
        if hand_detected and predicted_label:
            is_correct = (predicted_label == gesture_label)
            pred_color = COLOR_GREEN if is_correct else COLOR_RED
            icon = "OK" if is_correct else "X"
            cv2.putText(frame, f"Predicted: {predicted_label} [{icon}]", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
        else:
            cv2.putText(frame, "Predicted: No hand detected", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Time progress bar at bottom
        progress = 1.0 - (time_left / self.hold_seconds)
        self._draw_progress_bar(frame, 20, h - 40, w - 40, 20, progress, COLOR_GREEN)
        cv2.putText(frame, f"{time_left:.1f}s left", (w // 2 - 40, h - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        # Frame counter
        if frames_recorded > 0:
            acc = correct_frames / frames_recorded * 100
            cv2.putText(frame, f"Frames: {frames_recorded} | Correct: {correct_frames} ({acc:.0f}%)",
                        (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

        return frame

    def _draw_transition_screen(self, frame, message, seconds_left):
        """Draw transition screen between gestures."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - 200, h//2 - 50), (w//2 + 200, h//2 + 50), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        self._put_centered_text(frame, message, h // 2, 0.8, COLOR_GREEN, 2)
        self._put_centered_text(frame, f"Next in {seconds_left}...", h // 2 + 40, 0.6, COLOR_GRAY, 1)

        return frame

    def _generate_trial_order(self):
        """Generate randomized trial order: each gesture appears `rounds` times."""
        random.seed(SEED)
        trials = []
        for _ in range(self.rounds):
            round_gestures = list(self.class_names)
            random.shuffle(round_gestures)
            trials.extend(round_gestures)
        return trials

    def run(self):
        """Main evaluation session."""
        trials = self._generate_trial_order()
        total_trials = len(trials)

        print("\n" + "=" * 60)
        print("  REAL-TIME LIVE EVALUATION")
        print("=" * 60)
        print(f"  Rounds:           {self.rounds}")
        print(f"  Gestures/round:   {len(self.class_names)}")
        print(f"  Total trials:     {total_trials}")
        print(f"  Hold time:        {self.hold_seconds}s per gesture")
        print(f"  Countdown:        {self.countdown_seconds}s")
        print()
        print("  Controls:")
        print("    SPACE — Skip current gesture")
        print("    Q     — Abort session")
        print()
        print("  Starting webcam...")

        cap = cv2.VideoCapture(REALTIME["camera_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REALTIME["camera_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REALTIME["camera_height"])

        if not cap.isOpened():
            print("  ERROR: Cannot open webcam!")
            return

        aborted = False

        try:
            for trial_idx, true_gesture in enumerate(trials):
                trial_num = trial_idx + 1
                gesture_label = GESTURE_LABELS.get(true_gesture, true_gesture)
                true_class_idx = self.class_names.index(true_gesture)

                # ── PHASE 1: COUNTDOWN ──
                countdown_end = time.time() + self.countdown_seconds
                while time.time() < countdown_end:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    seconds_left = max(0, int(countdown_end - time.time()) + 1)
                    frame = self._draw_countdown_screen(
                        frame, gesture_label, seconds_left,
                        trial_num, total_trials
                    )
                    cv2.imshow("Real-Time Evaluation", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        aborted = True
                        break
                    if key == ord(' '):
                        break

                if aborted:
                    break

                # ── PHASE 2: RECORDING ──
                record_end = time.time() + self.hold_seconds
                trial_preds = []
                trial_confs = []
                trial_correct = 0
                frames_recorded = 0

                while time.time() < record_end:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    # Detect hand and predict
                    detection = self.hand_detector.detect(frame)
                    predicted_name = None
                    confidence = 0.0
                    hand_detected = False

                    if detection is not None:
                        roi = self.hand_detector.extract_roi(frame, detection['bbox'])

                        # Draw landmarks for display
                        self.hand_detector.draw_landmarks(frame, detection['landmarks'])
                        self.hand_detector.draw_bbox(frame, detection['bbox'])

                        if roi is not None and roi.size > 0:
                            hand_detected = True
                            tensor = self.preprocess_roi(roi)
                            predicted_name, confidence = self.predict(tensor)

                            pred_class_idx = self.class_names.index(predicted_name)

                            # Record this frame
                            self.all_true_labels.append(true_class_idx)
                            self.all_pred_labels.append(pred_class_idx)
                            self.all_confidences.append(confidence)

                            trial_preds.append(pred_class_idx)
                            trial_confs.append(confidence)
                            frames_recorded += 1

                            if predicted_name == true_gesture:
                                trial_correct += 1

                    time_left = max(0, record_end - time.time())
                    predicted_label = GESTURE_LABELS.get(predicted_name, "") if predicted_name else ""
                    frame = self._draw_recording_screen(
                        frame, gesture_label, predicted_label,
                        confidence, time_left, frames_recorded,
                        trial_correct, trial_num, total_trials,
                        hand_detected
                    )
                    cv2.imshow("Real-Time Evaluation", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        aborted = True
                        break
                    if key == ord(' '):
                        break

                if aborted:
                    break

                # Record trial summary
                trial_acc = (trial_correct / frames_recorded * 100) if frames_recorded > 0 else 0
                avg_conf = np.mean(trial_confs) if trial_confs else 0
                self.per_trial_results.append({
                    'trial': trial_num,
                    'gesture': true_gesture,
                    'gesture_label': gesture_label,
                    'frames_recorded': frames_recorded,
                    'correct_frames': trial_correct,
                    'trial_accuracy': trial_acc,
                    'avg_confidence': avg_conf,
                })

                print(f"  Trial {trial_num:2d}/{total_trials}: {gesture_label:20s} → "
                      f"{trial_correct}/{frames_recorded} correct "
                      f"({trial_acc:.1f}%, conf={avg_conf:.2f})")

                # ── PHASE 3: TRANSITION ──
                if trial_idx < total_trials - 1:
                    transition_end = time.time() + self.transition_seconds
                    while time.time() < transition_end:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.flip(frame, 1)
                        secs = max(0, int(transition_end - time.time()) + 1)
                        frame = self._draw_transition_screen(frame, "Good! Rest...", secs)
                        cv2.imshow("Real-Time Evaluation", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):
                            aborted = True
                            break
                    if aborted:
                        break

        except KeyboardInterrupt:
            print("\n  Session interrupted by user.")
            aborted = True

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()

        if aborted and len(self.all_true_labels) == 0:
            print("\n  No data recorded. Exiting.")
            return

        # Calculate and save results
        self._calculate_results()

    def _calculate_results(self):
        """Calculate metrics and save results."""
        true_arr = np.array(self.all_true_labels)
        pred_arr = np.array(self.all_pred_labels)
        conf_arr = np.array(self.all_confidences)

        print("\n" + "=" * 60)
        print("  REAL-TIME EVALUATION RESULTS")
        print("=" * 60)

        # Overall metrics
        accuracy = accuracy_score(true_arr, pred_arr) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_arr, pred_arr, average='macro', zero_division=0
        )
        precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
            true_arr, pred_arr, average=None, zero_division=0,
            labels=list(range(len(self.class_names)))
        )

        print(f"\n  Total Frames Evaluated: {len(true_arr)}")
        print(f"  Average Confidence:    {conf_arr.mean():.2%}")
        print(f"\n  Overall Accuracy:      {accuracy:.2f}%")
        print(f"  Precision (macro):     {precision * 100:.2f}%")
        print(f"  Recall (macro):        {recall * 100:.2f}%")
        print(f"  F1-Score (macro):      {f1 * 100:.2f}%")

        # Per-class
        print(f"\n  Per-Class Performance:")
        print(f"  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("  " + "-" * 60)
        for i, class_name in enumerate(self.class_names):
            label = GESTURE_LABELS.get(class_name, class_name)
            print(f"  {label:<20s} {precision_pc[i]*100:>9.2f}% {recall_pc[i]*100:>9.2f}% "
                  f"{f1_pc[i]*100:>9.2f}% {int(support_pc[i]):>9d}")

        # Classification report
        report = classification_report(
            true_arr, pred_arr,
            target_names=self.class_names,
            zero_division=0
        )
        print(f"\n  Full Classification Report:")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(true_arr, pred_arr, labels=list(range(len(self.class_names))))
        self._plot_confusion_matrix(cm)

        # Load test-set results for comparison
        self._print_comparison(accuracy, precision * 100, recall * 100, f1 * 100)

        # Save results
        self._save_results(accuracy, precision * 100, recall * 100, f1 * 100,
                           precision_pc, recall_pc, f1_pc, support_pc,
                           cm, conf_arr)

    def _plot_confusion_matrix(self, cm):
        """Plot and save real-time confusion matrix."""
        plt.figure(figsize=(10, 8))
        display_names = [GESTURE_LABELS.get(name, name) for name in self.class_names]

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=display_names,
            yticklabels=display_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        plt.title('Real-Time Live Evaluation — Confusion Matrix',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual (Prompted)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(PATHS["confusion_matrices"],
                                 "vit_realtime_confusion_matrix.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Confusion matrix saved: {save_path}")

    def _print_comparison(self, rt_acc, rt_prec, rt_recall, rt_f1):
        """Print side-by-side comparison with test-set results."""
        test_results_path = os.path.join(PATHS["results"], "vit_eval_results.json")

        print("\n" + "=" * 60)
        print("  COMPARISON: Test Set vs Real-Time")
        print("=" * 60)

        if os.path.exists(test_results_path):
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)

            ts_acc = test_results.get('accuracy', 0)
            ts_prec = test_results.get('precision_macro', 0)
            ts_recall = test_results.get('recall_macro', 0)
            ts_f1 = test_results.get('f1_macro', 0)

            print(f"\n  {'Metric':<22s} {'Test Set':>12s} {'Real-Time':>12s} {'Diff':>10s}")
            print("  " + "-" * 58)
            print(f"  {'Accuracy':<22s} {ts_acc:>11.2f}% {rt_acc:>11.2f}% {rt_acc - ts_acc:>+9.2f}%")
            print(f"  {'Precision (macro)':<22s} {ts_prec:>11.2f}% {rt_prec:>11.2f}% {rt_prec - ts_prec:>+9.2f}%")
            print(f"  {'Recall (macro)':<22s} {ts_recall:>11.2f}% {rt_recall:>11.2f}% {rt_recall - ts_recall:>+9.2f}%")
            print(f"  {'F1-Score (macro)':<22s} {ts_f1:>11.2f}% {rt_f1:>11.2f}% {rt_f1 - ts_f1:>+9.2f}%")
        else:
            print("  Test set results not found. Run 'python training/evaluate.py' first.")
            print(f"\n  Real-Time Accuracy:  {rt_acc:.2f}%")
            print(f"  Real-Time Precision: {rt_prec:.2f}%")
            print(f"  Real-Time Recall:    {rt_recall:.2f}%")
            print(f"  Real-Time F1-Score:  {rt_f1:.2f}%")

    def _save_results(self, accuracy, precision, recall, f1,
                      precision_pc, recall_pc, f1_pc, support_pc,
                      cm, conf_arr):
        """Save all results to JSON."""
        results = {
            'evaluation_type': 'real-time_live',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'rounds': self.rounds,
                'hold_seconds': self.hold_seconds,
                'countdown_seconds': self.countdown_seconds,
            },
            'total_frames': len(self.all_true_labels),
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'avg_confidence': float(conf_arr.mean()),
            'min_confidence': float(conf_arr.min()),
            'max_confidence': float(conf_arr.max()),
            'per_class': {},
            'confusion_matrix': cm.tolist(),
            'per_trial': self.per_trial_results,
        }

        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision_pc[i] * 100),
                'recall': float(recall_pc[i] * 100),
                'f1': float(f1_pc[i] * 100),
                'support': int(support_pc[i]),
            }

        save_path = os.path.join(PATHS["results"], "vit_realtime_eval_results.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {save_path}")

        print("\n" + "=" * 60)
        print("  EVALUATION COMPLETE")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time live accuracy evaluation for ViT gesture model"
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of rounds (each round tests all 7 gestures)')
    parser.add_argument('--hold', type=int, default=5,
                        help='Seconds to hold each gesture')
    parser.add_argument('--countdown', type=int, default=3,
                        help='Countdown seconds before recording')
    args = parser.parse_args()

    evaluator = RealTimeEvaluator(
        checkpoint_path=args.checkpoint,
        rounds=args.rounds,
        hold_seconds=args.hold,
        countdown_seconds=args.countdown,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
