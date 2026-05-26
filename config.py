"""
config.py — Central Configuration for Gesture Control Project
==============================================================
All hyperparameters, paths, and gesture class definitions.
Vision Transformer (ViT) based gesture recognition system.
"""

import os
import torch

# ─────────────────────────────────────────────
# Project Root
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Device Auto-Detection
# ─────────────────────────────────────────────
def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# ─────────────────────────────────────────────
# Random Seed (Reproducibility)
# ─────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────
# Gesture Classes
# ─────────────────────────────────────────────
GESTURE_CLASSES = {
    0: "open_palm",
    1: "index_point",
    2: "two_fingers_up",
    3: "fist",
    4: "pinch",
    5: "three_fingers_up",
    6: "three_fingers_down",
}

NUM_CLASSES = len(GESTURE_CLASSES)

# Reverse mapping: name → id
GESTURE_NAME_TO_ID = {v: k for k, v in GESTURE_CLASSES.items()}

# Human-readable labels for display
GESTURE_LABELS = {
    "open_palm":          "Open Palm",
    "index_point":        "Index Point",
    "two_fingers_up":     "Two Fingers Up",
    "fist":               "Fist",
    "pinch":              "Pinch",
    "three_fingers_up":   "Three Fingers Up",
    "three_fingers_down": "Three Fingers Down",
}

# ─────────────────────────────────────────────
# Gesture → Cursor Action Mapping
# ─────────────────────────────────────────────
ACTION_MAP = {
    "open_palm":          "move_cursor",
    "index_point":        "left_click",
    "two_fingers_up":     "right_click",
    "fist":               "neutral",
    "pinch":              "drag",
    "three_fingers_up":   "scroll_up",
    "three_fingers_down": "scroll_down",
}

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PATHS = {
    "raw_data":       os.path.join(PROJECT_ROOT, "data", "raw_data"),
    "dataset":        os.path.join(PROJECT_ROOT, "data", "gesture_dataset"),
    "checkpoints":    os.path.join(PROJECT_ROOT, "checkpoints"),
    "results":        os.path.join(PROJECT_ROOT, "results"),
    "training_curves": os.path.join(PROJECT_ROOT, "results", "training_curves"),
    "confusion_matrices": os.path.join(PROJECT_ROOT, "results", "confusion_matrices"),
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# ─────────────────────────────────────────────
# ViT Configuration
# ─────────────────────────────────────────────
VIT_CONFIG = {
    # "model_name":     "vit_base_patch16_224",   # timm model name
    # "model_name":     "vit_small_patch16_224",   # timm model name
    "model_name":     "vit_tiny_patch16_224",   # timm model name
    "input_size":     224,                      # 224×224 pixels
    "batch_size":     16,
    "learning_rate":  1e-4,
    "weight_decay":   1e-4,
    "epochs":         30,
    "dropout":        0.1,
    "optimizer":      "adamw",                  # "adamw"
    "scheduler":      "cosine",                 # "cosine"
    "label_smoothing": 0.1,
    "freeze_epochs":  5,                        # Phase 1: train head only
    "pretrained":     True,
}

# ─────────────────────────────────────────────
# Training Settings
# ─────────────────────────────────────────────
TRAINING = {
    "early_stopping_patience": 5,
    "num_workers":             2,              # DataLoader workers
    "pin_memory":              True,
    "train_split":             0.70,
    "val_split":               0.15,
    "test_split":              0.15,
}

# ─────────────────────────────────────────────
# Data Augmentation Parameters
# ─────────────────────────────────────────────
AUGMENTATION = {
    "horizontal_flip_prob":  0.5,
    "rotation_degrees":      15,
    "color_jitter_brightness": 0.3,
    "color_jitter_contrast":   0.3,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue":        0.1,
    "gaussian_blur_kernel":    3,
    "gaussian_blur_prob":      0.2,
    "random_crop_scale":       (0.85, 1.0),
}

# ─────────────────────────────────────────────
# Image Normalization (ImageNet stats)
# ─────────────────────────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# MediaPipe Settings
# ─────────────────────────────────────────────
MEDIAPIPE = {
    "max_num_hands":           1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence":  0.5,
    "roi_padding":             20,     # Extra pixels around hand bbox
}

# ─────────────────────────────────────────────
# Real-Time System Settings
# ─────────────────────────────────────────────
REALTIME = {
    "camera_id":           0,
    "camera_width":        640,
    "camera_height":       480,
    "smoothing_window":    5,         # Moving average window for cursor
    "debounce_frames":     2,         # Consecutive frames for action trigger
    "click_cooldown_ms":   200,       # Cooldown after click action
    "scroll_amount":       30,        # Scroll units per trigger
    "confidence_threshold": 0.6,      # Minimum prediction confidence
}

# ─────────────────────────────────────────────
# Webcam Data Collection Settings
# ─────────────────────────────────────────────
DATA_COLLECTION = {
    "images_per_class":    300,       # Target images per class
    "capture_delay_ms":    100,       # Delay between captures
    "camera_id":           0,
    "camera_width":        640,
    "camera_height":       480,
}
