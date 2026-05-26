# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

> Vision Transformer (ViT) Based Gesture Recognition for Human-Computer Interaction

---

## Overview

This project implements a **real-time gesture-controlled cursor system** using webcam input and a **Vision Transformer (ViT)** as the core classifier.

| Model | Architecture | Input Size | Parameters |
|-------|-------------|-----------|------------|
| **ViT** | ViT-Base-Patch16-224 (ImageNet pretrained) | 224×224 | ~86M |

### Gesture Classes (7)

| Gesture | Cursor Action |
|---------|--------------|
| Open Palm | Move cursor |
| Index Point | Left click |
| Two Fingers Up | Right click |
| Fist | Neutral / Stop |
| Pinch | Drag |
| Three Fingers Up | Scroll up |
| Three Fingers Down | Scroll down |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- Webcam
- (Optional) NVIDIA GPU with CUDA for faster training

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm {timm.__version__}')"
python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')"
```

### 4. Download Pretrained Model (Optional)

If you want to use a pretrained model instead of training your own:

```bash
https://huggingface.co/pratikm27/gesture-recognition-model/blob/main/best_vit_model.pth
```

Place it here:
```
project/
├── checkpoints/
│   └── best_vit_model.pth
└── ...
```

---

## Usage Guide

### Step 1: Collect Training Data

```bash
python data/collect_data.py
```

- Press keys **0–6** to select a gesture class
- Press **SPACE** to start/stop auto-capture
- Press **S** to save a single frame
- Press **Q** to quit
- Target: **200–500 images per class**

### Step 2: Prepare Dataset (Split into Train/Val/Test)

```bash
python data/prepare_dataset.py
```

This creates the `data/gesture_dataset/{train,val,test}/` directory structure.

### Step 3: Train ViT Model

```bash
python training/train.py --epochs 30
```

Training uses a two-phase strategy:
- **Phase 1 (epochs 1–5):** Train classification head only (backbone frozen)
- **Phase 2 (epochs 6–30):** Fine-tune entire model

Training curves and best checkpoint are saved automatically.

### Step 4: Evaluate Model

```bash
python training/evaluate.py
```

Generates confusion matrix, classification report, and latency benchmarks.

### Step 5: Run Real-Time Gesture Control

```bash
python realtime/gesture_control.py
```

Debug mode (no cursor movement, just see predictions):
```bash
python realtime/gesture_control.py --no-cursor
```

Press **Q** to quit the real-time system.

### Step 6: Generate Research Report

```bash
python analysis/generate_report.py
```

Generates a structured research-paper-style report at `results/research_report.md`.

---

## Project Structure

```
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── EXECUTION_GUIDE.md         # Detailed step-by-step guide
├── data/
│   ├── collect_data.py        # Webcam data collection
│   ├── prepare_dataset.py     # Train/val/test split
│   └── gesture_dataset/       # Generated dataset
├── models/
│   └── vit_model.py           # Vision Transformer architecture
├── training/
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation & metrics
│   └── utils.py               # Helpers
├── realtime/
│   ├── gesture_control.py     # Main real-time loop
│   ├── hand_detector.py       # MediaPipe wrapper
│   ├── cursor_controller.py   # PyAutoGUI wrapper
│   └── gesture_smoother.py    # Smoothing & debounce
├── analysis/
│   └── generate_report.py     # Report generation
├── checkpoints/               # Saved model weights
└── results/                   # Output plots & metrics
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, timm |
| Hand Detection | MediaPipe |
| Video Capture | OpenCV |
| Cursor Control | PyAutoGUI |
| Data Science | NumPy, Matplotlib, scikit-learn |

---

## License

This project is developed for academic purposes as a Final Year Project.
