# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

> Vision Transformer (ViT) Based Gesture Recognition for Human-Computer Interaction

---

## Overview

This project implements a **real-time gesture-controlled cursor system** using webcam input and a **Vision Transformer (ViT)** as the core classifier.

| Model | Architecture | Input Size | Parameters |
|-------|-------------|-----------|------------|
| **ViT** | ViT-Tiny-Patch16-224 (ImageNet pretrained) | 224×224 | ~5.7M |

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
python -c "import pyautogui; print(f'PyAutoGUI {pyautogui.__version__}')"
```

---

> **📌 Important:** The training dataset (gesture images) is **not included** in this repository — only the source code and a pretrained model are provided. You have two options to get started:
>
> | Path | Description | Steps |
> |------|-------------|-------|
> | **Path A** — Use Pretrained Model | Download the pretrained model and run the system directly | Install → Download Model → Run |
> | **Path B** — Train From Scratch | Collect your own gesture data, train a new model, then run | Install → Collect Data → Prepare → Train → Run |

---

## Path A: Use Pretrained Model (Quick Start)

If you want to skip data collection and training, download the pretrained model:

[Download best_vit_model.pth](https://huggingface.co/pratikm27/new_vit_train_model/blob/main/best_vit_model.pth)

Place it here:
```
Final_Year_Project/
├── checkpoints/
│   └── best_vit_model.pth
└── ...
```

Then jump directly to **Step 4 (Evaluate)**, **Step 5 (Real-Time Evaluation)**, or **Step 6 (Run Gesture Control)** below.

---

## Path B: Collect Your Own Data & Train From Scratch

Follow Steps 1–3 below to collect gesture data, prepare the dataset, and train a new model.

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

Optional overrides:
```bash
python training/train.py --epochs 20 --batch-size 8 --lr 5e-5
```

Training curves and best checkpoint are saved automatically.

---

## Common Steps (Both Paths)

### Step 4: Evaluate Model

```bash
python training/evaluate.py
```

Generates confusion matrix, classification report, and latency benchmarks.

Use a custom checkpoint:
```bash
python training/evaluate.py --checkpoint path/to/your_checkpoint.pth
```

### Step 5: Real-Time Live Accuracy Evaluation

```bash
python training/realtime_evaluate.py
```

Tests your model's accuracy **live** — the system prompts you to perform each gesture in front of the webcam and measures how accurately the model predicts in real-time.

Options:
```bash
python training/realtime_evaluate.py --rounds 5 --hold 8
python training/realtime_evaluate.py --checkpoint path/to/model.pth
```

### Step 6: Run Real-Time Gesture Control

```bash
python realtime/gesture_control.py
```

Debug mode (no cursor movement, just see predictions):
```bash
python realtime/gesture_control.py --no-cursor
```

Silent mode (no visualization window):
```bash
python realtime/gesture_control.py --no-debug
```

Press **Q** to quit the real-time system.

### Step 7: Generate Research Report

```bash
python analysis/generate_report.py
```

Generates a structured research-paper-style report at `results/research_report.md`.


---

## Project Structure

```
├── config.py                    # Central configuration
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── EXECUTION_GUIDE.md           # Detailed step-by-step guide
├── data/
│   ├── collect_data.py          # Webcam data collection
│   ├── prepare_dataset.py       # Train/val/test split
│   ├── raw_data/                # Raw collected images
│   └── gesture_dataset/         # Generated dataset (train/val/test)
├── models/
│   ├── __init__.py
│   └── vit_model.py             # Vision Transformer architecture
├── training/
│   ├── __init__.py
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation & metrics
│   ├── realtime_evaluate.py     # Live real-time accuracy evaluation
│   └── utils.py                 # Helpers
├── realtime/
│   ├── __init__.py
│   ├── gesture_control.py       # Main real-time loop
│   ├── hand_detector.py         # MediaPipe wrapper
│   ├── cursor_controller.py     # PyAutoGUI wrapper
│   └── gesture_smoother.py      # Smoothing & debounce
├── analysis/
│   ├── __init__.py
│   └── generate_report.py       # Report generation
├── checkpoints/                 # Saved model weights
└── results/                     # Output plots & metrics
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch, timm |
| Hand Detection | MediaPipe |
| Video Capture | OpenCV |
| Cursor Control | PyAutoGUI |
| Data Science | NumPy, Matplotlib, seaborn, scikit-learn, pandas |
| Utilities | tqdm, Pillow, ONNX, ONNX Runtime |

---

## License

This project is developed for academic purposes as a Final Year Project.
