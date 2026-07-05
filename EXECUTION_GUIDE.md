# Execution Guide — Real-Time Cursor Control Using Hand Gestures with Vision Transformer

> Follow these steps **in order** to get the full project running.

---

## Prerequisites

- **Python 3.9+** installed
- **Webcam** connected
- **NVIDIA GPU with CUDA** (recommended for ViT training, not mandatory)

---

## Download Pretrained Model (Skip Training)

If you want to skip training and use a pretrained model directly:

[Download best_vit_model.pth](https://huggingface.co/pratikm27/new_vit_train_model/blob/main/best_vit_model.pth)

📂 Place it here:
```
Final_Year_Project/
├── checkpoints/
│   └── best_vit_model.pth
└── ...
```

Then jump straight to **Step 4 (Evaluate)** or **Step 5 (Run Real-Time)**.

---

## Step 1: Install Dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm {timm.__version__}')"
python -c "import mediapipe; print(f'MediaPipe {mediapipe.__version__}')"
python -c "import pyautogui; print(f'PyAutoGUI {pyautogui.__version__}')"
```

All four should print version numbers without errors.

---

## Step 2: Collect Gesture Data

```bash
python data/collect_data.py
```

A webcam window will open with a control panel overlay.

### Controls

| Key     | Action                        |
|---------|-------------------------------|
| 0–6     | Select gesture class          |
| SPACE   | Toggle auto-capture ON/OFF    |
| S       | Save a single frame           |
| Q       | Quit and save                 |

### What To Do

1. Press a **number key (0–6)** to select a gesture class
2. Perform that gesture in front of the webcam
3. Press **SPACE** to start auto-capturing (images save automatically)
4. Hold the gesture steady for a few seconds
5. Press **SPACE** again to stop auto-capture
6. Switch to the next class and repeat
7. **Target: 200–500 images per class** (progress bars shown on screen)

### Gesture Reference

| Key | Gesture              | What To Do With Your Hand                     |
|-----|----------------------|-----------------------------------------------|
| 0   | Open Palm            | Spread all 5 fingers wide, palm facing camera |
| 1   | Index Point          | Only index finger extended, others curled      |
| 2   | Two Fingers Up       | Index + middle finger extended (peace sign)    |
| 3   | Fist                 | Close all fingers into a fist                 |
| 4   | Pinch                | Touch thumb tip and index fingertip together   |
| 5   | Three Fingers Up     | Index + middle + ring fingers extended         |
| 6   | Three Fingers Down   | Three fingers curled/pointing downward         |

### Gesture → Cursor Action Mapping

| # | Gesture | Cursor Action | Details |
|---|---------|---------------|---------|
| 0 | **Open Palm** ✋ | **Move cursor** | Cursor follows your index fingertip position |
| 1 | **Index Point** ☝️ | **Left click** | Triggers after 2 steady frames (debounce) |
| 2 | **Two Fingers Up** ✌️ | **Right click** | Same debounce as left click |
| 3 | **Fist** ✊ | **Neutral / Stop** | No action — pause cursor control |
| 4 | **Pinch** 🤏 | **Drag** | Holds left mouse button; move hand to drag |
| 5 | **Three Fingers Up** 🤟 | **Scroll up** | Scrolls up per trigger |
| 6 | **Three Fingers Down**🤙 | **Scroll down** | Scrolls down per trigger |

> **Note:** This mapping is defined in `config.py` under `ACTION_MAP` — you can customize it.

### Tips for Better Data

- Record in different lighting conditions (bright, dim, mixed)
- Vary your hand distance from the camera (close, medium, far)
- Slightly rotate your hand angle between captures
- Use both left and right hands if possible
- Keep background varied

### Output

Images are saved to: `data/raw_data/{gesture_class}/`

---

## Step 3: Prepare Dataset (Split into Train/Val/Test)

```bash
python data/prepare_dataset.py
```

This takes your raw images and creates a structured dataset:

```
data/gesture_dataset/
├── train/   (70% of images)
├── val/     (15% of images)
└── test/    (15% of images)
```

Each split contains subfolders for each gesture class.

If prompted about overwriting, type `y` and press Enter.

---

## Step 4: Train the ViT Model

```bash
python training/train.py --epochs 30
```

### What Happens

- Downloads `vit_tiny_patch16_224` pretrained on ImageNet (first run only)
- **Phase 1 (epochs 1–5):** Backbone frozen — only classification head is trained
- **Phase 2 (epochs 6–30):** Full model fine-tuned with lower learning rate (LR × 0.1)
- Applies data augmentation (random crop, flip, rotation, color jitter, Gaussian blur)
- Early stopping triggers if validation loss doesn't improve for 5 consecutive epochs
- Saves best checkpoint to `checkpoints/best_vit_model.pth`
- Saves training curves to `results/training_curves/vit_training_curves.png`
- Saves training metrics to `results/vit_metrics.json`

### Optional Overrides

```bash
python training/train.py --epochs 20 --batch-size 8 --lr 5e-5
```

### Expected Training Time

| Hardware | Approx. Time |
|----------|-------------|
| NVIDIA GPU (CUDA) | 15–30 minutes |
| CPU only | 1–2 hours |

### Expected Output

```
  Epoch [1/30] | LR: 0.000100
    Train Loss: 0.xxxx | Train Acc: xx.xx%
    Val   Loss: 0.xxxx | Val   Acc: xx.xx%
    ★ New best saved! (val_loss=0.xxxx, val_acc=xx.xx%)

  Phase 2: UNFREEZING backbone at epoch 6
  ...
  TRAINING COMPLETE
  Best Val Accuracy: xx.xx%
```

---

## Step 5: Evaluate the ViT Model

```bash
python training/evaluate.py
```

### What You Get

- **Test Accuracy, Precision, Recall, F1-Score** (overall and per-class)
- **Confusion Matrix** heatmap → `results/confusion_matrices/vit_confusion_matrix.png`
- **Inference Latency** and **FPS** benchmarks (100-run average)
- **Model Size** in MB and parameter count
- Full **classification report** printed to terminal
- All metrics saved to `results/vit_eval_results.json`

### Use a Custom Checkpoint

```bash
python training/evaluate.py --checkpoint path/to/your_checkpoint.pth
```

---

## Step 5.5: Real-Time Live Accuracy Evaluation

```bash
python training/realtime_evaluate.py
```

This tests your model's accuracy **live** — the system prompts you to perform each gesture in front of the webcam and measures how accurately the model predicts in real-time.

### How It Works

1. **Countdown (3s):** Screen shows "Get ready to show: OPEN PALM"
2. **Recording (5s):** You hold the gesture while the model records predictions every frame
3. **Transition (2s):** Brief rest before the next gesture
4. Repeats for all 7 gestures × 3 rounds = **21 trials**

### Controls

| Key | Action |
|-----|--------|
| SPACE | Skip current gesture early |
| Q | Abort entire session |

### Options

```bash
# More rounds (5 instead of default 3)
python training/realtime_evaluate.py --rounds 5

# Longer hold time (8 seconds instead of default 5)
python training/realtime_evaluate.py --hold 8

# Custom checkpoint
python training/realtime_evaluate.py --checkpoint path/to/model.pth
```

### Output

- **Comparison table:** Test-set accuracy vs real-time accuracy (side by side)
- **Confusion matrix:** `results/confusion_matrices/vit_realtime_confusion_matrix.png`
- **Full results:** `results/vit_realtime_eval_results.json`
- **Per-trial breakdown:** Accuracy for each gesture trial printed to terminal

---

## Step 6: Run Real-Time Gesture Control

```bash
python realtime/gesture_control.py
```

### Debug Mode (No Cursor Movement)

Use this first to verify predictions before enabling cursor control:

```bash
python realtime/gesture_control.py --no-cursor
```

### Silent Mode (No Visualization Window)

```bash
python realtime/gesture_control.py --no-debug
```

### What You'll See

- Live webcam feed with MediaPipe hand landmarks drawn
- Bounding box around the detected hand
- Current gesture prediction + confidence score
- FPS counter and latency display
- Your cursor moves/clicks based on gestures!

### Controls

| Input | Action |
|-------|--------|
| Q key | Quit the system |
| Mouse to top-left corner | Emergency stop (PyAutoGUI fail-safe) |

---

## Step 7: Generate Research Report

```bash
python analysis/generate_report.py
```

Generates a structured research-paper-style report at `results/research_report.md` covering:

- Abstract, Introduction, Methodology, Results, Discussion, Conclusion, References

> Run this **after** evaluation so the report includes real metrics.

---

## Quick Reference — All Commands

```bash
# Install
pip install -r requirements.txt

# Collect Data
python data/collect_data.py

# Prepare Dataset
python data/prepare_dataset.py

# Train ViT
python training/train.py --epochs 30

# Evaluate ViT
python training/evaluate.py

# Real-Time Live Evaluation
python training/realtime_evaluate.py
python training/realtime_evaluate.py --rounds 5 --hold 8   # Custom

# Run Real-Time
python realtime/gesture_control.py
python realtime/gesture_control.py --no-cursor   # Debug mode

# Generate Report
python analysis/generate_report.py
```


---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Webcam not opening | Check `REALTIME["camera_id"]` in `config.py` |
| Low accuracy | Collect more data (500+ per class) in varied conditions |
| ViT training too slow | Use GPU, or reduce epochs: `--epochs 15` |
| Cursor going erratic | Use `--no-cursor` first to verify predictions |
| CUDA out of memory | Reduce batch size: `--batch-size 8` |
| `timm` model download fails | Check internet connection (first run downloads pretrained weights) |

---

## Output Files Summary

| File / Folder | Contents |
|---------------|----------|
| `checkpoints/best_vit_model.pth` | Best trained ViT weights |
| `results/vit_metrics.json` | Training metrics (loss, accuracy, time) |
| `results/vit_eval_results.json` | Test evaluation results |
| `results/training_curves/vit_training_curves.png` | Loss & accuracy plots |
| `results/confusion_matrices/vit_confusion_matrix.png` | Confusion matrix heatmap |
| `results/vit_realtime_eval_results.json` | Real-time live evaluation results |
| `results/confusion_matrices/vit_realtime_confusion_matrix.png` | Real-time confusion matrix |
| `results/research_report.md` | Full research-style report |
