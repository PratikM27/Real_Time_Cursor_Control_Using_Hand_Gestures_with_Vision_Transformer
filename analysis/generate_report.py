"""
generate_report.py — Research Report Generator (ViT)
=====================================================
Generates a structured markdown report for the Vision Transformer
gesture recognition project.

Usage:
    python analysis/generate_report.py
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, GESTURE_CLASSES, GESTURE_LABELS, VIT_CONFIG


def load_vit_results():
    """Load ViT evaluation and training results."""
    results = {}

    eval_path    = os.path.join(PATHS["results"], "vit_eval_results.json")
    metrics_path = os.path.join(PATHS["results"], "vit_metrics.json")

    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            results['eval'] = json.load(f)
    else:
        print(f"  WARNING: Eval results not found: {eval_path}")
        results['eval'] = {}

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['training'] = json.load(f)
    else:
        print(f"  WARNING: Training metrics not found: {metrics_path}")
        results['training'] = {}

    return results


def generate_report():
    """Generate a comprehensive research-style report for ViT."""
    results   = load_vit_results()
    vit_eval  = results.get('eval', {})
    vit_train = results.get('training', {})

    accuracy    = vit_eval.get('accuracy', 'TBD')
    fps         = vit_eval.get('latency', {}).get('fps', 'TBD')
    latency_ms  = vit_eval.get('latency', {}).get('mean_ms', 'TBD')
    model_size  = vit_eval.get('model_size_mb', 'TBD')
    total_params= vit_eval.get('total_params', 0)

    report = f"""# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

*Date: {datetime.now().strftime('%B %d, %Y')}*

---

## Abstract

This project presents a real-time gesture-controlled cursor system using a
Vision Transformer (ViT) as the core classification model. Webcam input is
processed by MediaPipe for hand detection and ROI extraction. The ViT model
classifies the hand region into one of {len(GESTURE_CLASSES)} gesture classes,
which are then mapped to cursor actions (move, click, scroll, drag) via
PyAutoGUI.

**Results:**
- Test Accuracy:  {accuracy:.2f}% (if available)
- Inference FPS:  {fps:.1f} (if available)
- Avg Latency:    {latency_ms:.2f} ms (if available)

---

## 1. Introduction

### 1.1 Background
Traditional human-computer interaction relies on physical input devices such as
mice and keyboards. These create barriers for users with motor disabilities and
limit natural gesture-based interaction.

### 1.2 Problem Statement
The system must:
- Detect hand gestures in real time via webcam
- Classify gestures accurately using a Vision Transformer
- Translate gestures into cursor actions with low latency (<100 ms)

### 1.3 Objectives
1. Build a real-time hand gesture recognition system using webcam input
2. Implement a Vision Transformer (ViT) classifier with ImageNet pretraining
3. Evaluate performance using Accuracy, F1-score, FPS, and Latency
4. Deploy a working gesture-controlled cursor system

### 1.4 Contributions
- A modular, end-to-end ViT-based gesture control pipeline
- Two-phase fine-tuning strategy (head-only → full backbone)
- Real-time cursor control via PyAutoGUI

---

## 2. Literature Review

### 2.1 Vision Transformers
Vision Transformers (ViTs) adapt the transformer architecture from NLP to
computer vision by dividing images into fixed-size patches, embedding them as
tokens, and processing them with multi-head self-attention (MHSA). ViTs capture
global spatial context from the first layer, which is advantageous for
distinguishing similar hand gestures.

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021.

### 2.2 Hand Gesture Recognition
MediaPipe Hands (Google) provides real-time hand landmark detection and bounding
box extraction using a lightweight ML pipeline. Combined with a ViT classifier,
it enables accurate gesture recognition without specialized hardware.

Reference: Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," 2020.

---

## 3. Methodology

### 3.1 System Architecture

```
Webcam → Frame Capture (OpenCV) → Hand Detection (MediaPipe)
    → ROI Extraction → Preprocessing (224×224, ImageNet norm)
    → Vision Transformer Inference
    → Gesture Classification → Action Mapping → Cursor Control (PyAutoGUI)
```

### 3.2 Dataset

**Gesture Classes:** {len(GESTURE_CLASSES)} classes

| Class | Gesture | Cursor Action |
|-------|---------|--------------:|
"""

    for class_id, class_name in GESTURE_CLASSES.items():
        label = GESTURE_LABELS.get(class_name, class_name)
        from config import ACTION_MAP
        action = ACTION_MAP.get(class_name, "—")
        report += f"| {class_id} | {label} | {action} |\n"

    acc = vit_eval.get('accuracy')
    acc_str = f"{acc:.2f}" if isinstance(acc, float) else "TBD"
    
    prec = vit_eval.get('precision_macro')
    prec_str = f"{prec:.2f}" if isinstance(prec, float) else "TBD"
    
    rec = vit_eval.get('recall_macro')
    rec_str = f"{rec:.2f}" if isinstance(rec, float) else "TBD"
    
    f1 = vit_eval.get('f1_macro')
    f1_str = f"{f1:.2f}" if isinstance(f1, float) else "TBD"
    
    lat = vit_eval.get('latency', {})
    fps = lat.get('fps')
    fps_str = f"{fps:.1f}" if isinstance(fps, float) else "TBD"
    
    mean_ms = lat.get('mean_ms')
    mean_ms_str = f"{mean_ms:.2f}" if isinstance(mean_ms, float) else "TBD"
    
    mod_size = vit_eval.get('model_size_mb')
    mod_size_str = f"{mod_size:.2f}" if isinstance(mod_size, float) else "TBD"
    
    params_str = f"{total_params/1e6:.2f}" if total_params else "TBD"

    report += f"""
- **Collection:** Custom images captured via webcam using MediaPipe hand ROI extraction
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Augmentation:** Random flip, rotation (±15°), color jitter, Gaussian blur

### 3.3 Vision Transformer Architecture

```
Input (3×{VIT_CONFIG['input_size']}×{VIT_CONFIG['input_size']})
  → Patch Embedding (16×16 patches → 196 tokens)
  → [CLS] Token + Positional Encoding
  → Transformer Encoder × 12 layers (MHSA + MLP)
  → [CLS] Output → Classification Head → {len(GESTURE_CLASSES)} classes
```

- **Base Model:** {VIT_CONFIG['model_name']} (pretrained on ImageNet-21k)
- **Parameters:** ~86 Million
- **Training Phase 1:** Freeze backbone — train classification head only ({VIT_CONFIG['freeze_epochs']} epochs)
- **Training Phase 2:** Unfreeze backbone — full fine-tuning with 10× lower LR
- **Optimizer:** AdamW (LR={VIT_CONFIG['learning_rate']}, weight_decay={VIT_CONFIG['weight_decay']})
- **Scheduler:** CosineAnnealingLR
- **Loss:** CrossEntropyLoss (label_smoothing={VIT_CONFIG['label_smoothing']})

---

## 4. Results

### 4.1 Classification Performance

| Metric            | ViT |
|-------------------|-----|
| Accuracy (%)      | {acc_str} |
| Precision (%)     | {prec_str} |
| Recall (%)        | {rec_str} |
| F1-Score (%)      | {f1_str} |

### 4.2 Speed & Efficiency

| Metric            | ViT |
|-------------------|-----|
| FPS               | {fps_str} |
| Latency (ms)      | {mean_ms_str} |
| Model Size (MB)   | {mod_size_str} |
| Parameters (M)    | {params_str} |

---

## 5. Discussion

### 5.1 ViT Strengths for Gesture Recognition
- **Global Attention:** Self-attention captures relationships between all finger positions simultaneously
- **Pretraining:** ImageNet-21k pretraining provides rich visual representations
- **Interpretability:** Attention maps can highlight which image regions drive predictions
- **Scalability:** Performance scales with data and model size

### 5.2 Real-Time Feasibility
For cursor control, the system requires ≥15 FPS. GPU inference achieves this
comfortably. For CPU-only deployment, ONNX Runtime or INT8 quantization is
recommended (see optimization/ scripts).

---

## 6. Conclusion

This project demonstrates a functional real-time gesture-controlled cursor system
using a Vision Transformer. The two-phase fine-tuning strategy efficiently adapts
the ImageNet-pretrained ViT to the gesture recognition task. The modular pipeline
allows easy extension to additional gestures or cursor actions.

---

## 7. Future Work

1. **Lightweight ViT variants** (DeiT, MobileViT) for faster CPU inference
2. **Landmark-based classification** using MediaPipe coordinates directly
3. **Dynamic gesture recognition** using temporal sequences (Video ViT)
4. **Model distillation** to compress ViT into a smaller student model
5. **Edge deployment** on mobile devices

---

## References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.
2. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," 2020.
3. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
4. Wightman, "PyTorch Image Models (timm)," GitHub 2019.
5. He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
"""

    save_path = os.path.join(PATHS["results"], "research_report.md")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  Research report saved to: {save_path}")
    return report


def main():
    print("=" * 60)
    print("  GENERATING ViT RESEARCH REPORT")
    print("=" * 60)
    generate_report()
    print("\n  Done!")


if __name__ == "__main__":
    main()
