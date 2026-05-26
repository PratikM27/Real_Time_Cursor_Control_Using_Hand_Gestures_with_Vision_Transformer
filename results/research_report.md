# Real-Time Cursor Control Using Hand Gestures with Vision Transformer

*Date: May 25, 2026*

---

## Abstract

This project presents a real-time gesture-controlled cursor system using a
Vision Transformer (ViT) as the core classification model. Webcam input is
processed by MediaPipe for hand detection and ROI extraction. The ViT model
classifies the hand region into one of 7 gesture classes,
which are then mapped to cursor actions (move, click, scroll, drag) via
PyAutoGUI.

**Results:**
- Test Accuracy:  99.91% (if available)
- Inference FPS:  199.3 (if available)
- Avg Latency:    5.02 ms (if available)

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

**Gesture Classes:** 7 classes

| Class | Gesture | Cursor Action |
|-------|---------|--------------:|
| 0 | Open Palm | move_cursor |
| 1 | Index Point | left_click |
| 2 | Two Fingers Up | right_click |
| 3 | Fist | neutral |
| 4 | Pinch | drag |
| 5 | Three Fingers Up (Rock sign🤟) | scroll_up |
| 6 | Three Fingers Down (Call me sign🤙) | scroll_down |

- **Collection:** Custom images captured via webcam using MediaPipe hand ROI extraction
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Augmentation:** Random flip, rotation (±15°), color jitter, Gaussian blur

### 3.3 Vision Transformer Architecture

```
Input (3×224×224)
  → Patch Embedding (16×16 patches → 196 tokens)
  → [CLS] Token + Positional Encoding
  → Transformer Encoder × 12 layers (MHSA + MLP)
  → [CLS] Output → Classification Head → 7 classes
```

- **Base Model:** vit_tiny_patch16_224 (pretrained on ImageNet-21k)
- **Parameters:** ~86 Million
- **Training Phase 1:** Freeze backbone — train classification head only (5 epochs)
- **Training Phase 2:** Unfreeze backbone — full fine-tuning with 10× lower LR
- **Optimizer:** AdamW (LR=0.0001, weight_decay=0.0001)
- **Scheduler:** CosineAnnealingLR
- **Loss:** CrossEntropyLoss (label_smoothing=0.1)

---

## 4. Results

### 4.1 Classification Performance

| Metric            | ViT |
|-------------------|-----|
| Accuracy (%)      | 99.91 |
| Precision (%)     | 99.91 |
| Recall (%)        | 99.92 |
| F1-Score (%)      | 99.91 |

### 4.2 Speed & Efficiency

| Metric            | ViT |
|-------------------|-----|
| FPS               | 199.3 |
| Latency (ms)      | 5.02 |
| Model Size (MB)   | 21.08 |
| Parameters (M)    | 5.53 |

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
