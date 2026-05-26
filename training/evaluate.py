"""
evaluate.py — ViT Model Evaluation & Metrics
=============================================
Evaluates the trained ViT model on the test set and generates
comprehensive metrics, confusion matrix, and report.

Usage:
    python training/evaluate.py
    python training/evaluate.py --checkpoint checkpoints/best_vit_model.pth
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
    VIT_CONFIG, PATHS, TRAINING,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES, GESTURE_CLASSES,
    GESTURE_LABELS, SEED
)
from models.vit_model import build_vit_model
from training.utils import (
    set_seed, get_device, count_parameters, measure_model_size, load_checkpoint
)


def get_eval_transform(input_size):
    """Get evaluation transforms."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def evaluate_model(model, test_loader, device, class_names):
    """Run full evaluation on test set."""
    model.eval()
    all_preds      = []
    all_labels     = []
    all_probs      = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            start_time = time.perf_counter()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) / images.size(0))

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)
    avg_inference_time_ms = np.mean(inference_times) * 1000
    fps = 1000.0 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0

    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    results = {
        'accuracy':            accuracy,
        'precision_macro':     precision * 100,
        'recall_macro':        recall * 100,
        'f1_macro':            f1 * 100,
        'per_class':           {},
        'confusion_matrix':    cm.tolist(),
        'avg_inference_time_ms': avg_inference_time_ms,
        'fps':                 fps,
        'classification_report': report,
    }

    for i, class_name in enumerate(class_names):
        results['per_class'][class_name] = {
            'precision': precision_pc[i] * 100,
            'recall':    recall_pc[i] * 100,
            'f1':        f1_pc[i] * 100,
            'support':   int(support_pc[i]),
        }

    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    display_names = [GESTURE_LABELS.get(name, name) for name in class_names]

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=display_names,
        yticklabels=display_names,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title('Vision Transformer — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved to: {save_path}")


def measure_latency_detailed(model, input_size, device, num_runs=100):
    """Detailed latency measurement with warmup."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return {
        'mean_ms':   float(np.mean(times)),
        'std_ms':    float(np.std(times)),
        'min_ms':    float(np.min(times)),
        'max_ms':    float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps':       float(1000.0 / np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT gesture recognition model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    config     = VIT_CONFIG
    input_size = config["input_size"]

    print("=" * 60)
    print("  EVALUATION: Vision Transformer (ViT) Model")
    print("=" * 60)

    # Load test dataset
    test_dir = os.path.join(PATHS["dataset"], 'test')
    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        return

    test_dataset = datasets.ImageFolder(
        test_dir, transform=get_eval_transform(input_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=TRAINING["num_workers"],
        pin_memory=TRAINING["pin_memory"],
    )

    class_names = test_dataset.classes
    print(f"  Test set: {len(test_dataset)} images")
    print(f"  Classes:  {class_names}")
    print()

    # Build model
    print("  Loading ViT model...")
    model = build_vit_model(
        model_name=config["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout=config["dropout"],
    )

    # Load checkpoint
    checkpoint_path = args.checkpoint or os.path.join(
        PATHS["checkpoints"], "best_vit_model.pth"
    )
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train the model first with: python training/train.py")
        return

    model, checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    print(f"  Loaded: {checkpoint_path}")
    print(f"  Checkpoint epoch:   {checkpoint.get('epoch', 'N/A')}")
    print(f"  Checkpoint val_acc: {checkpoint.get('val_acc', 0):.2f}%")
    print()

    total_params, _ = count_parameters(model)
    model_size      = measure_model_size(model)

    # Evaluate
    print("  Running evaluation on test set...")
    results = evaluate_model(model, test_loader, device, class_names)

    print("  Measuring latency...")
    latency = measure_latency_detailed(model, input_size, device)
    results['latency']      = latency
    results['total_params'] = total_params
    results['model_size_mb']= model_size

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Test Accuracy:     {results['accuracy']:.2f}%")
    print(f"  Precision (macro): {results['precision_macro']:.2f}%")
    print(f"  Recall (macro):    {results['recall_macro']:.2f}%")
    print(f"  F1-Score (macro):  {results['f1_macro']:.2f}%")
    print(f"\n  Model Size:        {model_size:.2f} MB")
    print(f"  Parameters:        {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\n  Avg Latency:       {latency['mean_ms']:.2f} ms ± {latency['std_ms']:.2f} ms")
    print(f"  FPS:               {latency['fps']:.1f}")

    print(f"\n  Per-Class Performance:")
    print(f"  {'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("  " + "-" * 60)
    for class_name, metrics in results['per_class'].items():
        label = GESTURE_LABELS.get(class_name, class_name)
        print(f"  {label:<20s} {metrics['precision']:>9.2f}% {metrics['recall']:>9.2f}% "
              f"{metrics['f1']:>9.2f}% {metrics['support']:>9d}")

    # Save confusion matrix
    cm_path = os.path.join(PATHS["confusion_matrices"], "vit_confusion_matrix.png")
    plot_confusion_matrix(np.array(results['confusion_matrix']), class_names, cm_path)

    # Save results JSON
    save_path = os.path.join(PATHS["results"], "vit_eval_results.json")
    save_results = {k: v for k, v in results.items() if k != 'classification_report'}
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved to: {save_path}")

    print(f"\n  Full Classification Report:")
    print(results['classification_report'])


if __name__ == "__main__":
    main()
