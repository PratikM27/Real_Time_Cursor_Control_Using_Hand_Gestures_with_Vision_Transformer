"""
utils.py — Training Utilities
===============================
Early stopping, checkpointing, plotting, and helper functions.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")
    return device


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_model_size(model):
    """Measure model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        verbose: Print messages
    """
    
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"    EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("    Early stopping triggered!")
                return True
        
        return False


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, filepath, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


def plot_training_curves(history, save_dir, model_name):
    """
    Plot and save training/validation loss and accuracy curves.
    
    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_dir: Directory to save plots
        model_name: model identifier used in filename (e.g. 'vit')
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=3)
    ax1.set_title(f'{model_name.upper()} — Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best epoch marker
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                label=f'Best epoch ({best_epoch})')
    ax1.legend()
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=3)
    ax2.set_title(f'{model_name.upper()} — Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to: {save_path}")


def save_metrics(metrics, save_dir, model_name):
    """Save training metrics to JSON."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {filepath}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for measuring training time."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def elapsed_str(self):
        """Return elapsed time as formatted string."""
        minutes = int(self.elapsed // 60)
        seconds = int(self.elapsed % 60)
        return f"{minutes}m {seconds}s"
