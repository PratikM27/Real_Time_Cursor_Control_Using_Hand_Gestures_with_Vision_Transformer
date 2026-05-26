"""
train.py — Vision Transformer Training Script
================================================
Trains a Vision Transformer (ViT) model on the gesture dataset.

Usage:
    python training/train.py
    python training/train.py --epochs 30
"""

import os
import sys
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIT_CONFIG, TRAINING, AUGMENTATION, PATHS,
    NORMALIZE_MEAN, NORMALIZE_STD, NUM_CLASSES, SEED
)
from models.vit_model import build_vit_model
from training.utils import (
    set_seed, get_device, count_parameters, measure_model_size,
    EarlyStopping, save_checkpoint, plot_training_curves,
    save_metrics, AverageMeter, Timer
)


def get_transforms(input_size, is_training=True):
    """Get data transforms for training or evaluation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size + 20, input_size + 20)),
            transforms.RandomResizedCrop(
                input_size,
                scale=AUGMENTATION["random_crop_scale"]
            ),
            transforms.RandomHorizontalFlip(AUGMENTATION["horizontal_flip_prob"]),
            transforms.RandomRotation(AUGMENTATION["rotation_degrees"]),
            transforms.ColorJitter(
                brightness=AUGMENTATION["color_jitter_brightness"],
                contrast=AUGMENTATION["color_jitter_contrast"],
                saturation=AUGMENTATION["color_jitter_saturation"],
                hue=AUGMENTATION["color_jitter_hue"],
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(AUGMENTATION["gaussian_blur_kernel"])
            ], p=AUGMENTATION["gaussian_blur_prob"]),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])


def create_data_loaders(dataset_dir, config):
    """Create train, validation, and test data loaders."""
    input_size = config["input_size"]
    batch_size = config["batch_size"]

    train_transform = get_transforms(input_size, is_training=True)
    eval_transform  = get_transforms(input_size, is_training=False)

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir   = os.path.join(dataset_dir, 'val')
    test_dir  = os.path.join(dataset_dir, 'test')

    for d, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"{name} directory not found: {d}\n"
                "Run 'python data/prepare_dataset.py' first."
            )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=TRAINING["num_workers"], pin_memory=TRAINING["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=TRAINING["num_workers"], pin_memory=TRAINING["pin_memory"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=TRAINING["num_workers"], pin_memory=TRAINING["pin_memory"],
    )

    print(f"  Train:   {len(train_dataset):5d} images ({len(train_loader)} batches)")
    print(f"  Val:     {len(val_dataset):5d} images ({len(val_loader)} batches)")
    print(f"  Test:    {len(test_dataset):5d} images ({len(test_loader)} batches)")
    print(f"  Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total   = 0

    pbar = tqdm(train_loader, desc="    Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc':  f'{100.*correct/total:.1f}%'
        })

    return loss_meter.avg, 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return loss_meter.avg, 100. * correct / total


def train_model(args):
    """Main training function for ViT."""
    set_seed(SEED)
    device = get_device()

    config = VIT_CONFIG.copy()
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr

    print("=" * 60)
    print("  TRAINING: Vision Transformer (ViT) — Gesture Recognition")
    print("=" * 60)
    print(f"  Config: {json.dumps(config, indent=4, default=str)}")
    print()

    # Data
    print("  Loading dataset...")
    train_loader, val_loader, test_loader, classes = create_data_loaders(
        PATHS["dataset"], config
    )
    print()

    # Model
    print("  Building ViT model...")
    model = build_vit_model(
        model_name=config["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=config["pretrained"],
        dropout=config["dropout"],
    )
    model.freeze_backbone()
    print("  Phase 1: Backbone FROZEN (head-only training)")

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    model_size = measure_model_size(model)
    print(f"  Total Parameters:     {total_params:>12,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:>12,} ({trainable_params/1e6:.2f}M)")
    print(f"  Model Size:           {model_size:.2f} MB")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    early_stopping = EarlyStopping(
        patience=TRAINING["early_stopping_patience"], verbose=True
    )

    checkpoint_path = os.path.join(PATHS["checkpoints"], "best_vit_model.pth")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'lr': [],
    }

    timer = Timer()
    timer.start()
    best_val_loss = float('inf')
    best_val_acc  = 0.0

    print("  Starting training...")
    print("-" * 60)

    for epoch in range(1, config["epochs"] + 1):
        # Phase 2: unfreeze backbone after freeze_epochs
        if epoch == config.get("freeze_epochs", 5) + 1:
            print(f"\n  Phase 2: UNFREEZING backbone at epoch {epoch}")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"] * 0.1,
                weight_decay=config["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"] - epoch + 1
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch [{epoch}/{config['epochs']}] | LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
            print(f"    ★ New best saved! (val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")

        if early_stopping(val_loss):
            print(f"\n  Early stopping at epoch {epoch}")
            break

    training_time = timer.stop()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total Training Time: {timer.elapsed_str()}")
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Best Val Accuracy:   {best_val_acc:.2f}%")
    print(f"  Checkpoint:          {checkpoint_path}")

    plot_training_curves(history, PATHS["training_curves"], "vit")

    metrics = {
        'model_type':            'vit',
        'config':                config,
        'best_val_loss':         best_val_loss,
        'best_val_acc':          best_val_acc,
        'training_time_seconds': training_time,
        'total_params':          total_params,
        'model_size_mb':         model_size,
        'epochs_trained':        len(history['train_loss']),
        'history':               history,
    }
    save_metrics(metrics, PATHS["results"], "vit")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train ViT gesture recognition model")
    parser.add_argument('--epochs',     type=int,   default=None, help='Training epochs')
    parser.add_argument('--batch-size', type=int,   default=None, help='Batch size')
    parser.add_argument('--lr',         type=float, default=None, help='Learning rate')
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
