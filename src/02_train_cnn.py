"""
Step 2: Train a CNN (ResNet-18) for Normal vs Abnormal binary classification.

Input : output/optical_flow/{Normal,Abnormal}/
Output: output/models/cnn_best.pth

Split: video-level 80/20  — 같은 영상의 프레임이 train/val에 섞이지 않도록 분리
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

_BASE      = Path(__file__).resolve().parent.parent
DATA_DIR   = _BASE / "output" / "optical_flow_crop"
MODEL_DIR  = _BASE / "output" / "models"
PLOT_DIR   = _BASE / "output" / "plots"
MODEL_PATH = MODEL_DIR / "cnn_best.pth"

BATCH_SIZE = 16
EPOCHS     = 20
LR         = 1e-4
IMG_SIZE   = 224
VAL_RATIO  = 0.2
SEED       = 42


class FrameDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples   = samples   # list of (Path, int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def video_level_split():
    """
    영상(폴더) 단위로 train/val 분리.
    구조: DATA_DIR/<class>/<video_name>/<frame>.png
    """
    classes     = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rng = random.Random(SEED)
    train_samples, val_samples = [], []
    summary = []

    for cls in classes:
        label     = class_to_idx[cls]
        video_dirs = sorted((DATA_DIR / cls).iterdir())
        video_dirs = [v for v in video_dirs if v.is_dir()]

        shuffled = video_dirs[:]
        rng.shuffle(shuffled)

        n_val   = max(1, int(len(shuffled) * VAL_RATIO))
        val_dirs   = shuffled[:n_val]
        train_dirs = shuffled[n_val:]

        for vdir in train_dirs:
            for img in sorted(vdir.glob("*.png")):
                train_samples.append((img, label))
        for vdir in val_dirs:
            for img in sorted(vdir.glob("*.png")):
                val_samples.append((img, label))

        summary.append(
            f"  [{cls}] total={len(video_dirs)} videos  "
            f"train={len(train_dirs)}({len(train_dirs)*30} frames)  "
            f"val={len(val_dirs)}({len(val_dirs)*30} frames)"
        )

    print("=== Video-level split ===")
    for line in summary:
        print(line)
    print(f"  Total: train={len(train_samples)} frames, val={len(val_samples)} frames")
    print()
    return train_samples, val_samples, classes


def get_dataloaders():
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_samples, val_samples, classes = video_level_split()

    loaders = {
        "train": DataLoader(FrameDataset(train_samples, train_tfm),
                            batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
        "val":   DataLoader(FrameDataset(val_samples,   val_tfm),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }
    return loaders, classes


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Epochs : {EPOCHS},  LR : {LR},  Batch : {BATCH_SIZE}")
    print()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    loaders, classes = get_dataloaders()
    print(f"Classes: {classes}  (idx 0={classes[0]}, 1={classes[1]})\n")

    model     = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(EPOCHS):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            all_preds, all_labels = [], []

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            n        = len(loaders[phase].dataset)
            acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / n
            f1       = f1_score(all_labels, all_preds, average="weighted")
            loss_avg = running_loss / n
            history[f"{phase}_loss"].append(round(loss_avg, 6))
            history[f"{phase}_acc"].append(round(acc, 6))
            print(f"Epoch {epoch+1:02d}/{EPOCHS} [{phase:5s}] "
                  f"Loss: {loss_avg:.4f}  Acc: {acc:.4f}  F1: {f1:.4f}")

            if phase == "val" and acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"           -> Best model saved (acc={best_acc:.4f})")

        scheduler.step()
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        with open(PLOT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print()

    # 최종 val 리포트
    print("=" * 55)
    print(f"Training complete.  Best val acc: {best_acc:.4f}")
    print("=" * 55)
    print("\n[Final Val Classification Report]")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loaders["val"]:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    print(classification_report(all_labels, all_preds, target_names=classes))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(all_labels, all_preds, classes, PLOT_DIR / "confusion_matrix_val.png")
    print(f"\nConfusion matrix saved: {PLOT_DIR / 'confusion_matrix_val.png'}")

    _save_training_curve(history, PLOT_DIR / "training_curve.png")
    print(f"Training curve saved:  {PLOT_DIR / 'training_curve.png'}")


def _save_training_curve(history: dict, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], "o-", label="Train")
    ax_loss.plot(epochs, history["val_loss"],   "s--", label="Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], "o-", label="Train")
    ax_acc.plot(epochs, history["val_acc"],   "s--", label="Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.suptitle("ResNet-18 Training Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Validation Set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train()
