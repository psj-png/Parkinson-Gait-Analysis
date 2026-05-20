"""
Step 2: Train a CNN (ResNet-18) for Normal vs Abnormal binary classification.

Input : output/optical_flow/{Normal,Abnormal}/
Output: output/models/cnn_best.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from pathlib import Path

_BASE      = Path(__file__).resolve().parent.parent
DATA_DIR   = _BASE / "output" / "optical_flow"
MODEL_DIR  = _BASE / "output" / "models"
MODEL_PATH = MODEL_DIR / "cnn_best.pth"

BATCH_SIZE = 16
EPOCHS     = 20
LR         = 1e-4
IMG_SIZE   = 224
VAL_RATIO  = 0.2
SEED       = 42


def get_dataloaders():
    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
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

    # 같은 인덱스 분할을 두 transform에 각각 적용
    full_train = datasets.ImageFolder(DATA_DIR, transform=train_tfm)
    full_val   = datasets.ImageFolder(DATA_DIR, transform=val_tfm)

    n       = len(full_train)
    n_val   = int(n * VAL_RATIO)
    n_train = n - n_val
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()

    train_set = Subset(full_train, indices[:n_train])
    val_set   = Subset(full_val,   indices[n_train:])

    print(f"Dataset split: train={n_train}, val={n_val} (total={n})")

    loaders = {
        "train": DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        "val":   DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False),
    }
    return loaders, full_train.classes


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    loaders, classes = get_dataloaders()
    print(f"Classes: {classes}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_acc = running_corrects / len(loaders[phase].dataset)
            print(f"Epoch {epoch+1}/{EPOCHS} [{phase}] Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  -> Best model saved (acc={best_acc:.4f})")

        scheduler.step()

    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    train()
