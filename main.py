# ===================== main.py =====================
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import Preprocess as pre
import models1 as models

torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)

IMAGE_SIZE = 224
DATA_DIR = Path("DATASET") / "images"
ARTIFACT_DIR = Path("artifacts")

EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5


# -----------------------------------------------------------
# TRAIN ONE EPOCH
# -----------------------------------------------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=None, use_amp=False):
    model.train()
    total_loss, total_correct, total_items = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_items += labels.size(0)

    return total_loss / len(train_loader), (total_correct / total_items) * 100


# -----------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100, loss_sum / len(loader)


# -----------------------------------------------------------
# SAVE CHECKPOINT
# -----------------------------------------------------------
def save_checkpoint(model_state, classes, image_size, model_name):
    model_dir = ARTIFACT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model_state,
            "arch": model_name,
            "image_size": image_size,
        },
        model_dir / "best_model.pt",
    )

    with open(model_dir / "classes.json", "w") as f:
        json.dump({"classes": classes}, f, indent=2)


# -----------------------------------------------------------
# PLOT INDIVIDUAL LOSS CURVES
# -----------------------------------------------------------
def plot_loss_curves(train_losses, val_losses, model_name):
    model_dir = ARTIFACT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves – {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(model_dir / "loss_curves.png")
    plt.close()


# -----------------------------------------------------------
# PLOT COMBINED LOSS CURVES (ALL MODELS)
# -----------------------------------------------------------
def plot_combined_loss_curves(all_val_losses):
    epochs = range(1, len(next(iter(all_val_losses.values()))) + 1)

    plt.figure(figsize=(8, 6))
    for model_name, val_losses in all_val_losses.items():
        plt.plot(epochs, val_losses, marker="o", label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison (ANN vs CNN vs ResNet‑18)")
    plt.legend()
    plt.grid(True)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    plt.savefig(ARTIFACT_DIR / "combined_loss_curves.png")
    plt.show()


# -----------------------------------------------------------
# PRECISION / RECALL / F1
# -----------------------------------------------------------
def compute_precision_recall_f1(model, val_loader, classes, device, save_path):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, target_names=classes, digits=4
    )

    print("\n📊 Precision / Recall / F1‑Score:\n")
    print(report)

    with open(save_path, "w") as f:
        f.write(report)


# -----------------------------------------------------------
# TRAIN SINGLE MODEL
# -----------------------------------------------------------
def train_model(model_name, ModelClass, num_classes,
                train_loader, val_loader, device, classes):

    print("\n============================================")
    print(f"🔵 Training Model → {model_name}")
    print("============================================")

    model = ModelClass(num_classes=num_classes).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )

    use_amp = device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    best_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler=scaler, use_amp=use_amp
        )
        val_acc, val_loss = evaluate(
            model, val_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"[{model_name}] Epoch {epoch}/{EPOCHS} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model.state_dict(), classes, IMAGE_SIZE, model_name
            )

    plot_loss_curves(train_losses, val_losses, model_name)
    return best_acc, val_losses


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device → {device}")

    train_loader, val_loader, classes = pre.load_data(
        DATA_DIR,
        batch_size=32,
        augment=True,
        num_workers=min(8, os.cpu_count()),
    )

    num_classes = len(classes)

    MODELS = {
        "ANNClassifier": models.ANNClassifier,
        "CNNRegularized": models.CNNRegularized,
        "ResNetTransfer": models.ResNetTransfer,
    }

    results = {}
    all_val_losses = {}

    for model_name, ModelClass in MODELS.items():
        acc, val_losses = train_model(
            model_name,
            ModelClass,
            num_classes,
            train_loader,
            val_loader,
            device,
            classes,
        )
        results[model_name] = acc
        all_val_losses[model_name] = val_losses

    plot_combined_loss_curves(all_val_losses)

    best_model = max(results, key=results.get)
    print("\n🏆 BEST MODEL:", best_model)

    if best_model == "ResNetTransfer":
        model = models.ResNetTransfer(num_classes=num_classes).to(device)
        checkpoint = torch.load(
            ARTIFACT_DIR / "ResNetTransfer" / "best_model.pt",
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state"])

        compute_precision_recall_f1(
            model,
            val_loader,
            classes,
            device,
            ARTIFACT_DIR / "ResNetTransfer" / "classification_report.txt",
        )


if __name__ == "__main__":
    main()
