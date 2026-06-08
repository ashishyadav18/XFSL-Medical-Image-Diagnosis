import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pandas as pd
import random
import numpy as np
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score  # NEW: MCC and AUC added

# =========================
# CONFIG
# =========================
SEED = 42
DATASET_PATH = "dataset/train"
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4

# HARDWARE AUTO-DETECTION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs("logs", exist_ok=True)

# =========================
# HARDWARE OPTIMIZATIONS
# =========================
if IS_CUDA:
    # Boosts speed for NVIDIA GPUs with fixed image sizes
    torch.backends.cudnn.benchmark = True
else:
    # Optimizes for AMD Ryzen / Intel CPUs (matching your 6 physical cores)
    torch.set_num_threads(6)

# =========================
# MAIN FUNCTION
# =========================
def main():

    # =========================
    # SEED
    # =========================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if IS_CUDA:
        torch.cuda.manual_seed_all(SEED)

    # =========================
    # TRANSFORMS
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # =========================
    # DATASET & SPLIT
    # =========================
    full_dataset = datasets.ImageFolder(DATASET_PATH)
    num_classes = len(full_dataset.classes)

    print("Classes:", full_dataset.classes)

    pd.DataFrame({"class": full_dataset.classes}).to_csv("logs/class_mapping.csv", index=False)

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transform)

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # =========================
    # WEIGHTED SAMPLER
    # =========================
    train_targets = [full_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
    sample_weights = [class_weights[t] for t in train_targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # DYNAMIC DATALOADER PARAMS
    dl_workers = 4 # Safe for Windows 11 with Ryzen 5
    dl_pin_memory = True if IS_CUDA else False # Only useful for GPU

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler, 
        num_workers=dl_workers,
        pin_memory=dl_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=dl_workers,
        pin_memory=dl_pin_memory
    )

    # =========================
    # MODEL
    # =========================
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    # =========================
    # FOCAL LOSS & OPTIMIZER
    # =========================
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return loss.mean()

    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # AMP Scaler (Auto-disables if running on CPU)
    scaler = torch.amp.GradScaler("cuda", enabled=IS_CUDA)

    # =========================
    # TRAIN LOOP
    # =========================
    best_val_f1 = 0.0  
    history = []

    for epoch in range(EPOCHS):
        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # SPEED OPTIMIZATION: set_to_none=True clears memory faster
            optimizer.zero_grad(set_to_none=True)

            # AUTO-SWITCH AMP: Speeds up NVIDIA GPUs, falls back to normal on AMD/CPU
            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)

        # ===== VALIDATION =====
        model.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0
        
        all_val_preds = []
        all_val_labels = []
        all_val_probs = [] # Needed for AUC

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # AMP for inference speedup if GPU is present
                with torch.amp.autocast("cuda", enabled=IS_CUDA):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1) # Get probabilities for AUC
                preds = torch.argmax(outputs, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        
        # Calculate Advanced Metrics
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        
        # AUC needs a try/except block just in case a validation batch somehow misses a class
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs, multi_class='ovr')
        except ValueError:
            val_auc = 0.0 

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | F1 {val_f1:.4f} | MCC {val_mcc:.4f} | AUC {val_auc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_mcc": val_mcc,
            "val_auc": val_auc,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "trained_model.pth")
            print(f"--> Saved best model (Macro F1: {best_val_f1:.4f})")
            
        scheduler.step()

    pd.DataFrame(history).to_csv("logs/training_log.csv", index=False)

    print(f"\nBest Validation Macro F1: {best_val_f1:.4f}")
    print("Model saved as trained_model.pth")

if __name__ == "__main__":
    main()