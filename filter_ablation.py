import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# =========================
# CONFIG & SETUP
# =========================
DATASET_PATH = "dataset/test"
K_SHOT = 20
RUNS = 10
SEED = 42

# --- THE MASTER TOGGLE ---
# Set to False first, then change to True for the second run
USE_TTA = True

# Added 0.0 (0% filter) to test the pure baseline
FILTER_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs("logs", exist_ok=True)

if IS_CUDA:
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(6)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if IS_CUDA:
    torch.cuda.manual_seed_all(SEED)

# =========================
# TRANSFORMS
# =========================
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_hflip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_vflip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# HELPER FUNCTIONS
# =========================
def load_dataset():
    data = {}
    for cls in os.listdir(DATASET_PATH):
        cls_path = os.path.join(DATASET_PATH, cls)
        if os.path.isdir(cls_path):
            images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) >= K_SHOT + 1:
                data[cls] = images
    return data

def split_data(data, k):
    support, query = {}, {}
    for cls, images in data.items():
        images_copy = images.copy()
        random.shuffle(images_copy)
        support[cls] = images_copy[:k]
        query[cls] = images_copy[k:]
    return support, query

def get_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model

# Dynamic Feature Extractor (Handles both Standard and TTA)
def extract_feature(model, img_path, use_tta):
    img = Image.open(img_path).convert("RGB")
    
    if use_tta:
        t_base = transform_base(img).unsqueeze(0).to(DEVICE)
        t_hf = transform_hflip(img).unsqueeze(0).to(DEVICE)
        t_vf = transform_vflip(img).unsqueeze(0).to(DEVICE)

        with torch.amp.autocast("cuda", enabled=IS_CUDA):
            with torch.no_grad():
                f_base = model(t_base).squeeze().cpu().numpy()
                f_hf = model(t_hf).squeeze().cpu().numpy()
                f_vf = model(t_vf).squeeze().cpu().numpy()
                
        feat = (f_base + f_hf + f_vf) / 3.0
    else:
        t_base = transform_base(img).unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=IS_CUDA):
            with torch.no_grad():
                feat = model(t_base).squeeze().cpu().numpy()
                
    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# EVALUATION (DYNAMIC FILTERING)
# =========================
def evaluate_with_filter(X_support, y_support, X_query, y_query, filter_ratio):
    classes = np.unique(y_support)
    prototypes = {}
    
    for cls in classes:
        features = X_support[y_support == cls]
        center = np.mean(features, axis=0)
        distances = np.linalg.norm(features - center, axis=1)
        
        # Calculate exactly how many to keep
        keep = int(len(features) * (1.0 - filter_ratio))
        idx = np.argsort(distances)[:keep]
        
        proto = np.mean(features[idx], axis=0)
        prototypes[cls] = proto / (np.linalg.norm(proto) + 1e-8)
        
    y_pred = []
    for x_q in X_query:
        scores = {cls: np.dot(x_q, proto) for cls, proto in prototypes.items()}
        y_pred.append(max(scores, key=scores.get))
        
    return accuracy_score(y_query, y_pred), f1_score(y_query, y_pred, average='macro'), matthews_corrcoef(y_query, y_pred)

# =========================
# MAIN EVALUATION LOOP
# =========================
def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    data = load_dataset()
    model = get_model(len(data))
    
    tta_status = "ON" if USE_TTA else "OFF"
    print(f"\nStarting Filter Ablation Study (K={K_SHOT}, Runs={RUNS}, TTA={tta_status})...")
    
    results = {f_ratio: {"acc": [], "f1": [], "mcc": []} for f_ratio in FILTER_RATIOS}
    
    for i in range(RUNS):
        print(f"Processing Run {i+1}/{RUNS} (Extracting CNN Features)...", end="\r")
        support, query = split_data(data, K_SHOT)
        
        X_support, y_support = [], []
        for cls, imgs in support.items():
            for img in imgs:
                X_support.append(extract_feature(model, img, USE_TTA))
                y_support.append(cls)
        X_support, y_support = np.array(X_support), np.array(y_support)

        X_query, y_query = [], []
        for cls, imgs in query.items():
            for img in imgs:
                X_query.append(extract_feature(model, img, USE_TTA))
                y_query.append(cls)
        X_query, y_query = np.array(X_query), np.array(y_query)

        for f_ratio in FILTER_RATIOS:
            acc, f1, mcc = evaluate_with_filter(X_support, y_support, X_query, y_query, f_ratio)
            results[f_ratio]["acc"].append(acc)
            results[f_ratio]["f1"].append(f1)
            results[f_ratio]["mcc"].append(mcc)
            
    print(f"Processing Run {RUNS}/{RUNS} (Complete!)" + " " * 20)

    print("\n" + "=" * 65)
    print(f" 📊 FILTERING ABLATION RESULTS (TTA {tta_status})")
    print("=" * 65)
    print(f"{'Drop %':<8} | {'Kept Imgs':<10} | {'Mean Acc (%)':<15} | {'Mean F1':<10} | {'Mean MCC':<10}")
    print("-" * 65)
    
    csv_data = []
    for f_ratio in FILTER_RATIOS:
        mean_acc = np.mean(results[f_ratio]["acc"]) * 100
        mean_f1 = np.mean(results[f_ratio]["f1"])
        mean_mcc = np.mean(results[f_ratio]["mcc"])
        images_kept = int(K_SHOT * (1.0 - f_ratio))
        
        drop_pct = f"{int(f_ratio * 100)}%"
        print(f"{drop_pct:<8} | {images_kept:<10} | {mean_acc:<15.2f} | {mean_f1:<10.4f} | {mean_mcc:<10.4f}")
        
        csv_data.append({
            "Drop_Percentage": drop_pct,
            "Images_Kept": images_kept,
            "Mean_Accuracy": mean_acc,
            "Mean_F1": mean_f1,
            "Mean_MCC": mean_mcc
        })
    print("=" * 65)
    
    csv_filename = f"logs/filter_ablation_tta_{tta_status.lower()}.csv"
    pd.DataFrame(csv_data).to_csv(csv_filename, index=False)
    print(f"\n💾 Successfully saved results to: {csv_filename}")

if __name__ == "__main__":
    main()