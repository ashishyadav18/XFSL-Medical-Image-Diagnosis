import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# =========================
# CONFIG & SETUP
# =========================
DATASET_PATH = "dataset/test"
K_SHOT = 20
RUNS = 10
SEED = 42

# HARDWARE AUTO-DETECTION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs("logs", exist_ok=True)

# =========================
# HARDWARE OPTIMIZATIONS
# =========================
if IS_CUDA:
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(6)

# =========================
# SEED
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if IS_CUDA:
    torch.cuda.manual_seed_all(SEED)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

def extract_feature(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():
            feat = model(img).squeeze().cpu().numpy()
            
    # Standard L2 Normalization
    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# BASELINE LOGIC: MATCHING NETWORKS
# =========================
def evaluate_matching_networks(X_support, y_support, X_query, y_query):
    # Because embeddings are L2 normalized, dot product = cosine similarity
    sim_matrix = np.dot(X_query, X_support.T) 
    
    # Apply softmax to get attention weights (with numerical stability adjustment)
    exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
    attn_weights = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    
    y_pred = []
    for q_idx in range(len(X_query)):
        class_scores = defaultdict(float)
        
        # Accumulate attention weights for each class
        for s_idx in range(len(X_support)):
            class_scores[y_support[s_idx]] += attn_weights[q_idx, s_idx]
            
        # Select the class with the highest total weight
        best_class = max(class_scores, key=class_scores.get)
        y_pred.append(best_class)
        
    acc = accuracy_score(y_query, y_pred)
    f1 = f1_score(y_query, y_pred, average='macro')
    mcc = matthews_corrcoef(y_query, y_pred)
    
    return acc, f1, mcc

# =========================
# MAIN EVALUATION LOOP
# =========================
def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    data = load_dataset()
    
    model = get_model(len(data))
    
    print(f"\nRunning Matching Networks Baseline (K={K_SHOT}, Runs={RUNS})...")
    print("-" * 65)
    print(f"{'Run':<5} | {'Accuracy (%)':<15} | {'Macro F1':<12} | {'MCC':<12}")
    print("-" * 65)
    
    results_log = []
    accuracies, f1_scores, mcc_scores = [], [], []
    
    for i in range(RUNS):
        support, query = split_data(data, K_SHOT)
        
        # --- FEATURE CACHING OPTIMIZATION ---
        X_support, y_support = [], []
        for cls, imgs in support.items():
            for img in imgs:
                X_support.append(extract_feature(model, img))
                y_support.append(cls)
        X_support, y_support = np.array(X_support), np.array(y_support)

        X_query, y_query = [], []
        for cls, imgs in query.items():
            for img in imgs:
                X_query.append(extract_feature(model, img))
                y_query.append(cls)
        X_query, y_query = np.array(X_query), np.array(y_query)

        # --- EVALUATE BASELINE ---
        acc, f1, mcc = evaluate_matching_networks(X_support, y_support, X_query, y_query)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        
        # Log for CSV
        results_log.append({
            "Run": i + 1,
            "Accuracy": acc * 100,
            "Macro_F1": f1,
            "MCC": mcc
        })
        
        print(f"{i+1:<5} | {acc*100:<15.2f} | {f1:<12.4f} | {mcc:<12.4f}")

    # --- FINAL CALCULATIONS ---
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    mean_f1 = np.mean(f1_scores)
    mean_mcc = np.mean(mcc_scores)

    print("-" * 65)
    print(f"🏁 FINAL MATCHING NETWORKS RESULTS (K={K_SHOT})")
    print("=" * 65)
    print(f"Mean Accuracy : {mean_acc:.2f}% (Std Dev: {std_acc:.2f}%)")
    print(f"Mean Macro F1 : {mean_f1:.4f}")
    print(f"Mean MCC      : {mean_mcc:.4f}")
    print("=" * 65)
    
    # Save detailed logs
    csv_path = "logs/baseline_matching_network.csv"
    pd.DataFrame(results_log).to_csv(csv_path, index=False)
    print(f"\n💾 Successfully saved detailed run logs to: {csv_path}")

if __name__ == "__main__":
    main()