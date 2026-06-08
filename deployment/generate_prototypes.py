import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset/test"
K_SHOTS = [20]
RUNS = 2  # Updated to 10 runs for better statistical pooling
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
    torch.set_num_threads(6) # Speeds up inference on your AMD Ryzen CPU

# =========================
# SEED
# =========================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if IS_CUDA:
    torch.cuda.manual_seed_all(SEED)

# =========================
# TRANSFORMS (TTA INTEGRATED)
# =========================
transform_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# LOAD DATASET
# =========================
def load_dataset():
    data = {}
    for cls in os.listdir(DATASET_PATH):
        cls_path = os.path.join(DATASET_PATH, cls)
        if os.path.isdir(cls_path):
            images = [
                os.path.join(cls_path, img)
                for img in os.listdir(cls_path)
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if len(images) >= max(K_SHOTS) + 1:
                data[cls] = images
    return data

# =========================
# SPLIT SUPPORT / QUERY
# =========================
def split_data(data, k):
    support = {}
    query = {}

    for cls, images in data.items():
        images_copy = images.copy()
        random.shuffle(images_copy)

        support[cls] = images_copy[:k]
        query[cls] = images_copy[k:]

    return support, query

# =========================
# MODEL
# =========================
def get_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model

# =========================
# FEATURE EXTRACTION (TTA + Power Norm)
# =========================
def extract_feature(model, img_path):
    img = Image.open(img_path).convert("RGB")
    
    t_a = transform_1(img).unsqueeze(0).to(DEVICE)
    t_b = transform_2(img).unsqueeze(0).to(DEVICE)
    t_c = transform_3(img).unsqueeze(0).to(DEVICE)

    # AMP Autocast for faster inference if on GPU
    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():
            f_a = model(t_a).squeeze().cpu().numpy()
            f_b = model(t_b).squeeze().cpu().numpy()
            f_c = model(t_c).squeeze().cpu().numpy()

    # TTA Averaging
    feat = (f_a + f_b + f_c) / 3.0
    
    # Power Normalization (Tukey's Transform)
    feat = np.sign(feat) * np.power(np.abs(feat), 0.5)

    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# PROTOTYPES (0% Filter - Clean Mean)
# =========================
def create_prototypes(model, support):
    prototypes = {}

    for cls, images in support.items():
        features = [extract_feature(model, img) for img in images]
        features = np.array(features)

        # Pure mathematical center
        proto = np.mean(features, axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)

        prototypes[cls] = proto

    return prototypes

# =========================
# CLASSIFY
# =========================
def classify(prototypes, feature):
    scores = {}

    for cls, proto in prototypes.items():
        scores[cls] = np.dot(feature, proto)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_class, best_score = sorted_scores[0]
    second_class, second_score = sorted_scores[1]

    diff = best_score - second_score

    return best_class, best_score, diff

# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    data = load_dataset()
    print("Classes:", list(data.keys()))

    model = get_model(len(data))

    ablation_results = []
    per_class_results = []
    
    # Unified list for K=20 detailed plotting data
    k20_detailed_logs = []

    print("\nStarting N-Shot Ablation Study (Proposed Framework)...")
    print("-" * 105)
    print(f"{'Shots (K)':<10} | {'Mean Acc (%)':<15} | {'Mean F1':<10} | {'Mean MCC':<10} | {'Std Dev Acc':<15}")
    print("-" * 105)

    for k in K_SHOTS:
        accuracies = []
        f1_scores = []  
        mcc_scores = [] 
        
        class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for i in range(RUNS):
            support, query = split_data(data, k)
            prototypes = create_prototypes(model, support)

            y_true = []
            y_pred = []

            for cls, images in query.items():
                for img in images:
                    feature = extract_feature(model, img)
                    pred, score, diff = classify(prototypes, feature)

                    y_true.append(cls)
                    y_pred.append(pred)

                    # Clean, run-tagged logging for K=20
                    if k == 20:
                        k20_detailed_logs.append({
                            "Run": i + 1,
                            "True_Class": cls,
                            "Predicted_Class": pred,
                            "Confidence": score,
                            "Similarity_Gap": diff
                        })

                    class_stats[cls]["total"] += 1
                    if pred == cls:
                        class_stats[cls]["correct"] += 1

            # Calculate metrics for this specific run
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            mcc = matthews_corrcoef(y_true, y_pred)
            
            accuracies.append(acc)
            f1_scores.append(f1)
            mcc_scores.append(mcc)
            
            # Print full breakdown on the final run of K=20
            if k == 20 and i == RUNS - 1:
                print("\n" + "=" * 55)
                print(" 📊 CLASSIFICATION REPORT (Optimal Baseline: K=20)")
                print("=" * 55)
                print(classification_report(y_true, y_pred))
                print("=" * 55 + "\n")

                # --- SAVE THE PROTOTYPES ---
                np.save("logs/saved_k20_prototypes.npy", prototypes)
                print("💾 Successfully saved optimal K=20 prototypes for future use!")                

        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        mean_f1 = np.mean(f1_scores)  
        mean_mcc = np.mean(mcc_scores) 

        ablation_results.append({
            "K_Shots": k,
            "Mean_Accuracy": mean_acc,
            "Mean_F1": mean_f1,
            "Mean_MCC": mean_mcc,
            "Std_Deviation": std_acc
        })

        print(f"{k:<10} | {mean_acc:<15.2f} | {mean_f1:<10.4f} | {mean_mcc:<10.4f} | {std_acc:<15.2f}")

        for cls in class_stats:
            cls_acc = (class_stats[cls]["correct"] / class_stats[cls]["total"]) * 100

            per_class_results.append({
                "K_Shots": k,
                "Class": cls,
                "Accuracy": cls_acc
            })

    print("-" * 105)

    # Save to CSV
    pd.DataFrame(ablation_results).to_csv("logs/ablation_results.csv", index=False)
    pd.DataFrame(per_class_results).to_csv("logs/per_class_ablation.csv", index=False)
    pd.DataFrame(k20_detailed_logs).to_csv("logs/k20_detailed_logs.csv", index=False) 

    print("\nSaved:")
    print(" - ablation_results.csv")
    print(" - per_class_ablation.csv")
    print(" - k20_detailed_logs.csv (Master plotting file with Run tags!)")


if __name__ == "__main__":
    main()