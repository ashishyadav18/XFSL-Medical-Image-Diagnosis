import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import pandas as pd   # ✅ ADDED

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = "dataset/test"
K_SHOT = 20
RUNS = 10
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs("logs", exist_ok=True)   # ✅ ADDED (same as baselines)

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
# PREPROCESSING PIPELINE
# =========================
transform_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_3 = transforms.Compose([
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

# =========================
# FEATURE EXTRACTION
# =========================
def extract_embeddings(model, img_path):
    img = Image.open(img_path).convert("RGB")
    
    t_a = transform_1(img).unsqueeze(0).to(DEVICE)
    t_b = transform_2(img).unsqueeze(0).to(DEVICE)
    t_c = transform_3(img).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():
            f_a = model(t_a).squeeze().cpu().numpy()
            f_b = model(t_b).squeeze().cpu().numpy()
            f_c = model(t_c).squeeze().cpu().numpy()
            
    feat = (f_a + f_b + f_c) / 3.0
    feat = np.sign(feat) * np.power(np.abs(feat), 0.5)
    
    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# PROPOSED FRAMEWORK
# =========================
def evaluate_framework(X_support, y_support, X_query, y_query):
    classes = np.unique(y_support)
    prototypes = {}
    
    for cls in classes:
        features = X_support[y_support == cls]
        proto = np.mean(features, axis=0)
        prototypes[cls] = proto / (np.linalg.norm(proto) + 1e-8)
        
    y_pred = []
    for x_q in X_query:
        scores = {cls: np.dot(x_q, proto) for cls, proto in prototypes.items()}
        y_pred.append(max(scores, key=scores.get))
        
    acc = accuracy_score(y_query, y_pred)
    f1 = f1_score(y_query, y_pred, average='macro')
    mcc = matthews_corrcoef(y_query, y_pred)
    
    return acc, f1, mcc

# =========================
# MAIN EVALUATION
# =========================
def main():
    print(f"Loading dataset...")
    data = load_dataset()
    model = get_model(len(data))
    
    print(f"\nEvaluating Proposed Framework (K={K_SHOT})")
    print("-" * 60)
    
    accuracies, f1_scores, mcc_scores = [], [], []
    results_log = []   # ✅ ADDED

    for i in range(RUNS):
        support, query = split_data(data, K_SHOT)
        
        X_support, y_support = [], []
        for cls, imgs in support.items():
            for img in imgs:
                X_support.append(extract_embeddings(model, img))
                y_support.append(cls)
        X_support, y_support = np.array(X_support), np.array(y_support)

        X_query, y_query = [], []
        for cls, imgs in query.items():
            for img in imgs:
                X_query.append(extract_embeddings(model, img))
                y_query.append(cls)
        X_query, y_query = np.array(X_query), np.array(y_query)

        acc, f1, mcc = evaluate_framework(X_support, y_support, X_query, y_query)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        mcc_scores.append(mcc)

        # ✅ LOG EACH RUN
        results_log.append({
            "Run": i + 1,
            "Accuracy": acc * 100,
            "Macro_F1": f1,
            "MCC": mcc
        })
        
        print(f"Run {i+1:<2} | Acc: {acc*100:.2f}% | F1: {f1:.4f} | MCC: {mcc:.4f}")

    print("=" * 60)
    print(f"MEAN ACCURACY : {np.mean(accuracies)*100:.2f}%")
    print(f"MEAN F1 SCORE : {np.mean(f1_scores):.4f}")
    print(f"MEAN MCC      : {np.mean(mcc_scores):.4f}")
    print("=" * 60)

    # ✅ SAVE CSV (FINAL ADDITION)
    csv_path = "logs/proposed_xfsl.csv"
    pd.DataFrame(results_log).to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

if __name__ == "__main__":
    main()