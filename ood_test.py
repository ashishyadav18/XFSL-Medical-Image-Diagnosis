import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# =========================
# CONFIG
# =========================
OOD_FAR_PATH = "dataset/ood/far_ood"
OOD_NEAR_PATH = "dataset/ood/near_ood"

THRESHOLDS = [0.50, 0.55, 0.60]
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs("logs", exist_ok=True)

# =========================
# HARDWARE OPTIMIZATION
# =========================
if IS_CUDA:
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(6)

# =========================
# TRANSFORMS (TTA)
# =========================
transform_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# MODEL
# =========================
def get_model():
    model = models.efficientnet_b0(weights=None)

    # NOTE: number of classes irrelevant since we remove classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    if not os.path.exists("trained_model.pth"):
        raise FileNotFoundError("trained_model.pth not found")

    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))

    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()

    return model

# =========================
# FEATURE EXTRACTION (FIXED)
# =========================
def extract_feature(model, img_path):
    img = Image.open(img_path).convert("RGB")

    t_a = transform_1(img).unsqueeze(0).to(DEVICE)
    t_b = transform_2(img).unsqueeze(0).to(DEVICE)
    t_c = transform_3(img).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():
            f_a = model(t_a).squeeze().cpu().numpy()
            f_b = model(t_b).squeeze().cpu().numpy()
            f_c = model(t_c).squeeze().cpu().numpy()

    # TTA averaging
    feat = (f_a + f_b + f_c) / 3.0

    # Power normalization (Tukey)
    feat = np.sign(feat) * np.power(np.abs(feat), 0.5)

    # L2 normalization
    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# LOAD PROTOTYPES (FIXED)
# =========================
def load_prototypes():
    path = "logs/saved_k20_prototypes.npy"

    if not os.path.exists(path):
        raise FileNotFoundError("Run main.py first to generate prototypes")

    prototypes = np.load(path, allow_pickle=True).item()
    print("✅ Loaded saved K=20 prototypes\n")

    return prototypes

# =========================
# TEST FOLDER
# =========================
def test_folder(model, prototypes, folder_path, name):
    if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
        print(f"{name} skipped (empty or missing)\n")
        return [], 0

    print(f"--- Testing {name} ---")

    images = [
        os.path.join(folder_path, img)
        for img in os.listdir(folder_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    scores_list = []

    for img_path in images:
        feat = extract_feature(model, img_path)

        scores = {
            cls: np.dot(feat, proto)
            for cls, proto in prototypes.items()
        }

        max_score = max(scores.values())
        scores_list.append(max_score)

        print(f"{os.path.basename(img_path)[:15]:<15} | {max_score:.3f}")

    print()
    return scores_list, len(images)

# =========================
# MAIN
# =========================
def main():
    model = get_model()
    prototypes = load_prototypes()

    print("Evaluating OOD Detection...\n")

    far_scores, far_total = test_folder(
        model, prototypes, OOD_FAR_PATH, "Far-OOD (Random Images)"
    )

    near_scores, near_total = test_folder(
        model, prototypes, OOD_NEAR_PATH, "Near-OOD (Unseen Lesions)"
    )

    results = []

    print("=" * 65)
    print(f"{'Threshold':<10} | {'Far-OOD (%)':<15} | {'Near-OOD (%)':<15}")
    print("=" * 65)

    for t in THRESHOLDS:
        far_caught = sum(1 for s in far_scores if s < t)
        near_caught = sum(1 for s in near_scores if s < t)

        far_rate = (far_caught / far_total) * 100 if far_total > 0 else 0
        near_rate = (near_caught / near_total) * 100 if near_total > 0 else 0

        print(f"{t:<10} | {far_rate:<15.2f} | {near_rate:<15.2f}")

        results.append({
            "Threshold": t,
            "Far_OOD_Detection (%)": far_rate,
            "Near_OOD_Detection (%)": near_rate
        })

    print("=" * 65)

    # Save results
    pd.DataFrame(results).to_csv("logs/ood_results.csv", index=False)

    print("\nSaved results to logs/ood_results.csv")


if __name__ == "__main__":
    main()