import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # ✅ NEW
import cv2

# =========================
# CONFIG
# =========================
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "outputs/heatmaps"
PROTOTYPE_PATH = "logs/saved_k20_prototypes.npy"
THRESHOLD = 0.55

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# TRANSFORMS (TTA)
# =========================
transform_1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# LOAD MODELS
# =========================
def load_models(num_classes):

    model_full = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model_full.classifier[1] = nn.Linear(model_full.classifier[1].in_features, num_classes)
    model_full.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model_full = model_full.to(DEVICE)
    model_full.eval()

    model_feat = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model_feat.classifier[1] = nn.Linear(model_feat.classifier[1].in_features, num_classes)
    model_feat.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    model_feat.classifier = nn.Identity()
    model_feat = model_feat.to(DEVICE)
    model_feat.eval()

    return model_full, model_feat

# =========================
# LOAD PROTOTYPES
# =========================
def load_prototypes():
    if not os.path.exists(PROTOTYPE_PATH):
        raise FileNotFoundError("Run main.py first to generate prototypes")

    return np.load(PROTOTYPE_PATH, allow_pickle=True).item()

# =========================
# FEATURE EXTRACTION
# =========================
def extract_feature(model, img_path):
    img = Image.open(img_path).convert("RGB")

    t1 = transform_1(img).unsqueeze(0).to(DEVICE)
    t2 = transform_2(img).unsqueeze(0).to(DEVICE)
    t3 = transform_3(img).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():
            f1 = model(t1).squeeze().cpu().numpy()
            f2 = model(t2).squeeze().cpu().numpy()
            f3 = model(t3).squeeze().cpu().numpy()

    feat = (f1 + f2 + f3) / 3.0
    feat = np.sign(feat) * np.power(np.abs(feat), 0.5)

    return feat / (np.linalg.norm(feat) + 1e-8)

# =========================
# CLASSIFICATION
# =========================
def classify(prototypes, feature):
    scores = {cls: np.dot(feature, proto) for cls, proto in prototypes.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best, best_score = sorted_scores[0]
    second, second_score = sorted_scores[1]

    diff = best_score - second_score

    if best_score < THRESHOLD:
        return "UNKNOWN", best_score, second, diff, "Uncertain", "Low similarity (OOD)"

    if diff > 0.10:
        clarity = "Clear"
        reason = f"Strong confidence: {best} distinct from {second} (Δ={diff:.2f})"
    elif diff > 0.05:
        clarity = "Moderate"
        reason = f"Moderate confidence: {best} slightly stronger than {second} (Δ={diff:.2f})"
    else:
        clarity = "Uncertain"
        reason = f"Ambiguous: {best} close to {second} (Δ={diff:.2f})"

    return best, best_score, second, diff, clarity, reason

# =========================
# MAIN
# =========================
def main():

    prototypes = load_prototypes()
    class_names = list(prototypes.keys())

    model_full, model_feat = load_models(len(class_names))
    gradcam = GradCAM(model=model_full, target_layers=[model_full.features[-1]])

    results = []

    for img_name in os.listdir(INPUT_FOLDER):

        if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
            continue

        img_path = os.path.join(INPUT_FOLDER, img_name)

        feature = extract_feature(model_feat, img_path)
        pred, conf, second, diff, clarity, reason = classify(prototypes, feature)

        # =========================
        # Grad-CAM FIXED
        # =========================
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform_1(img).unsqueeze(0).to(DEVICE)

        if pred != "UNKNOWN":
            target = [ClassifierOutputTarget(class_names.index(pred))]
            cam_map = gradcam(input_tensor=input_tensor, targets=target)[0]
        else:
            cam_map = None

        img_np = np.array(img.resize((224,224))) / 255.0

        if cam_map is not None:
            heatmap = show_cam_on_image(img_np, cam_map, use_rgb=True)
            save_path = os.path.join(OUTPUT_FOLDER, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        else:
            print("⚠️ Grad-CAM skipped (UNKNOWN prediction)")

        # =========================
        # PRINT
        # =========================
        print("\n" + "="*50)
        print(f"📷 Image: {img_name}")
        print(f"🔍 Prediction: {pred}")
        print(f"📊 Confidence: {conf:.2f}")
        print(f"⚖️ Alternative: {second}")
        print(f"📉 Difference: {diff:.2f}")
        print(f"📈 Clarity: {clarity}")
        print(f"🧠 Reason: {reason}")
        print("="*50)

        results.append({
            "image": img_name,
            "prediction": pred,
            "confidence": conf,
            "similarity_diff": diff,
            "clarity": clarity,
            "reason": reason
        })

    pd.DataFrame(results).to_csv("outputs/results.csv", index=False)
    print("\nResults saved to outputs/results.csv")


if __name__ == "__main__":
    main()