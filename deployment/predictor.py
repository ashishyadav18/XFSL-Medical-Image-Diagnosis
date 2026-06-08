import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# =========================
# CONFIG
# =========================

THRESHOLD = 0.55

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pth")
PROTOTYPE_PATH = os.path.join(BASE_DIR, "saved_k20_prototypes.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

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
# LOAD PROTOTYPES
# =========================

if not os.path.exists(PROTOTYPE_PATH):
    raise FileNotFoundError(
        f"Prototype file not found: {PROTOTYPE_PATH}"
    )

PROTOTYPES = np.load(
    PROTOTYPE_PATH,
    allow_pickle=True
).item()

CLASS_NAMES = list(PROTOTYPES.keys())

# =========================
# LOAD FEATURE EXTRACTOR
# =========================

MODEL = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

MODEL.classifier[1] = nn.Linear(
    MODEL.classifier[1].in_features,
    len(CLASS_NAMES)
)

MODEL.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

MODEL.classifier = nn.Identity()

MODEL = MODEL.to(DEVICE)
MODEL.eval()

# =========================
# FEATURE EXTRACTION
# =========================

def extract_feature(img_path):

    img = Image.open(img_path).convert("RGB")

    t1 = transform_1(img).unsqueeze(0).to(DEVICE)
    t2 = transform_2(img).unsqueeze(0).to(DEVICE)
    t3 = transform_3(img).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=IS_CUDA):
        with torch.no_grad():

            f1 = MODEL(t1).squeeze().cpu().numpy()
            f2 = MODEL(t2).squeeze().cpu().numpy()
            f3 = MODEL(t3).squeeze().cpu().numpy()

    feature = (f1 + f2 + f3) / 3.0

    feature = np.sign(feature) * np.power(
        np.abs(feature),
        0.5
    )

    feature = feature / (
        np.linalg.norm(feature) + 1e-8
    )

    return feature

# =========================
# CLASSIFICATION
# =========================

def classify(feature):

    scores = {
        cls: np.dot(feature, proto)
        for cls, proto in PROTOTYPES.items()
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best, best_score = sorted_scores[0]
    second, second_score = sorted_scores[1]

    delta = best_score - second_score

    if best_score < THRESHOLD:

        return {
            "prediction": "UNKNOWN",
            "confidence": round(float(best_score) * 100, 2),
            "alternative": second,
            "delta": float(delta),
            "clarity": "Uncertain",
            "reason": "Low similarity (OOD)",
            "all_scores": {
                k: round(float(v) * 100, 2)
                for k, v in scores.items()
            }
        }

    if delta > 0.10:
        clarity = "Clear"
        reason = (
            f"Strong confidence: "
            f"{best} distinct from {second} "
            f"(Δ={delta:.2f})"
        )

    elif delta > 0.05:
        clarity = "Moderate"
        reason = (
            f"Moderate confidence: "
            f"{best} slightly stronger than {second} "
            f"(Δ={delta:.2f})"
        )

    else:
        clarity = "Uncertain"
        reason = (
            f"Ambiguous: "
            f"{best} close to {second} "
            f"(Δ={delta:.2f})"
        )

    return {
        "prediction": best,
        "confidence": round(float(best_score) * 100, 2),
        "alternative": second,
        "delta": float(delta),
        "clarity": clarity,
        "reason": reason,
        "all_scores": {
            k: round(float(v) * 100, 2)
            for k, v in scores.items()
        }
    }

# =========================
# PUBLIC API
# =========================

def predict(image_path):

    feature = extract_feature(image_path)

    return classify(feature)


# =========================
# LOCAL TEST
# =========================

if __name__ == "__main__":

    image_path = input("Image Path: ")

    result = predict(image_path)

    print("\nPrediction Result\n")
    print(result)