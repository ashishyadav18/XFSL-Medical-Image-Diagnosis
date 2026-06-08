import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ===================================
# CONFIG
# ===================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pth")
PROTOTYPE_PATH = os.path.join(BASE_DIR, "saved_k20_prototypes.npy")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===================================
# LOAD PROTOTYPES
# ===================================

PROTOTYPES = np.load(
    PROTOTYPE_PATH,
    allow_pickle=True
).item()

CLASS_NAMES = list(PROTOTYPES.keys())

# ===================================
# LOAD MODEL
# ===================================

MODEL = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

MODEL.classifier[1] = nn.Linear(
    MODEL.classifier[1].in_features,
    len(CLASS_NAMES)
)

MODEL.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

MODEL = MODEL.to(DEVICE)
MODEL.eval()

# ===================================
# TRANSFORM
# ===================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===================================
# GENERATE HEATMAP
# ===================================

def generate_heatmap(
    image_path,
    predicted_class
):

    img = Image.open(image_path).convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(
        model=MODEL,
        target_layers=[MODEL.features[-1]]
    )

    target = [
        ClassifierOutputTarget(
            CLASS_NAMES.index(predicted_class)
        )
    ]

    cam_map = gradcam(
        input_tensor=input_tensor,
        targets=target
    )[0]

    img_np = np.array(
        img.resize((224, 224))
    ) / 255.0

    heatmap = show_cam_on_image(
        img_np,
        cam_map,
        use_rgb=True
    )

    filename = os.path.basename(image_path)

    save_path = os.path.join(
        OUTPUT_DIR,
        f"heatmap_{filename}"
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    return save_path


# ===================================
# TEST
# ===================================

if __name__ == "__main__":

    image_path = input("Image Path: ")
    prediction = input("Predicted Class: ")

    path = generate_heatmap(
        image_path,
        prediction
    )

    print(path)