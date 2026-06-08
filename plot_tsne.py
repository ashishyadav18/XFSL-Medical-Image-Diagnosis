import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "dataset/test" 
CLASSES = ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']
# Academic color palette
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# LOAD MODEL (Feature Extractor)
# =========================
def get_feature_extractor():
    print("Loading model...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
    
    # Remove classification layer to get raw embeddings
    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model

# =========================
# EXTRACT EMBEDDINGS
# =========================
def extract_all_features(model):
    print("Extracting features from the test set...")
    features = []
    labels = []

    with torch.no_grad():
        for cls in CLASSES:
            cls_dir = os.path.join(TEST_DIR, cls)
            if not os.path.isdir(cls_dir): continue
            
            for img_name in os.listdir(cls_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == 'cuda')):
                    feat = model(tensor).squeeze().cpu().numpy()
                
                # L2 Normalize (Crucial to match your cosine similarity logic)
                feat = feat / (np.linalg.norm(feat) + 1e-8)
                
                features.append(feat)
                labels.append(cls)

    return np.array(features), np.array(labels)

# =========================
# MAIN
# =========================
def main():
    model = get_feature_extractor()
    X, y = extract_all_features(model)

    print("Running t-SNE dimensionality reduction (this takes a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # Plotting
    print("Generating graph...")
    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")
    
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=y, hue_order=CLASSES, palette=COLORS,
        alpha=0.8, edgecolor="w", s=60
    )

    plt.title("t-SNE Visualization of the Learned Embedding Space", pad=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Lesion Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs("outputs/graphs", exist_ok=True)
    save_path = "outputs/graphs/10_tsne_visualization.png"
    plt.savefig(save_path)
    print(f"✅ Success! Graph saved to {save_path}")

if __name__ == "__main__":
    main()