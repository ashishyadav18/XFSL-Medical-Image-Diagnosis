# Explainable Prototype-Based Few-Shot Learning for Skin Lesion Classification

This project presents an Explainable Few-Shot Learning (XFSL) framework for medical image diagnosis using prototype-based classification, similarity-based uncertainty estimation, and Grad-CAM explainability.

The framework is designed for low-data medical imaging scenarios and integrates:

- EfficientNet-B0 feature extraction
- Prototype-based Few-Shot Learning
- Similarity-Gap (∆) uncertainty metric
- Out-of-Distribution (OOD) detection
- Grad-CAM visual explanations
- Test-Time Augmentation (TTA)
- Power Normalization

---

# Features

- Lightweight XFSL framework
- Interpretable prototype-based predictions
- Similarity-based uncertainty reasoning
- OOD rejection using cosine similarity thresholding
- Grad-CAM heatmap generation
- N-shot ablation experiments
- Baseline comparison with multiple FSL methods

---

# Project Structure

```bash
XFSL-Medical-Image-Diagnosis/
│
├── train_model.py
├── main.py
├── ood_test.py
├── predict_and_explain.py
├── filter_ablation.py
│
├── Baseline_FSLs/
├── article/
├── dataset/
├── logs/
├── outputs/
├── Run_CMD/
│
├── plot_graphs.py
├── plot_radar.py
├── plot_tsne.py
│
├── trained_model.pth
├── requirements.txt
├── README.md
└── .gitignore





Dataset

This project uses the HAM10000 / ISIC 2019 skin lesion dataset.

Dataset folders are NOT included due to size limitations.

Expected structure:

dataset/
│
├── train/
├── test/
└── ood/





Installation

Create virtual environment:

python -m venv venv


Activate environment (Windows):

venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt





Training

Run CNN training:

python train_model.py
Few-Shot Evaluation

Run XFSL evaluation:

python main.py
OOD Detection

Run OOD evaluation:

python ood_test.py
Explainability

Run prediction + Grad-CAM visualization:

python predict_and_explain.py
Paper

The complete research article is available inside the article/ folder.




Author

Ashish Yadav
