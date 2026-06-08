# XFSL Medical AI

### Explainable Few-Shot Learning for Skin Lesion Analysis

A deployable implementation of the XFSL (Explainable Few-Shot Learning) framework developed for medical image diagnosis.

This application provides:

* Skin lesion classification
* Prototype-based reasoning
* Similarity-gap uncertainty estimation
* Out-of-Distribution (OOD) detection
* Grad-CAM explainability
* Interactive web dashboard

The deployment version is derived from the research repository and optimized for demonstration and inference.

---

# Features

### Explainable AI

Visualize model reasoning using Grad-CAM heatmaps.

### Few-Shot Learning

Prototype-based classification using a small support set.

### OOD Detection

Reject unfamiliar samples using cosine similarity thresholding.

### Similarity-Gap Reasoning

Estimate prediction certainty through prototype similarity separation.

### Interactive Dashboard

Modern frontend interface for image upload, diagnosis, and visualization.

---

# System Architecture

```text
User Upload
      ↓
Frontend (HTML/CSS/JavaScript)
      ↓
FastAPI Backend
      ↓
XFSL Inference Engine
      ↓
Prototype Matching
      ↓
OOD Detection
      ↓
Grad-CAM Explainability
      ↓
Diagnosis + Visualization
```

---

# Project Structure

```text
deployment/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── api.py
├── engine.py
├── predictor.py
├── explainability.py
├── generate_prototypes.py
│
├── trained_model.pth
├── saved_k20_prototypes.npy
│
├── uploads/
├── outputs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Model Files

### trained_model.pth

Fine-tuned EfficientNet-B0 backbone.

Used for:

* Feature extraction
* Classification
* Grad-CAM generation

---

### saved_k20_prototypes.npy

Saved prototype vectors generated from the optimal K=20 experiment.

Used for:

* Few-shot classification
* Similarity scoring
* OOD detection

---

# Local Installation

Clone repository:

```bash
git clone https://github.com/ashishyadav18/XFSL-Medical-Image-Diagnosis.git
```

Navigate into deployment folder:

```bash
cd deployment
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Backend

Start FastAPI:

```bash
uvicorn api:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

Open:

```text
frontend/index.html
```

in a browser.

For production deployment, update:

```javascript
const API_URL = "YOUR_BACKEND_URL";
```

inside:

```text
frontend/app.js
```

---

# API Endpoint

### POST /predict

Upload a skin lesion image.

Returns:

```json
{
  "prediction": "MEL",
  "confidence": 0.78,
  "alternative": "NV",
  "delta": 0.12,
  "clarity": "Clear",
  "reason": "Strong confidence prediction",
  "heatmap": "/outputs/heatmap_xxx.jpg"
}
```

---

# Deployment

## Frontend

Recommended platform:

```text
Vercel
```

Deploy:

```text
deployment/frontend/
```

---

## Backend

Recommended platform:

```text
Render
```

Deploy:

```text
deployment/
```

using:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

---

# Screenshots

Add screenshots later inside:

```text
assets/
```

Suggested files:

```text
assets/
├── dashboard-home.png
├── dashboard-results.png
├── gradcam-example.png
└── architecture.png
```

---

# Troubleshooting

### Model Not Found

Verify:

```text
trained_model.pth
saved_k20_prototypes.npy
```

exist in the deployment directory.

---

### Heatmap Not Generated

Verify:

```text
outputs/
```

exists and is writable.

---

### API Connection Error

Ensure backend is running:

```bash
uvicorn api:app --reload
```

and frontend API URL is correctly configured.

---

# Research Background

This deployment package is based on the dissertation project:

**Explainable Prototype-Based Few-Shot Learning for Skin Lesion Classification**

located in the root repository.

The deployment version preserves the original XFSL methodology while providing a production-style demonstration interface.

---

# Author

Ashish Yadav

GitHub:
https://github.com/ashishyadav18

---

# Disclaimer

This software is intended for research, educational, and demonstration purposes only.

It should not be used as a substitute for professional medical diagnosis or clinical decision making.
