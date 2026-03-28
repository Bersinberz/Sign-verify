<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Space+Grotesk&weight=700&size=42&pause=1000&color=EE4C2C&center=true&vCenter=true&width=900&lines=Sign-Verify;Deep+Learning+Signature+Verification;Siamese+Neural+Networks;Detect+Forgeries+with+Precision" alt="Typing SVG" />

<p align="center">
  <strong>An end-to-end signature authentication system powered by Siamese Neural Networks — compare, score, and verify handwritten signatures in real time.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://flask.palletsprojects.com/"><img src="https://img.shields.io/badge/Flask-2.3-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" /></a>
  <a href="https://python-pillow.org/"><img src="https://img.shields.io/badge/Pillow-9.5%2B-000000?style=for-the-badge&logo=python&logoColor=white" alt="Pillow" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Siamese%20CNN-blueviolet?style=flat-square" />
</p>

</div>

---

## 📽️ Project Overview

**Sign-Verify** is an end-to-end machine learning project designed to authenticate handwritten signatures. By leveraging the power of **Siamese Neural Networks**, the system compares a query signature against a reference signature and calculates the similarity distance.

If the distance is below a learned threshold, the signatures are marked as a **MATCH** (Genuine); otherwise, they are flagged as **DO NOT MATCH** (Potential Forgery).

---

## ✨ Features

- 🧠 **Siamese Network Architecture** — Twin CNN branches extract feature vectors and compare them via Euclidean distance.
- ⚡ **Real-time Verification** — Upload any two signature images and get instant results via AJAX.
- 📊 **Similarity Scoring** — Provides a numerical confidence score alongside the verdict.
- 🖼️ **Optimized Preprocessing** — Automatic grayscale conversion and resizing to `155×220` for consistent inference.
- 🌐 **Responsive Web Interface** — Clean, minimal UI built with Flask and Vanilla CSS.
- 🔁 **GPU / CPU Agnostic** — Automatically detects and uses CUDA if available, falls back to CPU seamlessly.

---

## 🧠 Architecture

The core is a **Siamese Network** — two identical CNN branches sharing weights, designed to learn similarity rather than classification.

```
Signature 1 ──► [ CNN Branch ] ──► Feature Vector 1 ──┐
                                                        ├──► Euclidean Distance ──► Match / Forgery
Signature 2 ──► [ CNN Branch ] ──► Feature Vector 2 ──┘
```

- **CNN Backbone:** 3 convolutional layers (32 → 64 → 128 filters) with ReLU + MaxPool
- **FC Head:** `51200 → 512 → 128` embedding space
- **Decision Rule:** `distance < 1.0` → Genuine, else → Forgery

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Backend | Flask 2.3, Werkzeug, Gunicorn |
| Deep Learning | PyTorch 2.0+, TorchVision |
| Computer Vision | Pillow, torchvision.transforms |
| Scientific | NumPy, scikit-learn, Matplotlib |
| Frontend | HTML5, CSS3 (Vanilla) |
| Training | Jupyter Notebook (train.ipynb) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU (optional but recommended for training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sign-verify.git
   cd Sign-verify
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the trained model checkpoint exists at:
   ```
   checkpoints/siamese_best.pth
   ```
   > Train your own using `train.ipynb`, or drop in a pre-trained checkpoint.

### Running the App

```bash
python webapp.py
```

Navigate to `http://127.0.0.1:5000` and upload two signature images to verify.

---

## 📂 Project Structure

```text
Sign-verify/
├── checkpoints/              # Saved model weights (.pth)
├── signatures_dataset/
│   ├── full_org/             # 1320 genuine signature samples
│   └── full_forg/            # 1320 forged signature samples
├── templates/
│   └── index.html            # Flask UI template
├── uploads/                  # Temporary storage for uploaded images
├── webapp.py                 # Flask server & inference pipeline
├── train.ipynb               # Model training & experimentation
└── requirements.txt          # Python dependencies
```

---

## 📈 Future Roadmap

- [ ] Contrastive / Triplet Loss for improved embedding quality
- [ ] REST API with Swagger docs for third-party integrations
- [ ] Batch verification support
- [ ] Mobile-friendly UI overhaul
- [ ] PostgreSQL integration for audit logging

---

<div align="center">
  <p>Built with ❤️ for Secure Document Verification.</p>
</div>
