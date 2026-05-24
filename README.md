# Deep Learning Architecture Showdown — EuroSAT

**Course:** Deep Learning — AI & Big Data Program | UIR 2025-2026  
**Prof:** Hakim Hafidi  

## 📌 Objective
Compare 4 deep learning architectures on satellite image classification (EuroSAT dataset).

## 🏗️ Architectures
| Model | Test Accuracy | Params |
|-------|--------------|--------|
| MLP Baseline | 40.41% | ~6.3M |
| Custom CNN | 80.85% | ~164K |
| ResNet-18 (Transfer Learning) | 93.91% | ~11.2M |
| ViT-B/16 (Fine-tuned) | 81.43% | ~86M |

## 📂 Dataset
- **EuroSAT** — 27,000 satellite images, 10 classes
- Split: 80% train / 20% test

## 🔬 Ablation Studies
3 experiments on CNN architecture:
1. CNN without BatchNorm (baseline)
2. CNN with BatchNorm
3. Deeper CNN (3 conv layers + Dropout)

## 📊 Visualizations
- Training loss & accuracy curves
- Confusion matrices
- Grad-CAM on ResNet-18

## 🚀 How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TON_LIEN_COLAB_ICI)

### 2. Install dependencies
```python
pip install grad-cam timm
```

### 3. Run all cells in order

## 🔧 Requirements
- Python 3.12
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- grad-cam

## 📈 Key Findings
- **ResNet-18** achieves best accuracy (93.91%) thanks to transfer learning
- **ViT underperforms** ResNet on EuroSAT due to limited dataset size (ViTs need more data)
- **BatchNorm** significantly improves CNN convergence
- **MLP** struggles without spatial inductive bias

## 🌱 Reproducibility
All experiments use `random seed = 42`
