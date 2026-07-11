# 🛩️ Small Object Detection for UAV Images using Improved YOLOv8s

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00BFFF)](https://github.com/ultralytics/ultralytics)

This repository contains the implementation of an enhanced YOLOv8s model designed for **small object detection in Unmanned Aerial Vehicle (UAV) imagery**. Our model outperforms the current state‑of‑the‑art on both **VisDrone2019** and **DOTA** datasets while being more efficient.

---

## 📊 Key Results

| Dataset      | mAP50 (%) | mAP50‑95 (%) | Params (M) | FLOPs (G) | Input Size |
|--------------|-----------|--------------|------------|-----------|------------|
| VisDrone2019 | **48.06**  | 31.9         | 9.5        | 50.0      | 768        |
| DOTA         | **74.4**   | 49.7         | 9.5        | 50.0      | 768        |

Compared to the baseline YOLOv8s (36.4% VisDrone, 70.7% DOTA) and the previous improved YOLOv8s by Ni et al. (47.1% VisDrone, 74.2% DOTA), our model achieves **higher accuracy with fewer parameters and lower FLOPs**.

---

## 🚀 Features

- **Lightweight PMSE** – depthwise separable convolutions reduce parameters.
- **SCFPN** – scale compensation FPN with an ultra‑small detection layer (P2).
- **Coordinated Attention** – improves positional awareness after SCFPN.
- **WIOUv3 Loss** – dynamic focusing for small objects.
- **Dynamic Label Assignment** – better anchor matching for tiny objects.
- **EMA** – exponential moving average for stable training.
- **Multi‑scale Training** – configurable for scale invariance.
- **All Paper Hyperparameters** – learning rate 0.0005, momentum 0.9, loss weights, etc.
- **Checkpoint Resume** – auto‑detects `last.pt` and resumes.
- **Live Metrics** – epoch‑by‑epoch mAP50, precision, recall.
- **Comprehensive Plots** – training curves, confusion matrix, PR curve, F1 curve.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.4.0+ (CUDA 12.4 recommended)
- Ultralytics YOLOv8

### Clone the repository
```bash
git clone https://github.com/Avinashsingh1215/Small_Object_Detection_Using_Improved_YOLOv8s_for_UAV_Images.git
cd Small_Object_Detection_Using_Improved_YOLOv8s_for_UAV_Images
