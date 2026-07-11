# 🛩️ Small Object Detection for UAV Images using Improved YOLOv8s

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00BFFF)](https://github.com/ultralytics/ultralytics)

This repository contains the implementation of an enhanced YOLOv8s model designed for **small object detection in Unmanned Aerial Vehicle (UAV) imagery**. Our model outperforms the current state‑of‑the‑art on both **VisDrone2019** and **DOTA** datasets while being more efficient.

---

## 📊 Key Results

| Model          | Dataset      | mAP50 (%) | mAP50‑95 (%) | Params (M) | FLOPs (G) |
|----------------|--------------|-----------|--------------|------------|-----------|
| YOLOv8s        | VisDrone     | 36.4      | 21.6         | 11.1       | 28.5      |
| Ni et al. (2024) | VisDrone   | 47.1      | 28.7         | 10.2       | 64.9      |
| **Ours**       | VisDrone     | **48.06** | **31.9**     | **9.5**    | **50.0**  |
| YOLOv8s        | DOTA         | 70.7      | 46.7         | 11.1       | 28.5      |
| Ni et al. (2024) | DOTA       | 74.2      | 49.7         | 10.2       | 64.9      |
| **Ours**       | DOTA         | **74.4**  | **49.7**     | **9.5**    | **50.0**  |

Our model surpasses the previous state‑of‑the‑art (Ni et al., 2024) with **higher accuracy** (+0.96% on VisDrone, +0.2% on DOTA), **fewer parameters** (9.5M vs 10.2M), and **lower FLOPs** (50G vs 64.9G).

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

## 📦 Install dependencies

### Prerequisites
- Python 3.10+
- PyTorch 2.4.0+ (CUDA 12.4 recommended)
- Ultralytics YOLOv8

### Clone the repository
```bash
git clone https://github.com/Avinashsingh1215/Small_Object_Detection_Using_Improved_YOLOv8s_for_UAV_Images.git
cd Small_Object_Detection_Using_Improved_YOLOv8s_for_UAV_Images
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation
#### VisDrone2019

1. Download from VisDrone official.

2. Place the VisDrone2019-DET-train and VisDrone2019-DET-val folders in data/visdrone/raw/.

3. Convert annotations to YOLO format:
   ```bash
   python scripts/convert_visdrone_to_yolo.py
   ```

#### DOTA v1.0

1. Download DOTA images and annotations from the official DOTA website.

2. Place train/ and val/ folders inside data/dota/raw/.

3. Tile images (640×640, 200 overlap) and generate the YOLO dataset:
    ```bash
   python tile_dota_640.py
     ```

## 🏋️ Training

#### VisDrone
```bash
python train.py --config configs/visdrone_config.yaml
```

#### DOTA
```bash
python train.py --config configs/dota_config.yaml
```

Use a larger model (YOLOv8m) for potentially higher accuracy:
```bash
python train.py --config configs/dota_config.yaml --large
```

## 🔍 Evaluation
#### Evaluate a trained model on the validation set:
```bash
python eval.py --weights runs/.../weights/best.pt --data configs/visdrone_config.yaml
```

#### Test‑Time Augmentation (TTA)
TTA typically adds 1–2% mAP:
```bash
yolo val model=best.pt data=data/dota/dota_tiled.yaml imgsz=768 augment=True plots=True save_json=True
```

## 📁 Project Structure
```bash
.
├── configs/                 # YAML configuration files (dataset paths, hyperparameters)
│   ├── dota_config.yaml
│   ├── dota_multiscale_training.yaml
│   └── visdrone_config.yaml
├── losses/                  # Custom loss functions (WIOUv3)
│   └── wiou_loss.py
├── models/                  # Custom neural network modules
│   ├── attention.py         # Coordinated Attention
│   ├── custom_yolo.py       # Custom YOLOv8 assembly
│   ├── pmse.py              # PMSE and LightweightPMSE
│   └── scfpn.py             # SCFPN with weighted fusion
├── scripts/                 # Dataset conversion utilities
│   └── convert_visdrone_to_yolo.py
├── utils/                   # Training utilities
│   ├── assigner.py          # Dynamic label assigner
│   ├── ema.py               # Exponential Moving Average
│   └── plots.py             # Plotting training curves
├── train.py                 # Main training entry point
├── eval.py                  # Evaluation script
├── tile_dota_640.py         # DOTA tiling script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

## 🙏 Acknowledgements

- **Base paper: Ni et al., A Small‑Object Detection Model Based on Improved YOLOv8s for UAV Image Scenarios, Remote Sensing 2024.

- **VisDrone2019 dataset: Zhu et al., Detection and Tracking Meet Drones Challenge, TPAMI 2021.

- **DOTA dataset: Xia et al., DOTA: A Large-scale Dataset for Object Detection in Aerial Images, CVPR 2018.

- **Ultralytics YOLOv8 framework – the foundation of our implementation.
