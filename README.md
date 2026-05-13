# W-FDM: Robust Object Detection via Wavelet Denoising

**University of Massachusetts Amherst — CS682 Project**

This repository contains the official PyTorch and Ultralytics YOLOv8 implementation of the **Wavelet Feature Denoising Module (W-FDM)**. 

Standard object detectors severely degrade under adverse visual conditions (low light, fog, blur, and noise). Instead of relying on computationally heavy image-enhancement preprocessing, this project bridges the domain gap by performing frequency-domain denoising *directly inside* the YOLOv8 architecture using Wavelet transforms. 

---

## 📂 Repository Structure

```text
wfdm-robust-object-detection/
├── data/
│   ├── raw_images/         # Raw JPEGs (12 classes)
│   ├── raw_annotations/    # PMT text files (12 classes)
│   ├── imageclasslist.txt  # Official ExDark split definitions
│   ├── exdark_corrupt.yaml # YOLO configuration file for corrupted data
│   └── exdark.yaml         # YOLO configuration file for clean data
├── models/
│   └── wfdm.py             # Core Wavelet Feature Denoising Module
├── model_checkpoints/      # Saved .pt weights for evaluation
├── prep_exdark.py          # Formats PMT annotations to YOLO format
├── generate_corrupt_all.py # Applies Fog, Blur, and Noise corruptions
├── train_baseline_clean.py # Trains YOLOv8n on clean data
├── train_baseline_corrupt.py # Trains YOLOv8n on corrupted data
├── train_wfdm_best.py      # Fine-tunes YOLOv8n with the injected W-FDM module
└── eval_models.py          # Runs zero-shot validation across all models
```

---

## 🛠️ Step 1: Environment Setup

Apple Silicon requires the nightly build of PyTorch for optimal MPS (GPU) support. 

```bash
conda create -n wfdm python=3.10 -y
conda activate wfdm
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install ultralytics albumentations opencv-python
```

---

## 🖼️ Step 2: Dataset Acquisition & Preparation

The original ExDark dataset uses a custom Piotr's Computer Vision Matlab Toolbox (PMT) text format for annotations. Our pipeline mathematically normalizes these into the standard YOLO format and enforces the official literature splits.

**1. Download the Data**
Download the raw images, annotations, and `imageclasslist.txt` from the [ExDark GitHub Repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) and place them in the `data/` directory.

**2. Format the Annotations**
Run the data preparation script to normalize the YOLO coordinates and distribute them into `train`, `val`, and `test` folders:
```bash
python prep_exdark.py
```

**3. Generate the Corrupted Dataset**
To evaluate domain robustness, run the unified corruption script. This applies Random Fog, Gaussian Blur, and Gaussian Noise using the Albumentations library. (Train gets a 50% chance of corruption, Val/Test get a 100% chance).
```bash
python generate_corrupt_all.py
```

---

## 🚀 Step 3: Training the Models

We provide 3 main training scripts to establish the baselines and train our custom module.

**1. Clean Baseline:** Train the vanilla YOLOv8n model on the standard, clean ExDark dataset.
```bash
python train_baseline_clean.py
```

**2. Corrupt Baseline:** Train the vanilla YOLOv8n model directly on the corrupted dataset to create a strong "Corrupt Baseline".
```bash
python train_baseline_corrupt.py
```

**3. W-FDM Fine-Tuning (Ours):** Unfreezes the backbone and dynamically injects the Wavelet Feature Denoising Module (W-FDM) into the P3 stage, fine-tuning the spatial attention gates using AdamW.
```bash
python train_wfdm_best.py
```

---

## 📊 Step 4: Evaluating the Models

We have included our final trained weights directly in the `model_checkpoints/` directory so you can verify our results immediately without retraining. 

To reproduce our final results, run the unified evaluation script. It automatically detects the trained weights and evaluates them on the strict 100% corrupted Test set using Test-Time Augmentation (TTA).

```bash
python eval_models.py
```

### 🏆 Final Results

| Model variant | mAP@0.5 | mAP@0.5:0.95 | vs. Base |
| :--- | :--- | :--- | :--- |
| **YOLOv8n zero-shot** | 0.0013 | 0.0003 | −99.6% |
| **YOLOv8n baseline (FT)** | 0.359 | 0.202 | — |
| **YOLOv8n + W-FDM (ours)** | **0.404** | **0.234** | **+12.5%** |

*Note: W-FDM successfully bridges the domain gap without relying on computationally heavy image-enhancement preprocessing, performing frequency-domain denoising directly inside the YOLOv8 architecture.*
