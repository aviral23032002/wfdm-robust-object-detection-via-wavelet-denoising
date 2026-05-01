# W-FDM: Robust Object Detection via Wavelet Denoising

**University of Massachusetts Amherst — CS682 Project**

This repository contains the official PyTorch and Ultralytics YOLOv8 implementation of the **Wavelet Feature Denoising Module (W-FDM)**. 

Standard object detectors severely degrade under adverse visual conditions (low light, fog, blur, and noise). Instead of relying on computationally heavy image-enhancement preprocessing, this project bridges the domain gap by performing frequency-domain denoising *directly inside* the YOLOv8 architecture. 

Our primary evaluation metric is **mAP@0.5** on the strict ExDark Test dataset.

---

## 🛠️ Step 1: Apple Silicon (M-Series) Environment Setup
To ensure PyTorch utilizes the Mac's GPU (MPS - Metal Performance Shaders) instead of the CPU, you must use the native ARM64 architecture via Miniforge.

**1. Install Miniforge (if not already installed)**
Download and install [Miniforge for macOS ARM64](https://github.com/conda-forge/miniforge).

**2. Create the Conda Environment**
Open your terminal and run:
```bash
conda create -n wfdm python=3.10 -y
conda activate wfdm
```

**3. Install PyTorch Nightly & Dependencies**
Apple Silicon requires the nightly build of PyTorch for optimal MPS support.
```bash
pip install --pre torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/nightly/cpu](https://download.pytorch.org/whl/nightly/cpu)
pip install ultralytics albumentations opencv-python
```

---

## 📂 Step 2: Clone the Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/aviral23032002/wfdm-robust-object-detection-via-wavelet-denoising.git](https://github.com/aviral23032002/wfdm-robust-object-detection-via-wavelet-denoising.git)
cd wfdm-robust-object-detection-via-wavelet-denoising
```

---

## 🖼️ Step 3: Dataset Acquisition & Preparation
The original ExDark dataset uses a custom Piotr's Computer Vision Matlab Toolbox (PMT) text format for annotations. Our pipeline mathematically normalizes these into the standard YOLO format and strictly enforces the official literature splits (3,000 Train / 1,800 Val / 2,563 Test).

*(Note: Due to file size constraints, the 7GB dataset is `.gitignore`'d and must be downloaded manually).*

**1. Download the Data**
1. Download the raw images and annotations from the [ExDark GitHub Repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset).
2. Download the `imageclasslist.txt` file from the same repository (this contains the official split mapping).

**2. Organize the Folders**
Inside the cloned repository, create a `data` folder and arrange it exactly like this:
```text
wfdm-robust-object-detection/
├── data/
│   ├── raw_images/         # Contains the 12 class folders of JPEGs
│   ├── raw_annotations/    # Contains the 12 class folders of PMT text files
│   └── imageclasslist.txt  # The official split file
```

**3. Run the Formatting Script**
Execute the data preparation script. This script opens every single image, calculates the normalized YOLO coordinates, and distributes them into `train`, `val`, and `test` folders based on the strict `imageclasslist.txt` specifications.
```bash
python prep_exdark.py
```
*Wait for the `✅ Official Split Conversion complete!` message.*

---

## ⚙️ Step 4: YOLO Configuration

**1. Set Local Output Directories**
By default, Ultralytics saves runs and weights to global system folders. To keep everything contained within this project folder, run this configuration command:
```bash
yolo settings runs_dir="$(pwd)/runs" datasets_dir="$(pwd)/data" weights_dir="$(pwd)/weights"
```

**2. Configure the Data YAML**
Ensure your `data/exdark.yaml` file exists and contains the following paths and class mappings. This tells the YOLO trainer where to find the newly formatted data.

```yaml
# data/exdark.yaml
path: ./data
train: images/train
val: images/val
test: images/test

names:
  0: Bicycle
  1: Boat
  2: Bottle
  3: Bus
  4: Car
  5: Cat
  6: Chair
  7: Cup
  8: Dog
  9: Motorbike
  10: People
  11: Table
```

---

## 🚀 Step 5: Running the Vanilla Baseline (Row 1)
With the environment active and data prepped, you are ready to train the unmodified YOLOv8n model for 100 epochs. This will establish the initial domain gap baseline (Row 1 of our ablation study).

```bash
python train_exdark_baseline.py
```

The script will automatically detect your Apple Silicon chip, route the training to `device='mps'`, and evaluate the model blindly on the strict 2,563-image Test Set at the very end to generate the final `mAP@0.5` score.

---

## 🧠 Architecture: The W-FDM Module
The core of this project is the W-FDM block (located in `models/wfdm.py`), which acts as a smart frequency filter between the YOLOv8 Backbone and Neck.

1. **HaarDWT (Discrete Wavelet Transform):** Decomposes the incoming feature maps into four frequency bands:
   - `LL` (Low-Frequency): The core geometric shapes and spatial bounding box data.
   - `LH, HL, HH` (High-Frequency): The textural details, which contain the majority of the synthetic fog, noise, and low-light static.
2. **Denoising Block:** A lightweight, learned convolutional network (SiLU + BatchNorm) that scrubs the concatenated high-frequency bands without touching the critical `LL` geometry.
3. **HaarIWT (Inverse Wavelet Transform):** Recombines the clean `LL` band with the newly scrubbed high-frequency bands back into a standard spatial tensor for YOLO to process.
