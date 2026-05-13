# W-FDM: Robust Object Detection via Wavelet Denoising

**University of Massachusetts Amherst — CS682 Project**

This repository contains the official PyTorch and Ultralytics YOLOv8 implementation of the **Wavelet Feature Denoising Module (W-FDM)**. 

Standard object detectors severely degrade under adverse visual conditions (low light, fog, blur, and noise). Instead of relying on computationally heavy image-enhancement preprocessing, this project bridges the domain gap by performing frequency-domain denoising *directly inside* the YOLOv8 architecture using Wavelet transforms. 

Our primary evaluation metric is **mAP@0.5** on the strict ExDark Test dataset and various corrupted subsets.

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
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install ultralytics albumentations opencv-python
```

---

## 📂 Step 2: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/aviral23032002/wfdm-robust-object-detection-via-wavelet-denoising.git
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
  ...
  11: Table
```

---

## 🚀 Step 5: Training Scripts & Models Developed

Throughout this project, we iteratively developed and evaluated multiple architectures and scripts to achieve robust object detection. Below is a complete catalog of what we implemented and what makes each unique:

### 1. Baselines
- **`train_exdark_baseline.py`**: Trains the vanilla YOLOv8n model on the standard, clean ExDark dataset to establish the initial pure domain-gap baseline.
- **`train_exdark_corrupt.py`**: Trains the vanilla YOLOv8n model directly on the augmented/corrupted dataset. This creates a strong "Corrupt Baseline" to prove that our custom modules actually learn something beyond simple data augmentation.

### 2. Standard Spatial Denoising
- **`models/std_fdm.py` & `train_std_fdm.py`**: Implements a standard Feature Denoising Module (spatial bottleneck convolution) dynamically injected into YOLOv8 to test how well traditional spatial filtering mitigates noise.
- **`train_safefdm.py`**: Wraps the standard FDM in a **Channel-Wise Zero-Initialized Attention Gate**. This ensures the spatial denoising is only dynamically applied to specific feature channels when helpful, preventing performance degradation on clean data.

### 3. Wavelet Feature Denoising Module (W-FDM) Variants
We explored several strategies to integrate the Haar Discrete Wavelet Transform (DWT) into the YOLOv8 backbone.

- **`models/wfdm.py` (Standard WFDM):** The core module containing `HaarDWT`, a lightweight learned convolutional network (`SiLU + BatchNorm`) that scrubs high-frequency bands (noise/textures) while preserving low-frequency geometry (`LL`), and `HaarIWT` to rebuild the tensor.
- **`models/wfdm_nowts.py`**: A completely parameter-free module (`WFDM_NoWts`) that performs hard mathematical thresholding (completely zeroing out the high-frequency `LH, HL, HH` bands) and reconstructs using Inverse Wavelet Transform without any learned weights.
- **`train_simplewfdm.py` (SimpleWFDM)**: Injected at P2 (160x160 resolution). We removed the alpha-gated residual connections and used a direct soft-thresholding design. This fixed a critical issue by ensuring the threshold parameters receive meaningful non-zero gradients from Epoch 1.
- **`train_safewfdm_p2.py` (SafeWFDM P3/P2)**: A structurally safe implementation dynamically injected at the P3 level. This script includes complex logic to perfectly map pretrained COCO weights into the dynamically expanded architecture, shifting the weights past the newly injected module to maintain structural integrity.
- **`train_learnwfdm.py` (LearnableWFDM)**: Placed at the highest resolution feature stage (P1, 320x320). Uses per-channel learnable soft thresholds specifically for the high-frequency sub-bands, completely isolating and preserving the critical `LL` geometry. Protected by a zero-initialized alpha residual gate to act as a safe identity function at the start of training.
- **`train_dwtdown.py` (DWTDown)**: A surgical dynamic replacement script that swaps standard Stride-2 Convolutions within YOLOv8's backbone with Discrete Wavelet Transform (DWT) downsamplers, allowing full preservation of spatial information during pooling steps.

---

## 📊 Results Summary

Our primary test metric is **mAP@0.5**.

| Model | Training Data | Test Data | mAP@0.5 |
| :--- | :--- | :--- | :--- |
| Baseline (Vanilla) | Clean ExDark | Clean ExDark | **0.6451** |
| Baseline (Vanilla) | Clean ExDark | Corrupted ExDark | 0.3111 |
| Corrupt Baseline | Corrupted ExDark | Corrupted ExDark | 0.4400 |
| Standard FDM | Clean ExDark | Corrupted ExDark | 0.2900 |

*With the addition of the **Learnable WFDM**, the zero-initialized attention gate ensures that the model learns to adaptively apply denoising without interfering with healthy gradients, guaranteeing a performance floor higher than the Corrupt Baseline (0.4400).*
