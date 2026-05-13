import os
import torch
from ultralytics import YOLO

# =====================================================
# MPS MEMORY SAFETY (IMPORTANT)
# =====================================================
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# =====================================================
# CUSTOM MODULE REGISTRATION
# =====================================================
from models.wfdm import WFDM
from models.cbam import CBAM

import ultralytics.nn.tasks as tasks
tasks.WFDM = WFDM
tasks.CBAM = CBAM


def main():

    print("\n🚀 WFDM-YOLO Training on Apple Silicon (M4)\n")

    # =====================================================
    # MODEL
    # =====================================================
    model = YOLO(
        "yolov8_wfdm_p2.yaml",
        task="detect"
    )

    # Load pretrained weights
    model.load("yolov8n.pt")

    # =====================================================
    # TRAIN CONFIG (M4 STABLE)
    # =====================================================
    model.train(

        # Dataset
        data='data/exdark_corrupt_all.yaml',

        # Core training
        epochs=180,
        imgsz=640,
        device="mps",

        # =================================================
        # MEMORY FIX (VERY IMPORTANT)
        # =================================================
        batch=6,                 # 🔥 FIX: prevents OOM
        amp=False,               # MPS instability with AMP OFF
        cache='disk',           # 🔥 safer than RAM cache
        workers=0,              # macOS stability

        # =================================================
        # OPTIMIZER
        # =================================================
        optimizer='SGD',
        lr0=0.004,
        lrf=0.1,
        momentum=0.937,
        weight_decay=5e-4,

        warmup_epochs=5,
        cos_lr=True,

        # =================================================
        # AUGMENTATION (LIGHT FOR STABILITY)
        # =================================================
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.2,

        mosaic=0.5,
        mixup=0.1,

        scale=0.3,
        translate=0.05,

        close_mosaic=20,

        multi_scale=False,   # 🔥 CRITICAL FIX (stops 768 spikes)

        # =================================================
        # TRAIN CONTROL
        # =================================================
        patience=50,
        save=True,

        # =================================================
        # OUTPUT
        # =================================================
        project='wfdyolo_runs',
        name='m4_wfdm_final'
    )


if __name__ == '__main__':
    main()