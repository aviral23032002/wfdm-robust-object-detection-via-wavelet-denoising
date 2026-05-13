"""
train_safewfdm_p3.py — YOLOv8 + SafeWFDM P3 Wavelet Training Script
==================================================================
PHASE 2: SEMANTIC FREQUENCY DENOISING (STABILIZED)
------------------------------------------------------------------
FIXES:
  1. Attribute Error: Fixed .nc access (now using model.nc).
  2. Index Shifting: insertion point at Index 5 (P3).
  3. Nano Scaling: Ensure YAML uses [64] for SafeWFDM.
"""

import time
import torch
import ultralytics.nn.tasks as tasks

# --- CUSTOM MODULE INJECTION (must happen before YOLO parses the YAML) ---
from models.wfdm import SafeWFDM
tasks.SafeWFDM = SafeWFDM
# -------------------------------------------------------------------------

from ultralytics import YOLO

def load_pretrained_safe_p3(model, pt_path: str):
    """
    Safely transfer weights from COCO-pretrained YOLOv8n into P3-Wavelet architecture.
    """
    print(f"\n[Step 3] Loading weights from {pt_path} with P3 Index Shift (+1)...")
    
    # weights_only=False required for PyTorch 2.6+ 
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'].float().state_dict() if isinstance(checkpoint, dict) else checkpoint.state_dict()
    
    target_state_dict = model.model.state_dict()
    new_state_dict = {}
    matched_layers = 0

    for key, value in state_dict.items():
        if key.startswith('model.'):
            parts = key.split('.')
            layer_idx = int(parts[1])
            
            # SafeWFDM is at Index 5. Original layers 5+ move to idx+1.
            if layer_idx >= 5:
                parts[1] = str(layer_idx + 1)
            
            new_key = '.'.join(parts)
        else:
            new_key = key

        # Shape verification (Filters out the 80-class Detection Head)
        if new_key in target_state_dict:
            if value.shape == target_state_dict[new_key].shape:
                new_state_dict[new_key] = value
                matched_layers += 1

    model.model.load_state_dict(new_state_dict, strict=False)
    print(f"  ✅ Successfully mapped {matched_layers} pretrained layers.")
    print(f"  ✅ Detection head reset for {model.model.yaml['nc']} classes.")
    return model

def main():
    # ------------------------------------------------------------------
    # 1. Haar Round-Trip Verification
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STEP 1: HAAR WAVELET MATHEMATICAL SANITY CHECK")
    print("=" * 65)
    from models.wfdm import HaarDWT, HaarIWT
    all_ok = True
    for c in [3, 32, 64]:
        dwt = HaarDWT(c)
        iwt = HaarIWT(c)
        x   = torch.randn(2, c, 64, 64)
        rec = iwt(dwt(x))
        ok  = torch.allclose(x, rec, atol=1e-5)
        all_ok = all_ok and ok
        print(f"  Channels: {c:>3d} | Reconstruction: {'✅ PASS' if ok else '❌ FAIL'}")

    if not all_ok:
        raise RuntimeError("Wavelet Reconstruction Error. Check models/wfdm.py")
    print("  Frequency Domain Logic Confirmed.\n")

    # ------------------------------------------------------------------
    # 2. Build Architecture
    # ------------------------------------------------------------------
    print("=" * 65)
    print("STEP 2: BUILDING P3 WAVELET ARCHITECTURE")
    print("=" * 65)
    # Ensure your YAML has SafeWFDM set to [64]
    model = YOLO('yolov8n-safewfdm-p2.yaml', task='detect')
    print(f"  Architecture defined with {model.model.yaml['nc']} classes.")

    # ------------------------------------------------------------------
    # 3. Load Backbone Weights
    # ------------------------------------------------------------------
    model = load_pretrained_safe_p3(model, 'yolov8n.pt')

    # ------------------------------------------------------------------
    # 4. Train on NVIDIA A100
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 4: TRAINING ON NVIDIA A100")
    print("=" * 65)

    start_time = time.time()

    results = model.train(
        data='data/exdark_corrupt_train_test.yaml',  
        epochs=100,               
        imgsz=640,
        device='mps',             
        batch=16,                 
        project='wfdm_runs',      
        name='safewfdm_p3_recovery_v2',       
        plots=True,
        verbose=True,
        amp=True,                 
        cache=True,               
        
        # --- RECOVERY HYPERPARAMETERS ---
        lr0=0.01,                 # Increased for detection head convergence
        warmup_epochs=5.0,        # Extended warmup to stabilize the head
        cos_lr=True,              # Smooth decay
        close_mosaic=10,   
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # ------------------------------------------------------------------
    # 5. Evaluate on 100% Corrupted Test Set
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("STEP 5: ZERO-SHOT EVALUATION")
    print("=" * 65)

    metrics = model.val(
        data='data/exdark_corrupt_test.yaml',
        split='test'
    )

    print(f"\nFINAL RECOVERY RESULTS:")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  Precision    : {metrics.box.p.mean():.4f}")
    print(f"  Recall       : {metrics.box.r.mean():.4f}")

if __name__ == '__main__':
    main()