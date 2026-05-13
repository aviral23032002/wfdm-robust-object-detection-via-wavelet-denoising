import time
import torch
from ultralytics import YOLO
from models.wfdm import DWTDown

def perform_architecture_surgery(model):
    """
    Dynamically swaps standard Stride-2 Convolutions with DWTDown modules.
    This bypasses the YOLO YAML parser bugs while keeping all pretrained weights intact.
    """
    print("\n[Architecture Surgery] Replacing Stride-2 Convs with DWTDown...")
    
    # Nano Architecture Scaling: Layer Index -> (In_Channels, Out_Channels)
    replacements = {
        1: (16, 32),    # P2 Downsampler
        3: (32, 64),    # P3 Downsampler
        5: (64, 128),   # P4 Downsampler
        7: (128, 256)   # P5 Downsampler
    }
    
    for idx, (c1, c2) in replacements.items():
        # Grab the original YOLO layer metadata
        old_module = model.model.model[idx]
        
        # Initialize our Wavelet replacement
        new_module = DWTDown(c1, c2)
        
        # Transfer the internal tracking metadata so YOLO doesn't break
        new_module.i = old_module.i
        new_module.f = old_module.f
        new_module.type = 'DWTDown'
        
        # Perform the swap
        model.model.model[idx] = new_module
        print(f"  ✅ Layer {idx} Replaced: Conv -> DWTDown(In: {c1}, Out: {c2})")
        
    return model

def main():
    print("STEP 1: Loading Standard Pretrained YOLOv8n...")
    # Load directly from the .pt file (No YAML required)
    model = YOLO('yolov8n.pt')
    
    # STEP 2: The Swap
    model = perform_architecture_surgery(model)

    # STEP 3: Train
    print("\nSTEP 3: Training on NVIDIA A100...")
    start_time = time.time()

    model.train(
        data='data/exdark_corrupt_all.yaml',
        epochs=100,
        imgsz=640,
        device="mps",         
        batch=16,        
        amp=True,
        cache=True,
        
        # Standard power settings
        lr0=0.01,          
        warmup_epochs=3.0, 
        
        project='wfdm_runs',
        name='dwtdown_full_backbone_surgical'
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\n✅ Training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == '__main__':
    main()