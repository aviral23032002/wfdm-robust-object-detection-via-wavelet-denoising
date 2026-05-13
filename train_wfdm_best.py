import torch
from ultralytics import YOLO
from models.wfdm import inject_wfdm

def main():
    print("Loading trained Phase 1 weights...")
    # NOTE: Assuming weights are placed in runs/wfdm_phase1_backbone-2/weights/last.pt
    # Update this path if the weights are stored elsewhere locally
    try:
        model2 = YOLO('runs/wfdm_phase1_backbone-2/weights/last.pt', task='detect')
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure the Phase 1 weights exist at: runs/wfdm_phase1_backbone-2/weights/last.pt")
        return

    model2 = inject_wfdm(model2)

    # Unfreeze all parameters including the WFDM block
    for p in model2.model.parameters():
        p.requires_grad = True
    print("✅ All layers unfrozen — starting Phase 2")

    # The config parameters are kept EXACTLY as requested
    model2.train(
        data='data/exdark_corrupt.yaml',
        epochs=100,             
        imgsz=640, 
        device=0, 
        batch=32,
        amp=True, 
        cache='ram', 
        workers=8,
        
        # --- CRITICAL OPTIMIZER CHANGES ---
        optimizer='AdamW',      
        lr0=0.0005,             
        weight_decay=0.01,
        warmup_epochs=0,        
        cos_lr=True,
        
        # --- LOW-LIGHT AUGMENTATIONS ---
        mosaic=0.3,             
        mixup=0.1,
        hsv_v=0.4,              
        close_mosaic=15,        
        
        project='runs',
        name='wfdm_phase2_optimized'
    )

    print("Saved as best_wfdm_phase2_optimized.pt!")

    # =========================================================
    # VALIDATION WITH TEST-TIME AUGMENTATION (TTA)
    # =========================================================
    metrics = model2.val(
        data='data/exdark_corrupt.yaml',
        device=0,
        augment=True            
    )

    print("\n🔥 FINAL RESULTS")
    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)

if __name__ == '__main__':
    main()
