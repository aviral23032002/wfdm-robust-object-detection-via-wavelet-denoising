import os
import torch
from ultralytics import YOLO

# CRITICAL: Import the WFDM module so PyTorch knows how to unpickle the custom class
import models.wfdm 

def main():
    checkpoint_dir = "model_checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Directory '{checkpoint_dir}' does not exist.")
        return
        
    # Add pure baseline (yolov8n.pt without ExDark training) to the list
    checkpoints = ["yolov8n.pt"] + sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    
    results_summary = {}

    # Automatically handle device selection (Colab CUDA vs Mac MPS)
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
        
    print(f"Using device: {device}")

    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Evaluating Model: {ckpt}")
        print(f"{'='*60}")
        
        try:
            model = YOLO(ckpt, task='detect')
            
            # Using relative path so it works on both Mac and Colab
            metrics = model.val(
                data='data/exdark_corrupt.yaml',
                device=device,
                augment=True
            )
            
            map50 = metrics.box.map50
            map95 = metrics.box.map
            
            results_summary[ckpt] = {
                "mAP@0.5": map50,
                "mAP@0.5:0.95": map95
            }
            
            print(f"\n🔥 RESULTS FOR {os.path.basename(ckpt)}:")
            print(f"mAP@0.5: {map50:.4f}")
            print(f"mAP@0.5:0.95: {map95:.4f}\n")
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt}: {e}")

    print("\n" + "="*60)
    print("🏆 FINAL EVALUATION SUMMARY")
    print("="*60)
    for ckpt, res in results_summary.items():
        name = os.path.basename(ckpt)
        print(f"{name}:")
        print(f"  - mAP@0.5:      {res['mAP@0.5']:.4f}")
        print(f"  - mAP@0.5:0.95: {res['mAP@0.5:0.95']:.4f}")

if __name__ == '__main__':
    main()
