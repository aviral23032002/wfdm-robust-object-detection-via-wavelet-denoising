from ultralytics import YOLO
import time

print("Loading vanilla YOLOv8 nano model for Augmentation Run...")
model = YOLO('yolov8n.pt')

print("Starting ExDark Corrupted Training on M4 GPU (MPS)...")
start_time = time.time()

# Train the model on the corrupted ExDark split
results = model.train(
    data='data/exdark_corrupt.yaml',  # Points to the YAML with the corrupted train folder
    epochs=100,               
    imgsz=640,
    device='mps',             # Routing to the M4 GPU
    batch=16,                 
    project='wfdm_runs',      # Keeps results in the same project folder as Aviral's run
    name='baseline_corrupt_aug',  # Distinct name for this run
    plots=True,
    verbose=True
)

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"✅ Corrupted training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")

# Automatically evaluate on the strict (CLEAN) Test set after training
print("\nEvaluating Corrupted Model on the clean ExDark Test Set...")
metrics = model.val(split='test')
print(f"Final Corrupted Baseline mAP@0.5: {metrics.box.map50:.4f}")