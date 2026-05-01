from ultralytics import YOLO
import time

print("Loading vanilla YOLOv8 nano model...")
model = YOLO('yolov8n.pt')

print("Starting ExDark Baseline Training on M4 GPU (MPS)...")
start_time = time.time()

# Train the model on the official ExDark split
results = model.train(
    data='data/exdark.yaml',  
    epochs=100,               # The 100 epochs specified in your proposal
    imgsz=640,
    device='mps',             # Routing to your M4 GPU
    batch=16,                 # Standard batch size that fits comfortably in Apple Silicon memory
    project='wfdm_runs',      # Creates a neat folder for your experiment results
    name='baseline_yolov8n',  # Names this specific training run
    plots=True,
    verbose=True
)

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"✅ Baseline training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")

# Automatically evaluate on the strict Test set after training
print("\nEvaluating on the official ExDark Test Set...")
metrics = model.val(split='test')
print(f"Final Baseline mAP@0.5: {metrics.box.map50:.4f}")