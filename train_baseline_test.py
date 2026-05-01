from ultralytics import YOLO
import time

print("Loading YOLOv8 nano model...")
model = YOLO('yolov8n.pt')  # This will download the tiny 3MB starting weights

print("Starting baseline training on M4 GPU (MPS)...")
start_time = time.time()

# Train the model
results = model.train(
    data='coco8.yaml',  # Tiny test dataset
    epochs=5,           # Just 5 epochs to test the hardware pipeline
    imgsz=640,
    device='mps',       # CRITICAL: This routes the math to your M4's GPU
    plots=True,
    verbose=True
)

end_time = time.time()
print(f"Hardware test complete in {end_time - start_time:.2f} seconds!")