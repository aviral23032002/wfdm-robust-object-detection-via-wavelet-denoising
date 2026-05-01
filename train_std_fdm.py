from ultralytics import YOLO
import time

# --- DYNAMIC MODULE INJECTION ---
import ultralytics.nn.tasks as tasks
from models.std_fdm import StandardFDM

# Inject our custom class directly into YOLO's internal registry
tasks.StandardFDM = StandardFDM
# ---------------------------------

print("Building Custom YOLOv8 with Standard Residual FDM Architecture...")
# CRITICAL FIX: Load custom YAML, then explicitly load pretrained weights!
model = YOLO('yolov8n-fdm.yaml', task='detect').load('yolov8n.pt')

print("Starting Standard FDM Training on M4 GPU (MPS)...")
start_time = time.time()

results = model.train(
    data='data/exdark.yaml',  
    epochs=100,               
    imgsz=640,
    device='mps',             
    batch=16,                 
    project='wfdm_runs',      
    name='std_fdm_residual',  
    plots=True,
    verbose=True
)

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"✅ Standard FDM training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")

print("\nEvaluating Standard FDM on the clean ExDark Test Set...")
metrics = model.val(split='test')
print(f"Final Standard FDM mAP@0.5: {metrics.box.map50:.4f}")