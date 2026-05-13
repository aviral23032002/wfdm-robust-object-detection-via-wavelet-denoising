from functools import cache
from ultralytics import YOLO
import time
import ultralytics.nn.tasks as tasks
from models.std_fdm import SafeFDM

# --- DYNAMIC MODULE INJECTION ---
tasks.SafeFDM = SafeFDM
# ---------------------------------

print("Building YOLOv8 with Channel-Wise Zero-Initialized P3 FDM...")
model = YOLO('yolov8n-safefdm.yaml', task='detect').load('yolov8n.pt')

print("Starting Safe FDM Training on Apple Silicon (MPS)...")
start_time = time.time()

results = model.train(
    # MUST point to the corrupted dataset so the attention gates learn the noise
    data='data/exdark_corrupt_train_test.yaml',  
    epochs=50,               
    imgsz=640,
    device='mps',             
    batch=32,
    workers = 8,
    cache = True, 
    amp = True,                 
    project='wfdm_runs',      
    name='safefdm_channel_p3',  
    plots=True,
    verbose=True
)

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"✅ Safe Channel-Wise FDM training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")

# Automatically trigger zero-shot evaluation on the severe test set
print("\nEvaluating Safe FDM on the 100% Corrupted Test Set...")
metrics = model.val(data='data/exdark_corrupt_test.yaml', split='test')
print(f"Final Safe FDM mAP@0.5: {metrics.box.map50:.4f}")