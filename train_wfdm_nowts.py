from ultralytics import YOLO
import time
import ultralytics.nn.tasks as tasks
from models.wfdm_nowts import WFDM_NoWts

# Inject custom class
tasks.WFDM_NoWts = WFDM_NoWts

print("Building Custom YOLOv8 with Hard-Math Wavelets (No Weights)...")
# CRITICAL FIX: Load pretrained COCO weights!
model = YOLO('yolov8n-wfdm-nowts.yaml', task='detect').load('yolov8n.pt')

print("Starting W-FDM (No Weights) Training...")
start_time = time.time()

results = model.train(
    data='data/exdark.yaml',  
    epochs=100,               
    imgsz=640,
    device='mps',             
    batch=16,                 
    project='wfdm_runs',      
    name='wfdm_no_wts',       
    plots=True,
    verbose=True
)

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"✅ W-FDM (No Weights) training complete in {int(hours)}h {int(minutes)}m {seconds:.2f}s!")

print("\nEvaluating W-FDM on the clean ExDark Test Set...")
metrics = model.val(split='test')
print(f"Final W-FDM (No Wts) mAP@0.5: {metrics.box.map50:.4f}")