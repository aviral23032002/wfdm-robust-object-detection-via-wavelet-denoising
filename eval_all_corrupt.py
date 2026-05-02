from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from models.std_fdm import StandardFDM

# Inject custom class so YOLO doesn't crash reading your Std_FDM model
tasks.StandardFDM = StandardFDM

dataset_yaml = 'data/exdark_corrupt_test.yaml'
results = {}

print("\n=== STARTING ZERO-SHOT CORRUPTED EVALUATION ===\n")

# 1. Vanilla Baseline (Trained Clean)
print("--- 1. Testing Vanilla Baseline ---")
model_vanilla = YOLO('runs/detect/wfdm_runs/baseline_yolov8n/weights/best.pt')
metrics_vanilla = model_vanilla.val(data=dataset_yaml, split='test', project='wfdm_evals', name='eval_baseline_corrupt')
results['Vanilla Baseline (Clean Train)'] = metrics_vanilla.box.map50

# 2. Corrupted Baseline (Trained Corrupt)
print("\n--- 2. Testing Corrupted Baseline ---")
model_corrupt = YOLO('runs/detect/wfdm_runs/baseline_corrupt_aug/weights/best.pt')
metrics_corrupt = model_corrupt.val(data=dataset_yaml, split='test', project='wfdm_evals', name='eval_corruptaug_corrupt')
results['Corrupt Baseline (Corrupt Train)'] = metrics_corrupt.box.map50

# 3. Standard FDM (Trained Clean)
print("\n--- 3. Testing Standard FDM ---")
model_fdm = YOLO('runs/detect/wfdm_runs/std_fdm_residual/weights/best.pt')
metrics_fdm = model_fdm.val(data=dataset_yaml, split='test', project='wfdm_evals', name='eval_stdfdm_corrupt')
results['Standard FDM (Clean Train)'] = metrics_fdm.box.map50

# Print Final Cheat Sheet for your Paper
print("\n=== FINAL ABLATION SCORES (CORRUPTED TEST SET) ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")