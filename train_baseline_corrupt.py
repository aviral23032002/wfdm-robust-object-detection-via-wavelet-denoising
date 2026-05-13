from ultralytics import YOLO

def main():
    print("Loading vanilla YOLOv8 nano model for Augmentation Run...")
    model = YOLO('yolov8n.pt')

    print("Starting ExDark Corrupted Baseline Training...")
    results = model.train(
        data='data/exdark_corrupt.yaml', 
        epochs=100,               
        imgsz=640,
        batch=16,                 
        project='runs',      
        name='baseline_corrupt_aug',  
        plots=True,
        verbose=True
    )

    print("\nEvaluating Corrupted Model on the clean ExDark Test Set...")
    metrics = model.val(split='test')
    print(f"Final Corrupted Baseline mAP@0.5: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
