from ultralytics import YOLO

def main():
    print("Loading vanilla YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')

    print("Starting Clean ExDark Baseline Training...")
    results = model.train(
        data='data/exdark.yaml',  
        epochs=100,               
        imgsz=640,
        batch=16,                 
        project='runs',      
        name='baseline_yolov8n_clean',  
        plots=True,
        verbose=True
    )

    print("\nEvaluating on the official ExDark Test Set...")
    metrics = model.val(split='test')
    print(f"Final Baseline mAP@0.5: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
