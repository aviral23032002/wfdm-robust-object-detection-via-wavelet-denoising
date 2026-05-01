import os
import shutil
import cv2
from pathlib import Path

CLASSES = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
           'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

def setup_yolo_directories(base_path):
    """Creates the necessary folder structure for YOLO training, validation, and testing."""
    dirs = ['images/train', 'images/val', 'images/test', 
            'labels/train', 'labels/val', 'labels/test']
    for d in dirs:
        Path(os.path.join(base_path, d)).mkdir(parents=True, exist_ok=True)

def process_dataset(data_dir):
    raw_images_dir = os.path.join(data_dir, 'raw_images')
    raw_anno_dir = os.path.join(data_dir, 'raw_annotations')
    split_file_path = os.path.join(data_dir, 'imageclasslist.txt')
    
    if not os.path.exists(split_file_path):
        print(f"Error: {split_file_path} not found.")
        print("Please download 'imageclasslist.txt' from the GitHub repo and put it in your data folder.")
        return

    print("Loading official ExDark splits from imageclasslist.txt...")
    split_map = {}
    with open(split_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Skip the header or empty lines
            if not line or line.lower().startswith('image'): 
                continue
                
            parts = line.split()
            if len(parts) >= 5:
                # The file includes paths like 'Bicycle/2015_00001.jpg', we just need the filename
                img_name = os.path.basename(parts[0]) 
                split_code = parts[4]
                
                # Apply the official mapping: 1=Train, 2=Val, 3=Test
                if split_code == '1': split_map[img_name] = 'train'
                elif split_code == '2': split_map[img_name] = 'val'
                elif split_code == '3': split_map[img_name] = 'test'

    setup_yolo_directories(data_dir)
    print("Scanning and distributing images into Official Train/Val/Test splits...")
    
    for class_name in CLASSES:
        class_img_path = os.path.join(raw_images_dir, class_name)
        class_anno_path = os.path.join(raw_anno_dir, class_name)
        
        if not os.path.exists(class_img_path) or not os.path.exists(class_anno_path):
            continue
            
        images = [f for f in os.listdir(class_img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in images:
            split_name = split_map.get(img_file)
            if not split_name:
                continue # Skip if the image isn't in their official list

            txt_file = img_file + '.txt'
            txt_path = os.path.join(class_anno_path, txt_file)

            if not os.path.exists(txt_path):
                continue

            src_img = os.path.join(class_img_path, img_file)
            img = cv2.imread(src_img)
            if img is None:
                continue
            img_height, img_width = img.shape[:2]

            # Copy Image to the correct official split folder
            dst_img = os.path.join(data_dir, f'images/{split_name}', img_file)
            shutil.copy(src_img, dst_img)
            
            base_name = os.path.splitext(img_file)[0]
            dst_label = os.path.join(data_dir, f'labels/{split_name}', base_name + '.txt')
            
            try:
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    
                with open(dst_label, 'w') as out_file:
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('%'): continue
                            
                        parts = line.split()
                        if len(parts) < 5: continue
                            
                        cls_name = parts[0]
                        if cls_name not in CLASSES: continue
                            
                        cls_id = CLASSES.index(cls_name)
                        x_min, y_min = float(parts[1]), float(parts[2])
                        box_w, box_h = float(parts[3]), float(parts[4])
                        
                        x_center = max(0.0, min(1.0, (x_min + (box_w / 2.0)) / img_width))
                        y_center = max(0.0, min(1.0, (y_min + (box_h / 2.0)) / img_height))
                        norm_w = max(0.0, min(1.0, box_w / img_width))
                        norm_h = max(0.0, min(1.0, box_h / img_height))
                        
                        out_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                        
            except Exception as e:
                 print(f"Skipping {txt_path} due to error: {e}")

    print("✅ Official Split Conversion complete! Your data is fully prepped.")

if __name__ == '__main__':
    process_dataset(os.path.join(os.getcwd(), 'data'))