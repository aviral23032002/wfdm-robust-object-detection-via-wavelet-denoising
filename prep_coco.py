import os
from pycocotools.coco import COCO

# --- 1. SET YOUR PATHS ---
coco_annotation_file = './datasets/coco/annotations/instances_val2017.json' 
output_labels_dir = './datasets/coco/labels/val2017' 
# -------------------------

os.makedirs(output_labels_dir, exist_ok=True)

print("Loading COCO API (This takes a few seconds)...")
coco = COCO(coco_annotation_file)

# The Magic Dictionary: Maps COCO's 80 IDs -> ExDark's 12 IDs
# ExDark Names: ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']
coco_to_exdark = {
    1: 10,  # COCO Person      -> ExDark People
    2: 0,   # COCO Bicycle     -> ExDark Bicycle
    3: 4,   # COCO Car         -> ExDark Car
    4: 9,   # COCO Motorcycle  -> ExDark Motorbike
    6: 3,   # COCO Bus         -> ExDark Bus
    9: 1,   # COCO Boat        -> ExDark Boat
    17: 5,  # COCO Cat         -> ExDark Cat
    18: 8,  # COCO Dog         -> ExDark Dog
    44: 2,  # COCO Bottle      -> ExDark Bottle
    47: 7,  # COCO Cup         -> ExDark Cup
    62: 6,  # COCO Chair       -> ExDark Chair
    67: 11  # COCO Dining Table-> ExDark Table
}

print("\nTranslating COCO Annotations to ExDark YOLO Format...")

valid_coco_ids = list(coco_to_exdark.keys())
img_ids = coco.getImgIds()
converted_count = 0

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_w = img_info['width']
    img_h = img_info['height']
    
    # Get all annotations for this specific image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    yolo_lines = []
    
    for ann in anns:
        coco_cat_id = ann['category_id']
        
        # If it's a class we actually care about (like a Car, not a Giraffe)
        if coco_cat_id in valid_coco_ids:
            exdark_id = coco_to_exdark[coco_cat_id]
            
            # COCO bounding box format is [x_min, y_min, width, height]
            x_min, y_min, w, h = ann['bbox']
            
            # Convert to YOLO normalized center format [x_center, y_center, w, h]
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # Clamp values between 0 and 1 just to be safe
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            yolo_lines.append(f"{exdark_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Save the .txt file ONLY if there is at least one valid object in the image
    if yolo_lines:
        txt_filename = f"{img_info['file_name'].replace('.jpg', '.txt')}"
        txt_filepath = os.path.join(output_labels_dir, txt_filename)
        
        with open(txt_filepath, 'w') as f:
            f.writelines(yolo_lines)
            
        converted_count += 1

print(f"✅ Success! Created {converted_count} YOLO label files perfectly matched to your ExDark model.")