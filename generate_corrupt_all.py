import os
import shutil
import cv2
import random
import albumentations as A

def process_split(split_name, prob=1.0):
    clean_img_dir = f"data/images/{split_name}"
    clean_lbl_dir = f"data/labels/{split_name}"
    
    corrupt_img_dir = f"data/images/{split_name}_corrupt"
    corrupt_lbl_dir = f"data/labels/{split_name}_corrupt"
    
    # Check if the clean directory exists before processing
    if not os.path.exists(clean_img_dir):
        print(f"Skipping {split_name.upper()} - Source directory '{clean_img_dir}' not found.")
        return

    os.makedirs(corrupt_img_dir, exist_ok=True)
    os.makedirs(corrupt_lbl_dir, exist_ok=True)

    print(f"\nProcessing {split_name.upper()} split (Corruption Probability: {prob*100}%)...")
    
    # Updated Albumentations API Parameters
    transform = A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.3, 0.7), alpha_coef=0.08, p=1.0),
            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
            A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
        ], p=1.0)
    ])

    images = [f for f in os.listdir(clean_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(images)} images in {split_name}.")

    corrupted_count = 0
    for img_name in images:
        src_img = os.path.join(clean_img_dir, img_name)
        dst_img = os.path.join(corrupt_img_dir, img_name)
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(clean_lbl_dir, lbl_name)
        dst_lbl = os.path.join(corrupt_lbl_dir, lbl_name)

        # Apply corruption based on probability
        if random.random() < prob:
            img = cv2.imread(src_img)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = transform(image=img_rgb)
                aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_img, aug_img_bgr)
                corrupted_count += 1
            else:
                shutil.copy(src_img, dst_img) # Fallback
        else:
            # Copy the clean image directly if not corrupted
            shutil.copy(src_img, dst_img)
            
        # Always copy the label
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

    print(f"✅ {split_name.upper()} generation complete! Corrupted: {corrupted_count} | Clean: {len(images) - corrupted_count}")

def generate_all():
    print("Initializing Unified Weather Augmentation Pipeline...")
    
    # Configure probabilities:
    # Train = 50% corrupt (so the model learns clean data too)
    # Val & Test = 100% corrupt (to strictly evaluate robustness)
    splits = {
        'train': 0.5,
        'val': 1.0,
        'test': 1.0
    }
    
    for split, prob in splits.items():
        process_split(split, prob)
        
    print("\n🎉 ALL SPLITS SUCCESSFULLY PROCESSED!")

if __name__ == "__main__":
    generate_all()
