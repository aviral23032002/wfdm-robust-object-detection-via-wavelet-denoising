import os
import shutil
import cv2
import random
import albumentations as A

def create_corrupted_dataset():
    # Define paths
    clean_train_img = "data/images/train"
    clean_train_lbl = "data/labels/train"
    
    corrupt_train_img = "data/images/train_corrupt"
    corrupt_train_lbl = "data/labels/train_corrupt"
    
    # Create new directories
    os.makedirs(corrupt_train_img, exist_ok=True)
    os.makedirs(corrupt_train_lbl, exist_ok=True)

    print("Initializing Weather Augmentation Pipeline...")
    
    # Define the corruption pipeline
    transform = A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, p=1.0),
            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
            A.GaussNoise(var_limit=(20.0, 60.0), p=1.0),
        ], p=1.0)
    ])

    images = [f for f in os.listdir(clean_train_img) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Processing {len(images)} training images...")

    corrupted_count = 0
    for img_name in images:
        src_img = os.path.join(clean_train_img, img_name)
        dst_img = os.path.join(corrupt_train_img, img_name)
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(clean_train_lbl, lbl_name)
        dst_lbl = os.path.join(corrupt_train_lbl, lbl_name)

        # 50% chance to corrupt the image
        if random.random() < 0.5:
            img = cv2.imread(src_img)
            if img is not None:
                # Albumentations uses RGB, OpenCV uses BGR
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = transform(image=img_rgb)
                aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_img, aug_img_bgr)
                corrupted_count += 1
        else:
            # Copy the clean image directly
            shutil.copy(src_img, dst_img)
            
        # Always copy the label (bounding boxes remain the same)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

    print(f"✅ Dataset generation complete! Corrupted: {corrupted_count} | Clean: {len(images) - corrupted_count}")

if __name__ == "__main__":
    create_corrupted_dataset()