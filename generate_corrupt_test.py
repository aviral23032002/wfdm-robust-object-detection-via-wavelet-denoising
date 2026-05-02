import os
import shutil
import cv2
import albumentations as A

def create_corrupted_test_dataset():
    clean_test_img = "data/images/test"
    clean_test_lbl = "data/labels/test"
    
    corrupt_test_img = "data/images/test_corrupt"
    corrupt_test_lbl = "data/labels/test_corrupt"
    
    os.makedirs(corrupt_test_img, exist_ok=True)
    os.makedirs(corrupt_test_lbl, exist_ok=True)

    print("Initializing Weather Augmentation Pipeline for TEST set...")
    
    # --- UPDATED API PARAMETERS ---
    transform = A.Compose([
        A.OneOf([
            # Combined into fog_coef_range
            A.RandomFog(fog_coef_range=(0.3, 0.7), alpha_coef=0.08, p=1.0),
            
            A.GaussianBlur(blur_limit=(5, 9), p=1.0),
            
            # Replaced var_limit with std_range (scaled as a fraction of max pixel value)
            A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
        ], p=1.0)
    ])

    images = [f for f in os.listdir(clean_test_img) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Processing {len(images)} test images...")

    for img_name in images:
        src_img = os.path.join(clean_test_img, img_name)
        dst_img = os.path.join(corrupt_test_img, img_name)
        
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl = os.path.join(clean_test_lbl, lbl_name)
        dst_lbl = os.path.join(corrupt_test_lbl, lbl_name)

        # 100% chance to corrupt the test images
        img = cv2.imread(src_img)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = transform(image=img_rgb)
            aug_img_bgr = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_img, aug_img_bgr)
            
        # Always copy the label
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, dst_lbl)

    print(f"✅ Test dataset generation complete! 100% of {len(images)} images corrupted.")

if __name__ == "__main__":
    create_corrupted_test_dataset()