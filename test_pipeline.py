import albumentations as A
import cv2
import urllib.request
import numpy as np
import os

print("Downloading a sample test image...")
url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
urllib.request.urlretrieve(url, "sample.jpg")

# Load the image using OpenCV
image = cv2.imread("sample.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for Albumentations

print("Building the W-FDM corruption pipeline...")
# This matches the 0.5 probability requirement from your proposal
wfdm_transform = A.Compose([
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.Blur(blur_limit=7, p=1.0),
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, p=1.0)
    ], p=0.5), # 50% chance to apply one of the above adverse conditions
    A.Resize(640, 640) # Standard YOLOv8 input size
])

print("Applying corruptions and generating 3 test samples...")
for i in range(3):
    # Apply the transformation
    transformed = wfdm_transform(image=image)
    transformed_image = transformed["image"]
    
    # Convert back to BGR for saving with OpenCV
    transformed_image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
    
    output_name = f"corrupted_sample_{i+1}.jpg"
    cv2.imwrite(output_name, transformed_image_bgr)
    print(f"Saved {output_name}")

print("Done! Open your folder and check the corrupted_sample images.")