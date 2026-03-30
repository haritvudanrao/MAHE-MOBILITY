import os
import numpy as np
from PIL import Image

# Create folders just in case
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

print("Creating dummy data...")

# Create 50 fake images and masks
for i in range(50):
    # Fake image — random colors, size 256x256, 3 channels (RGB)
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(f"data/images/{i:04d}.jpg")

    # Fake mask — each pixel gets a class number
    # 0 = background, 1 = road, 2 = car
    mask_array = np.random.randint(0, 3, (256, 256), dtype=np.uint8)
    mask = Image.fromarray(mask_array)
    mask.save(f"data/masks/{i:04d}.png")
    
print(f"Done! Created 50 images in data/images/")
print(f"Done! Created 50 masks in data/masks/")
print("You can now run the training code!")