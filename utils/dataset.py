import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class DrivableDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image filenames
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        print(f"Found {len(self.images)} images and {len(self.masks)} masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Open image as RGB (3 colors)
        image = Image.open(img_path).convert("RGB")
        # Open mask as grayscale (just numbers, no color)
        mask = Image.open(mask_path).convert("L")

        # Resize both to 256x256
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        # Convert to numpy arrays
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.int64)

        # Normalize image pixels from 0-255 to 0-1
        image = image / 255.0

        # Convert to PyTorch tensors
        # Image: (Height, Width, Channels) -> (Channels, Height, Width)
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask)

        return image, mask