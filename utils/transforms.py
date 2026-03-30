import torch

def normalize(image):
    # Make pixel values between 0 and 1
    return image / 255.0

def to_tensor(image, mask):
    # Convert numpy arrays to PyTorch tensors
    image = torch.tensor(image).permute(2, 0, 1).float()
    mask = torch.tensor(mask).long()
    return image, mask