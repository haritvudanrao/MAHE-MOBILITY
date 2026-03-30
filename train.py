import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from utils.nuscenes_dataset import NuScenesDataset
from model.unet import UNet
from model.loss import CombinedLoss

# ==============================
# SETTINGS
# ==============================
DATA_ROOT = "data"
VERSION = "v1.0-mini"
NUM_CLASSES = 6
BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 0.001
SAVE_PATH = "outputs/model.pth"

def train():
    print("Starting training...")
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Load dataset
    dataset = NuScenesDataset(DATA_ROOT, VERSION)

    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {train_size} images, validating on {val_size} images")

    # Create model, loss, optimizer
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f} | Val Loss = {avg_val:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Model saved!")

    print("Training complete!")
    print(f"Best model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()