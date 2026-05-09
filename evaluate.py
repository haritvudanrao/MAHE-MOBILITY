import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from utils.nuscenes_dataset import NuScenesDataset
from model.unet import UNet

# ==============================
# SETTINGS
# ==============================
NUM_CLASSES = 6
DATA_ROOT = "data"
VERSION = "v1.0-mini"
MODEL_PATH = "outputs/model.pth"

CLASS_NAMES = {
    0: "background",
    1: "road",
    2: "vehicle",
    3: "pedestrian",
    4: "barrier",
    5: "cone"
}

# ==============================
# mIoU CALCULATION
# ==============================
def calculate_miou(model, dataloader, num_classes):
    model.eval()

    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    total_images = 0

    print("\nRunning mIoU evaluation on all images...")
    print("This may take a few minutes...\n")

    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                pred_cls = (preds == cls)
                mask_cls = (masks == cls)
                intersection[cls] += (pred_cls & mask_cls).sum().float()
                union[cls] += (pred_cls | mask_cls).sum().float()

            total_images += images.shape[0]

    # Calculate IoU per class
    iou_per_class = intersection / (union + 1e-6)
    miou = iou_per_class.mean().item()

    return miou, iou_per_class, total_images


# ==============================
# FPS CALCULATION
# ==============================
def calculate_fps(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)

    # Warmup runs — don't count these
    print("Warming up model...")
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    # Actual timing
    print("Measuring inference speed...")
    num_runs = 50
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(dummy_input)
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_runs / total_time
    ms_per_frame = (total_time / num_runs) * 1000

    return fps, ms_per_frame


# ==============================
# PRINT RESULTS
# ==============================
def print_results(miou, iou_per_class, fps, ms_per_frame, total_images):
    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nEvaluated on: {total_images} images")
    print(f"Image size  : 256x256")
    print(f"Num classes : {NUM_CLASSES}")
    print(f"Device      : CPU")

    print("\n── CLASS-WISE IoU ──────────────────────────────")
    for cls in range(NUM_CLASSES):
        iou = iou_per_class[cls].item()
        bar_filled = int(iou * 30)
        bar_empty = 30 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        print(f"  {CLASS_NAMES[cls]:12} [{bar}]  {iou*100:.2f}%")

    print("\n── KEY METRICS ─────────────────────────────────")
    print(f"  mIoU Score   : {miou*100:.2f}%")
    print(f"  FPS          : {fps:.2f} frames/sec")
    print(f"  Latency      : {ms_per_frame:.2f} ms per frame")

    print("\n── PERFORMANCE GRADE ───────────────────────────")
    if miou >= 0.60:
        grade = "Excellent"
        comment = "Strong segmentation quality"
    elif miou >= 0.45:
        grade = "Good"
        comment = "Decent results, more data would help"
    elif miou >= 0.30:
        grade = "Fair"
        comment = "Model learned basic patterns"
    else:
        grade = "Poor"
        comment = "Training masks were inaccurate — main bottleneck"

    print(f"  mIoU Grade   : {grade} — {comment}")

    if fps >= 30:
        fps_grade = "Real-time capable"
    elif fps >= 15:
        fps_grade = "Near real-time"
    elif fps >= 5:
        fps_grade = "Slow — GPU needed for real-time"
    else:
        fps_grade = "Too slow for real-time on CPU"

    print(f"  FPS Grade    : {fps_grade}")
    print("=" * 50)

    # What judges see
    print("\n── SUBMISSION METRICS (what judges evaluate) ───")
    print(f"  mIoU  = {miou*100:.2f}%")
    print(f"  FPS   = {fps:.2f}")
    print("=" * 50)


# ==============================
# MAIN
# ==============================
def evaluate():
    # Load dataset
    print("Loading dataset...")
    dataset = NuScenesDataset(DATA_ROOT, VERSION)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # Load model
    print("Loading model...")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location='cpu')
    )
    model.eval()
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate mIoU
    miou, iou_per_class, total_images = calculate_miou(
        model, dataloader, NUM_CLASSES
    )

    # Calculate FPS
    fps, ms_per_frame = calculate_fps(model)

    # Print everything
    print_results(
        miou, iou_per_class,
        fps, ms_per_frame,
        total_images
    )


if __name__ == "__main__":
    evaluate()
