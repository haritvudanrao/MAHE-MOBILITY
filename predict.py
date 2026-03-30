
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
from model.unet import UNet
from nuscenes.nuscenes import NuScenes

# ==============================
# SETTINGS
# ==============================
MODEL_PATH = "outputs/model.pth"
OUTPUT_DIR = "outputs/predictions"
NUM_CLASSES = 6
DATA_ROOT = "data"
VERSION = "v1.0-mini"

# Class colors (RGB)
CLASS_COLORS = {
    1: [0, 255, 0],       # road       → green
    2: [255, 0, 0],       # vehicle    → red
    3: [0, 100, 255],     # pedestrian → blue (slightly deeper for contrast)
    4: [255, 140, 0],     # barrier    → orange
    5: [255, 220, 0],     # cone       → yellow
}

# Alpha (opacity) per class:
#   Road gets a very light tint (0.25) so it's subtle & background-like.
#   All other classes get a stronger overlay (0.60) so they stand out on top.
CLASS_ALPHA = {
    1: 0.25,   # road      — light wash, shows texture underneath
    2: 0.65,   # vehicle   — bold
    3: 0.65,   # pedestrian— bold
    4: 0.65,   # barrier   — bold
    5: 0.65,   # cone      — bold
}

# Draw order: road first (background tint), then objects on top
DRAW_ORDER = [1, 4, 2, 3, 5]   # road → barrier → vehicle → pedestrian → cone


# ==============================
# HELPER FUNCTIONS
# ==============================

def clean_predictions(mask, min_area=200):
    cleaned = mask.copy()
    for cls in range(1, 6):
        # ── KEY FIX 4: Road needs much larger blobs to be kept ──────────
        area_threshold = 800 if cls == 1 else min_area
        binary = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < area_threshold:
                cleaned[labels == label] = 0
    return cleaned


def merge_road_to_single_block(mask):
    """
    Fixed version: only fills road in the BOTTOM 50% of the image,
    and requires a minimum density of road pixels per column before
    filling it — prevents spurious green in sky/tree/building columns.
    """
    img_h, img_w = mask.shape

    # Save non-road classes before modifying
    non_road_mask = mask.copy()
    non_road_mask[non_road_mask == 1] = 0

    # ── KEY FIX 1: Only look at bottom 50% (was 75%) ────────────────────
    cutoff = int(img_h * 0.50)
    road_binary = np.zeros((img_h, img_w), dtype=np.uint8)
    road_binary[cutoff:, :] = (mask[cutoff:, :] == 1).astype(np.uint8)

    if road_binary.sum() < 300:
        return mask.copy()

    # ── Step 1: Find topmost road pixel per column ───────────────────────
    top_boundary = np.full(img_w, img_h, dtype=np.float32)
    MIN_ROAD_PIXELS_PER_COL = 3   # ── KEY FIX 2: require density

    for col in range(img_w):
        road_rows = np.where(road_binary[:, col] == 1)[0]
        if len(road_rows) >= MIN_ROAD_PIXELS_PER_COL:
            top_boundary[col] = road_rows[0]

    # ── Step 2: Smooth boundary ──────────────────────────────────────────
    has_road = top_boundary < img_h
    if has_road.sum() > 10:
        cols_with_road = np.where(has_road)[0]
        cols_without = np.where(~has_road)[0]
        if len(cols_without) > 0:
            top_boundary[cols_without] = np.interp(
                cols_without, cols_with_road, top_boundary[cols_with_road]
            )
        top_boundary = cv2.GaussianBlur(
            top_boundary.reshape(1, -1), (51, 1), sigmaX=15
        ).flatten()

    # ── KEY FIX 3: Clamp boundary — never go above 40% of image height ──
    top_boundary = np.clip(top_boundary, img_h * 0.40, img_h)

    # ── Step 3: Fill downward from boundary ─────────────────────────────
    road_layer = np.zeros((img_h, img_w), dtype=np.int32)
    for col in range(img_w):
        start_row = int(top_boundary[col])
        if start_row < img_h:
            road_layer[start_row:, col] = 1

    # ── Step 4: Restore non-road classes on top ─────────────────────────
    merged = road_layer.astype(mask.dtype)
    merged[non_road_mask > 0] = non_road_mask[non_road_mask > 0]

    return merged


def smart_filter(mask, original_img_array):
    """
    Three filters applied per detected blob:

    1. ASPECT RATIO filter
       - Pedestrian (3): must be taller than wide (height > width)
       - Vehicle (2): must not be extremely tall and thin (h < w*2)
       - Barrier (4): must be wider than tall (w > h)

    2. COLOR-BASED filter
       - Pedestrian (3): green-dominant pixels underneath = tree, remove
       - Vehicle (2): orange-ish pixels underneath = barrier, reclassify

    3. RELATIVE SIZE filter
       - Pedestrian (3): can't span more than 30% of image width
       - Vehicle (2): must be at least 15x15 pixels
       - Barrier (4): width must be at least 1.5x height
       - Cone (5): can't be more than 10% of total image area
    """
    filtered = mask.copy()
    img_h, img_w = mask.shape

    if original_img_array.max() <= 1.0:
        orig = (original_img_array * 255).astype(np.uint8)
    else:
        orig = original_img_array.astype(np.uint8)

    for cls in range(1, 6):
        binary = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            blob_pixels = orig[labels == label]
            avg_r = blob_pixels[:, 0].mean()
            avg_g = blob_pixels[:, 1].mean()
            avg_b = blob_pixels[:, 2].mean()

            remove = False

            if cls == 3:   # PEDESTRIAN
                if w >= h:
                    remove = True
                if avg_g > avg_r + 20 and avg_g > avg_b + 20:
                    remove = True
                if w > img_w * 0.30:
                    remove = True

            elif cls == 2:   # VEHICLE
                if h > w * 2.0:
                    remove = True
                if avg_r > 160 and avg_g > 80 and avg_b < 80:
                    filtered[labels == label] = 4
                    continue
                if w < 15 or h < 15:
                    remove = True

            elif cls == 4:   # BARRIER
                if w < h:
                    remove = True
                if avg_g > avg_r + 25 and avg_g > avg_b + 25:
                    remove = True
                if w < h * 1.5:
                    remove = True

            elif cls == 5:   # CONE
                if area > img_h * img_w * 0.10:
                    remove = True
                if w > h * 2.5:
                    remove = True

            elif cls == 1:   # ROAD

                if w < 20:
                    remove = True
                if (y + h) < img_h * 0.45:
                    remove = True

            if remove:
                filtered[labels == label] = 0

    return filtered


def convert_to_rectangles(mask):
    """
    For each detected region per class, find its bounding box
    and fill it as a solid rectangle for clean rectangular blocks.
    Priority order: road first, then objects on top.
    """
    rect_mask = np.zeros_like(mask)
    priority_order = [1, 2, 3, 4, 5]
    for cls in priority_order:
        binary = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            rect_mask[y:y+h, x:x+w] = cls
    return rect_mask


def create_overlay(original_img, mask):
    """
    Layered overlay:
      - Road (class 1) is painted first with a LOW alpha (light green wash).
      - All other classes are painted ON TOP with HIGH alpha (vivid colours).
    This means a pedestrian standing on the road shows:
      light green road tint beneath + strong blue person block on top.
    """
    img_array = np.array(original_img).astype(np.float32)
    overlay = img_array.copy()

    for cls in DRAW_ORDER:
        color = np.array(CLASS_COLORS[cls], dtype=np.float32)
        alpha = CLASS_ALPHA[cls]
        pixels = mask == cls
        if pixels.sum() == 0:
            continue
        # Blend: overlay = (1 - alpha) * original + alpha * class_color
        overlay[pixels] = (1 - alpha) * img_array[pixels] + alpha * color

    return overlay.clip(0, 255).astype(np.uint8)


# ==============================
# MAIN PREDICT FUNCTION
# ==============================

def predict():
    print("Loading model...")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location='cpu')
    )
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading nuScenes...")
    nusc = NuScenes(
        version=VERSION,
        dataroot=DATA_ROOT,
        verbose=False
    )

    samples_to_predict = []
    for sample in nusc.sample[:5]:
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join(DATA_ROOT, cam_data['filename'])
        if os.path.exists(img_path):
            samples_to_predict.append(img_path)

    print(f"Predicting on {len(samples_to_predict)} images...")

    for img_path in samples_to_predict:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((256, 256))

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            max_probs, pred_mask = torch.max(probs, dim=1)
            pred_mask = pred_mask.squeeze().numpy()
            max_probs = max_probs.squeeze().numpy()
            pred_mask[max_probs < 0.2] = 0

        # Step 1: Remove tiny noise blobs
        pred_mask = clean_predictions(pred_mask, min_area=200)

        # Step 2: Smart filtering — aspect ratio + color + size
        pred_mask = smart_filter(pred_mask, img_array)

        # Step 3: Clean again after smart filtering
        pred_mask = clean_predictions(pred_mask, min_area=200)

        # Step 4: Merge all road blobs into one continuous filled region
        pred_mask = merge_road_to_single_block(pred_mask)

        # Step 5: Convert to solid rectangular blocks
        pred_mask = convert_to_rectangles(pred_mask)

        unique_classes = np.unique(pred_mask)
        class_names = {
            0: 'background', 1: 'road', 2: 'vehicle',
            3: 'pedestrian', 4: 'barrier', 5: 'cone'
        }
        detected = [class_names[c] for c in unique_classes]
        print(f"  [{os.path.basename(img_path)}] Detected: {detected}")

        overlay = create_overlay(image, pred_mask)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(
            "Segmentation\n"
            "Light Green=Road | Red=Vehicle | Blue=Person | "
            "Orange=Barrier | Yellow=Cone",
            fontsize=11
        )
        axes[1].axis("off")

        plt.tight_layout()
        img_name = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved → {save_path}")

    print(f"\nDone! Check {OUTPUT_DIR}/")


if __name__ == "__main__":
    predict()