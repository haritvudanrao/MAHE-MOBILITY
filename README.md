# Real-Time Drivable Space Segmentation

A semantic segmentation system for autonomous driving built entirely from scratch using U-Net. The model labels every pixel in a front-facing camera image — identifying road, vehicles, pedestrians, barriers, and traffic cones — in real time.

---

## Project Overview

Autonomous vehicles need to understand what is drivable and what is not, at every single frame. This project tackles that problem using **semantic segmentation** — classifying every pixel of a front-camera image into one of six categories:

| Class | Color in Output |
|---|---|
| Background | Original image (no overlay) |
| Road | Light green wash (subtle tint) |
| Vehicle | Red |
| Pedestrian | Deep blue |
| Barrier | Orange |
| Traffic Cone | Yellow |

The system is built for **Level 4 autonomous vehicles** and is optimized for real-time inference on CPU.

---

## Dataset

We use **nuScenes v1.0-mini** — an autonomous driving dataset with 404 front-camera keyframe images captured across four locations:
- Singapore Onenorth
- Singapore Queenstown
- Singapore Holland Village
- Boston Seaport

### Important: nuScenes does not include segmentation masks

We generated our own ground truth masks using two sources built into the nuScenes toolkit:

**1. Map API Projection**
Road polygon data from the nuScenes Map Expansion API is projected from world coordinates onto the camera image using the car's GPS ego-pose and the camera's intrinsic calibration matrix — producing a pixel-accurate road mask.

**2. 3D Bounding Box Projection**
3D bounding box annotations for every object are transformed from global → ego → camera frame using quaternion rotation, then projected onto the image to generate masks for vehicles, pedestrians, barriers, and cones.

**Priority ordering for overlapping classes:** cone > barrier > pedestrian > vehicle > road > background

### Download the Dataset

The nuScenes v1.0-mini dataset can be downloaded from the link below. After downloading, extract it and place it inside the `data/` folder before running the code.

📁 **[Download nuScenes v1.0-mini from Google Drive](https://drive.google.com/your-link-here)**

---

## Model Architecture

We implemented **U-Net from scratch** — no pretrained weights, fully compliant with hackathon rules.

```
Input Image (3 × 256 × 256)
        │
        ▼
┌─────────────────────┐
│      ENCODER        │
│  3  → 64  → MaxPool │
│  64 → 128 → MaxPool │
│  128→ 256 → MaxPool │
│  256→ 512 → MaxPool │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│     BOTTLENECK      │
│     512 → 1024      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│      DECODER        │
│  ConvTranspose2D    │
│  + Skip Connections │
│  1024→512→256→128→64│
└─────────────────────┘
        │
        ▼
  1×1 Conv → 6 class scores per pixel
```

Each encoder stage uses a **DoubleConv block**: two Conv2D layers each followed by BatchNorm and ReLU. Skip connections from the encoder are concatenated with decoder feature maps at matching resolutions to preserve fine spatial detail.

**Total trainable parameters: ~31 Million**

---

## Loss Function

Training uses a **combined loss** defined in `model/loss.py`:

```
Loss = CrossEntropy Loss + Dice Loss
```

- **CrossEntropy Loss** penalizes wrong class predictions at the pixel level
- **Dice Loss** measures overlap between predicted and ground truth masks per class, ensuring the model pays attention to rare classes like cones rather than ignoring them in favour of the dominant road class

---

## Training Strategy

| Setting | Value |
|---|---|
| Image size | 256 × 256 |
| Normalization | [0, 255] → [0, 1] |
| Train / Val split | 80% / 20% (323 / 81 images) |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 2 |
| Epochs | 50 |

The best model (lowest validation loss) is automatically saved to `outputs/model.pth` during training.

---

## Inference & Post-Processing

At inference time (`predict.py`) the model is run on new nuScenes images. To improve output quality without retraining, a multi-step post-processing pipeline is applied:

**Step 1 — Confidence thresholding**
Any pixel where the model's softmax confidence is below 20% is set to background, removing the weakest predictions across all classes.

**Step 2 — Noise removal**
Tiny detected blobs are removed. Road requires a larger minimum blob size (800 pixels) since small scattered green patches are almost always noise. All other classes use a 200 pixel minimum.

**Step 3 — Smart filtering (aspect ratio + color + size)**
Each detected blob is validated against real-world physical rules:
- *Pedestrian*: must be taller than wide, must not have green-dominant pixels underneath (removes trees), must not span more than 30% of image width
- *Vehicle*: must not be extremely tall and thin; if orange-ish pixels underneath → reclassified as barrier automatically
- *Barrier*: must be wider than tall (horizontal objects), width must be at least 1.5× the height
- *Cone*: must not exceed 10% of total image area, must not be very wide relative to height
- *Road*: blob must not be entirely in the top 45% of the image (road cannot appear in the sky)

**Step 4 — Road region merging**
All surviving road blobs are merged into a single continuous filled region. The top boundary of the road is computed per column — requiring a minimum density of road pixels per column before accepting it — and then smoothed with a Gaussian. The boundary is clamped so road never appears above 40% of image height. This produces a clean, continuous road region rather than scattered patches.

**Step 5 — Rectangular block visualization**
Each detected region is converted to its bounding box and filled as a **solid 2D rectangular block**. Road is drawn first as the base layer; objects (barrier → vehicle → pedestrian → cone) are drawn on top in priority order so higher-priority classes always win overlapping regions.

**Step 6 — Layered colored overlay**
Each class is blended onto the original image with its own opacity level:

| Class | Overlay opacity | Effect |
|---|---|---|
| Road | 25% | Subtle light green wash — road texture stays visible |
| Vehicle | 65% | Bold red block |
| Pedestrian | 65% | Bold deep blue block |
| Barrier | 65% | Bold orange block |
| Cone | 65% | Bold yellow block |
| Background | 0% | Original image, completely untouched |

Road is painted first at low opacity so it reads as a background region. All object classes are then painted on top at high opacity so they clearly stand out.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** (primary) | Mean Intersection over Union across all 6 classes |
| **FPS** (secondary) | Frames per second at inference time on CPU |

Evaluation is run via `evaluate.py`.

---
Sample Outputs
The images below show real outputs produced by our pipeline on nuScenes front-camera frames. Each result displays the original image alongside the segmentation overlay — with road marked as a light green wash and all detected objects (vehicles, pedestrians, barriers, cones) shown as bold colored rectangular blocks on top.
📂 https://drive.google.com/drive/folders/18e0ZAaIGKiT_lBPS1lcEVlRQ7yXdo0OB?usp=drive_link

If the output images in outputs/predictions/ do not open locally, use the Google Drive link above as an alternative to view the same results.

## Project Structure

```
├── data/                        ← Place nuScenes dataset here
├── model/
│   ├── unet.py                  ← U-Net architecture
│   └── loss.py                  ← Combined CrossEntropy + Dice loss
├── utils/
│   └── nuscenes_dataset.py      ← Dataset loader + ground truth generation
├── outputs/
│   ├── model.pth                ← Saved best model
│   └── predictions/             ← Output images saved here
├── train.py                     ← Training script
├── predict.py                   ← Inference + post-processing
├── evaluate.py                  ← mIoU + FPS evaluation
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/drivable-space-segmentation.git
cd drivable-space-segmentation
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download and set up the dataset**

Download the nuScenes v1.0-mini dataset from the Google Drive link above and place it inside the `data/` folder so the structure looks like this:

```
data/
└── v1.0-mini/
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── *.json
```

---

## How to Run

**Train the model**
```bash
python train.py
```
The best model will be saved to `outputs/model.pth` automatically.

**Run inference**
```bash
python predict.py
```
Output images are saved to `outputs/predictions/`. Each output shows the original image alongside the segmentation overlay with colored rectangular blocks.

**Evaluate the model**
```bash
python evaluate.py
```
Prints mIoU per class and overall FPS on CPU.

---

## Key Highlights

- Built **entirely from scratch** — no pretrained weights used
- Custom **ground truth generation pipeline** using nuScenes Map API and 3D bounding box projection
- **Combined loss function** handles class imbalance between dominant (road) and rare (cone) classes
- **Smart post-processing** with aspect ratio, color, and size validation filters false detections without retraining
- **Road region merging** produces a single clean continuous road region instead of scattered patches
- **Rectangular 2D block visualization** for clean, interpretable segmentation output
- **Per-class opacity layering** — road shown as a subtle background wash, objects shown as bold vivid blocks on top
- Designed and tested for **real-time CPU inference**

---

## Team

- V. Sirishree
- Vudanrao Harit
- Aarush Rawat
- Iha Rawka

---

## License

This project is for academic and hackathon purposes only.
