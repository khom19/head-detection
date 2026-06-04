# Head Detection

A people-counting system using YOLO11 to detect heads (all orientations) from CCTV footage.
Designed for deployment on Raspberry Pi + Hailo-10H HAT.

---

## File Structure

```
head-detection/
│
├── train_vid/                    # Original CCTV footage (not tracked by git)
│
├── data/
│   └── obj_train_data/           # Labels exported from CVAT (YOLO 1.1 format)
│                                 # (not tracked by git)
│
├── dataset/                      # Auto-generated when running train.py
│   ├── raw_frames/               # Frames extracted from video
│   ├── dataset/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── dataset.yaml
│   └── model/
│       └── head_detection_v1/
│           └── weights/
│               ├── best.pt       # ← use this
│               └── last.pt
│
├── train.py                      # Main training script
├── requirements.txt
├── verify.py                     # Verification script
├── .gitignore
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg (Windows)

```bash
winget install ffmpeg
```

---

## Training

### Edit the config section in train.py

```python
VIDEO_PATH     = r"path/to/your/video"
CVAT_LABELS    = r"path/to/obj_train_data"
OUTPUT_DIR     = r"path/to/output/directory"
MODEL_SIZE     = "yolo11s"
EPOCHS         = 150
BATCH          = 16

# Add more frames as labeling progresses
LABELED_FRAMES = list(range(0, <last_labeled_frame + 1>, 30))
```

### Run

```bash
python train.py
```

The script handles everything automatically:

```
[1/4] Extract frames from video
[2/4] Build dataset structure (80/20 train/val split)
[3/4] Create dataset.yaml
[4/4] Train YOLO11
```

---

## Example Dataset

| | Value |
|--|--|
| Video | 60fps, 1 min 46 sec |
| Frame interval | Every 30 frames (0.5 sec) |
| Total frames | 213 frames |
| Label format | YOLO (cx, cy, w, h) normalized |
| Classes | 1 — `head` |
| Avg heads/frame | ~85 heads |

---

## Labeling Notes

- Labeled using CVAT (cloud) with **Track mode**
- Covers all head orientations: front, side, back
- Interpolated frames between keyframes may have slightly offset boxes
  in segments with high motion — review if needed

---

## Export for Hailo-10H

Convert best.pt → ONNX → HEF after training completes.

```bash
# Step 1: Export to ONNX
yolo export model=best.pt format=onnx opset=11

# Step 2: Compile to HEF (requires Hailo DFC)
hailomz compile \
  --ckpt best.onnx \
  --yaml yolov8s.yaml \
  --classes 1
```

---

## Target Metrics

| Metric | Target |
|--------|--------|
| mAP50 | > 0.70 |
| mAP50-95 | > 0.45 |
| Inference (Pi + Hailo) | < 50ms/frame |
