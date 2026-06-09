"""
Head Detection Training Script

"""

import os
import shutil
import random
from pathlib import Path

# ========================================
# CONFIG
# ========================================

VIDEO_PATH   = r"train_vid/full_vid.mp4"  
CVAT_LABELS  = r"data/obj_train_data"    
OUTPUT_DIR   = r"dataset"            

MODEL_SIZE   = "yolo11s" 
EPOCHS       = 200
IMGSZ        = 640
BATCH        = 16      
VAL_SPLIT    = 0.2          # 20% for validation

# frames that labeled
LABELED_FRAMES = list(range(0, 2671, 30))  # 0, 30, 60, ... 1020

# ========================================
# STEP 1: Extract frames from video
# ========================================

def extract_frames():
    import subprocess

    frames_dir = Path(OUTPUT_DIR) / "raw_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting frames from video...")

    cmd = [
        "ffmpeg", "-i", VIDEO_PATH,
        "-vf", "select=not(mod(n\\,30))",
        "-vsync", "vfr",
        "-frame_pts", "1",
        "-q:v", "2",
        str(frames_dir / "frame_%06d.jpg")
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        return False

    extracted = len(list(frames_dir.glob("*.jpg")))
    print(f"   ✅ Extracted {extracted} frames → {frames_dir}")
    return True


# ========================================
# STEP 2: Create Dataset structure
# ========================================

def build_dataset():
    print(f"\n[2/4] Building dataset structure...")

    raw_frames = Path(OUTPUT_DIR) / "raw_frames"
    dataset    = Path(OUTPUT_DIR) / "dataset"
    labels_src = Path(CVAT_LABELS)

    # create folder
    for split in ["train", "val"]:
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    # labeled frame
    available = []
    for frame_num in LABELED_FRAMES:
        frame_name = f"frame_{frame_num:06d}"
        img_path   = raw_frames / f"{frame_name}.jpg"
        lbl_path   = labels_src / f"{frame_name}.txt"

        if img_path.exists() and lbl_path.exists():
            available.append(frame_name)
        else:
            if not img_path.exists():
                print(f"   ⚠️  Missing image: {frame_name}.jpg")
            if not lbl_path.exists():
                print(f"   ⚠️  Missing label: {frame_name}.txt")

    print(f"   Found {len(available)} labeled frames")

    # Split train/val
    random.seed(42)
    random.shuffle(available)
    split_idx  = int(len(available) * (1 - VAL_SPLIT))
    train_set  = available[:split_idx]
    val_set    = available[split_idx:]

    print(f"   Train: {len(train_set)} | Val: {len(val_set)}")

    # Copy files
    for frame_name in train_set:
        shutil.copy(raw_frames / f"{frame_name}.jpg",
                    dataset / "images" / "train" / f"{frame_name}.jpg")
        shutil.copy(labels_src / f"{frame_name}.txt",
                    dataset / "labels" / "train" / f"{frame_name}.txt")

    for frame_name in val_set:
        shutil.copy(raw_frames / f"{frame_name}.jpg",
                    dataset / "images" / "val" / f"{frame_name}.jpg")
        shutil.copy(labels_src / f"{frame_name}.txt",
                    dataset / "labels" / "val" / f"{frame_name}.txt")

    print(f"   ✅ Dataset ready → {dataset}")
    return dataset


# ========================================
# STEP 3: Create dataset.yaml
# ========================================

def create_yaml(dataset_path):
    print(f"\n[3/4] Creating dataset.yaml...")

    yaml_content = f"""# Head Detection Dataset - The-Beat
path: {str(dataset_path).replace(chr(92), '/')}
train: images/train
val: images/val

nc: 1
names:
  0: head
"""
    yaml_path = dataset_path / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"   ✅ Saved → {yaml_path}")
    return yaml_path


# ========================================
# STEP 4: Train YOLOv11
# ========================================

def train(yaml_path):
    print(f"\n[4/4] Starting training...")
    print(f"   Model : {MODEL_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch : {BATCH}")
    print(f"   Size  : {IMGSZ}px\n")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Not found ultralytics:")
        print("   pip install ultralytics")
        return

    model = YOLO(f"{MODEL_SIZE}.pt")

    results = model.train(
        data    = str(yaml_path),
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        name    = "head_detection_v1",
        project = str(Path(OUTPUT_DIR) / "model"),
        lr0           = 0.01,
        warmup_epochs = 10,
        close_mosaic = 20,

        # Augmentation
        hsv_h   = 0.5,
        hsv_s   = 0.5,
        hsv_v   = 0.0,
        fliplr  = 0.5,
        mosaic  = 1.0,
        degrees = 0.0,
        erasing = 0.0,
        auto_augment = "randaugment"
    )

    print(f"\n✅ Training complete!")
    print(f"   Best weights: {Path(OUTPUT_DIR)}/model/head_detection_v1/weights/best.pt")
    print(f"\n📦 Export for Hailo-10H:")
    print(f"   yolo export model=best.pt format=onnx opset=11")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    print("=" * 50)
    print("  Head Detection Training")
    print("=" * 50)

    # ติดตั้ง dependency
    os.system("pip install ultralytics -q")

    if not extract_frames():
        print("❌ Frame extraction failed")
        exit(1)

    dataset_path = build_dataset()
    yaml_path    = create_yaml(dataset_path)
    train(yaml_path)