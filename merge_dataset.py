"""
merge_datasets.py
=================
Merge two CVAT datasets into one flat directory containing both images and labels.
Hat dataset frames are renamed to seamlessly follow the original dataset frame numbers.

Original dataset : frame_000000 → frame_002670  (step 30)
Hat dataset      : frame_000000 → frame_000600  (step 30)
                   → renamed to frame_002700 → frame_003300 (step 30)

Usage:
    python merge_datasets.py
    python merge_datasets.py --dry-run
"""

import argparse
import shutil
from pathlib import Path


# ========================================
# CONFIG
# ========================================

ORIGINAL_DIR = r"data/obj_train_data"       # original CVAT export folder
HAT_DIR      = r"data/hat_train_data"       # hat CVAT export folder
OUTPUT_DIR   = r"data/merge_train_data"     # merged output folder (รวมไฟล์)

# Only use frames that were manually reviewed
ORIGINAL_LABELED = list(range(0, 2671, 30))   # 0, 30, 60, ... 2670
HAT_LABELED      = list(range(0, 601,  30))   # 0, 30, 60, ... 600

# คำนวณ Offset ให้นับต่อจากไฟล์สุดท้ายของ ORIGINAL
# เช่น ไฟล์สุดท้ายคือ 2670 + step 30 = 2700
HAT_FRAME_OFFSET = max(ORIGINAL_LABELED) + 30


# ========================================
# HELPERS
# ========================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def find_image(folder: Path, stem: str) -> Path | None:
    """Find image file by stem regardless of extension."""
    for ext in IMAGE_EXTS:
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def collect_pairs(src_dir: Path, labeled_frames: list[int],
                  frame_offset: int = 0) -> list[tuple[Path, Path, str]]:
    """
    Return list of (img_path, lbl_path, output_stem) for all valid labeled frames.
    """
    pairs = []
    missing_img = 0
    missing_lbl = 0

    for frame_num in labeled_frames:
        original_stem = f"frame_{frame_num:06d}"
        output_stem   = f"frame_{frame_num + frame_offset:06d}"

        img_path = find_image(src_dir, original_stem)
        lbl_path = src_dir / f"{original_stem}.txt"

        if img_path is None:
            missing_img += 1
            continue
        if not lbl_path.exists():
            missing_lbl += 1
            continue

        pairs.append((img_path, lbl_path, output_stem))

    if missing_img:
        print(f"   ⚠️  {missing_img} missing images in {src_dir.name}")
    if missing_lbl:
        print(f"   ⚠️  {missing_lbl} missing labels in {src_dir.name}")

    return pairs


# ========================================
# MERGE
# ========================================

def merge(args):
    original_dir = Path(ORIGINAL_DIR)
    hat_dir      = Path(HAT_DIR)
    out_dir      = Path(OUTPUT_DIR)

    print("=" * 60)
    print("  Dataset Merge (Flat Directory)")
    print("=" * 60)

    # Validate source folders
    for d, name in [(original_dir, "ORIGINAL_DIR"), (hat_dir, "HAT_DIR")]:
        if not d.exists():
            print(f"[ERROR] {name} not found: {d}")
            return

    # Collect valid pairs from each dataset
    print(f"\n[1/3] Scanning original dataset ({original_dir.name})...")
    original_pairs = collect_pairs(original_dir, ORIGINAL_LABELED, frame_offset=0)
    print(f"   Found {len(original_pairs)} valid frames")

    print(f"\n[2/3] Scanning hat dataset ({hat_dir.name})...")
    hat_pairs = collect_pairs(hat_dir, HAT_LABELED, frame_offset=HAT_FRAME_OFFSET)
    print(f"   Found {len(hat_pairs)} valid frames  "
          f"(renamed frame_XXXXXX → frame_{HAT_FRAME_OFFSET + 0:06d}...)")

    all_pairs = original_pairs + hat_pairs
    print(f"\n   Total combined: {len(all_pairs)} frames  "
          f"({len(original_pairs)} original + {len(hat_pairs)} hat)")

    if args.dry_run:
        print("\n[DRY RUN] No files will be created")
        return

    # Create flat output directory
    print(f"\n[3/3] Building merged dataset → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img_path, lbl_path, output_stem in all_pairs:
        ext = img_path.suffix
        # Save both image and label directly in OUTPUT_DIR
        shutil.copy(img_path, out_dir / f"{output_stem}{ext}")
        shutil.copy(lbl_path, out_dir / f"{output_stem}.txt")

    # Write dataset.yaml (adjusted for flat directory)
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(f"""# Merged Head Detection Dataset (original + hat)
path: {str(out_dir).replace(chr(92), '/')}
train: .  # Flat directory
val: .    # Flat directory

nc: 1
names:
  0: head
""")

    print(f"\n✅ Done → {out_dir} (All images and labels are in this folder)")
    print(f"   dataset.yaml → {yaml_path}")


# ========================================
# MAIN
# ========================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Preview counts without copying any files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args)