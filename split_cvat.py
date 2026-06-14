"""
split_cvat.py
=============
Separate images (.png / .jpg) and labels (.txt) from a CVAT export folder
into two output folders ready for CVAT re-import.

Output structure:
    output/
    ├── images/
    │   ├── frame_000000.png
    │   ├── frame_000030.png
    │   └── ...
    └── labels/
        ├── frame_000000.txt
        ├── frame_000030.txt
        └── ...

Usage:
    python split_cvat.py --dir obj_train_data
    python split_cvat.py --dir obj_train_data --out output
    python split_cvat.py --dir obj_train_data --dry-run
"""

import argparse
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def split(args):
    src_dir = Path(args.dir)
    out_dir = Path(args.out)

    if not src_dir.exists():
        print(f"[ERROR] Directory not found: {src_dir}")
        return

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"

    if not args.dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    # Collect all files
    all_files = sorted(src_dir.rglob("*"))
    images    = [f for f in all_files if f.suffix.lower() in IMAGE_EXTS]
    labels    = [f for f in all_files if f.suffix.lower() == ".txt"]

    # Stats
    paired   = [f for f in images if f.with_suffix(".txt").exists()]
    no_label = [f for f in images if not f.with_suffix(".txt").exists()]
    no_image = [f for f in labels if not any(
                    f.with_suffix(ext).exists() for ext in IMAGE_EXTS)]

    print(f"[INFO] Source      : {src_dir}")
    print(f"[INFO] Output      : {out_dir}")
    print(f"[INFO] Images found: {len(images)}")
    print(f"[INFO]   paired    : {len(paired)}")
    print(f"[INFO]   no label  : {len(no_label)}")
    print(f"[INFO] Labels found: {len(labels)}")
    print(f"[INFO]   no image  : {len(no_image)}")
    if args.dry_run:
        print("[INFO] DRY RUN — no files will be copied")
    print("─" * 60)

    # Copy images
    img_ok = 0
    for f in images:
        dst = img_out / f.name
        if not args.dry_run:
            shutil.copy2(f, dst)
        img_ok += 1

    # Copy labels
    lbl_ok = 0
    for f in labels:
        dst = lbl_out / f.name
        if not args.dry_run:
            shutil.copy2(f, dst)
        lbl_ok += 1

    print(f"[DONE] images → {img_out}  ({img_ok} files)")
    print(f"[DONE] labels → {lbl_out}  ({lbl_ok} files)")

    if no_label:
        print(f"\n[WARN] {len(no_label)} images have no matching label:")
        for f in no_label:
            print(f"         {f.name}")

    if no_image:
        print(f"\n[WARN] {len(no_image)} labels have no matching image:")
        for f in no_image:
            print(f"         {f.name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir",     required=True,
                   help="Path to CVAT export folder (e.g. obj_train_data)")
    p.add_argument("--out",     default="output",
                   help="Output folder (default: output)")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without copying any files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split(args)