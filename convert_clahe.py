"""
convert_clahe.py
================
Apply CLAHE preprocessing to images or video frames.
Converts images in-place — label .txt files are not touched.

CVAT export structure expected:
    obj_train_data/
    ├── frame_000000.jpg   ← image
    ├── frame_000000.txt   ← label (untouched)
    ├── frame_000030.jpg
    ├── frame_000030.txt
    └── ...

Usage (images folder):
    python convert_clahe.py --dir obj_train_data
    python convert_clahe.py --dir obj_train_data --clip-limit 5.0 --tile-size 8
    python convert_clahe.py --dir obj_train_data --dry-run

Usage (video):
    python convert_clahe.py --video cctv.mp4 --out frames/
    python convert_clahe.py --video cctv.mp4 --out frames/ --every 30
    python convert_clahe.py --video cctv.mp4 --out frames/ --clip-limit 5.0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ── Default CLAHE params — must match train.py and count-people.py ──
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_SIZE  = 8


def apply_clahe(img: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    """
    Convert image to grayscale, apply CLAHE, return as 3-channel grayscale.
    3-channel output keeps YOLO input shape (H, W, 3) unchanged.
    Works on both color and grayscale input images.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img if len(img.shape) == 2 else img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eq = clahe.apply(gray)
    return cv2.merge([eq, eq, eq])


def find_images(directory: Path) -> list[Path]:
    """Recursively find all jpg/png images under directory."""
    exts = ("*.jpg", "*.jpeg", "*.png")
    imgs = []
    for ext in exts:
        imgs.extend(directory.rglob(ext))
    return sorted(imgs)


# ─────────────────────── mode: folder ──────────────────────────

def convert_dir(args):
    """Convert all images in a CVAT export folder in-place."""
    src_dir = Path(args.dir)
    if not src_dir.exists():
        print(f"[ERROR] Directory not found: {src_dir}")
        return

    imgs = find_images(src_dir)
    if not imgs:
        print(f"[ERROR] No images found in: {src_dir}")
        return

    labeled   = [p for p in imgs if p.with_suffix(".txt").exists()]
    unlabeled = [p for p in imgs if not p.with_suffix(".txt").exists()]

    print(f"[INFO] Directory   : {src_dir}")
    print(f"[INFO] Images found: {len(imgs)}")
    print(f"[INFO]   with label: {len(labeled)}")
    print(f"[INFO]   no label  : {len(unlabeled)}  (will still be converted)")
    print(f"[INFO] CLAHE params: clip_limit={args.clip_limit}  "
          f"tile={args.tile_size}x{args.tile_size}")
    if args.dry_run:
        print("[INFO] DRY RUN — no files will be changed")
    print("─" * 60)

    ok     = 0
    failed = 0

    for i, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read: {img_path.name}")
            failed += 1
            continue

        original_shape = img.shape
        result         = apply_clahe(img, args.clip_limit, args.tile_size)

        if not args.dry_run:
            cv2.imwrite(str(img_path), result)

        ok += 1

        if i % 50 == 0 or i == len(imgs):
            print(f"  [{i:4d}/{len(imgs)}]  {img_path.name}  "
                  f"{original_shape[1]}x{original_shape[0]}  "
                  f"{'(skipped)' if args.dry_run else 'converted'}")

    print("─" * 60)
    if args.dry_run:
        print(f"[DRY RUN] Would convert {ok} images  ({failed} unreadable)")
    else:
        print(f"[DONE] Converted {ok} images  ({failed} failed)")
        print(f"[NOTE] Label .txt files were not modified")
        print(f"[NOTE] Use the same clip_limit={args.clip_limit} "
              f"tile_size={args.tile_size} in train.py and count-people.py")


# ─────────────────────── mode: video ───────────────────────────

def convert_video(args):
    """
    Extract frames from a video, apply CLAHE, and save as jpg files.
    Output filename format: frame_XXXXXX.jpg (zero-padded, matches CVAT naming).

    --every N  saves every N-th frame (default 1 = every frame).
               Use 30 to match train.py's extract_frames() interval.
    """
    video_path = Path(args.video)
    out_dir    = Path(args.out)

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    every        = args.every
    expected     = len(range(0, total_frames, every))

    print(f"[INFO] Video       : {video_path.name}")
    print(f"[INFO] Resolution  : {width}x{height}  {fps:.0f}fps  {total_frames} frames")
    print(f"[INFO] Save every  : {every} frame(s)  (~{expected} images)")
    print(f"[INFO] Output dir  : {out_dir}")
    print(f"[INFO] CLAHE params: clip_limit={args.clip_limit}  "
          f"tile={args.tile_size}x{args.tile_size}")
    if args.dry_run:
        print("[INFO] DRY RUN — no files will be written")
    print("─" * 60)

    frame_idx = 0
    saved     = 0
    failed    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            result   = apply_clahe(frame, args.clip_limit, args.tile_size)
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"

            if not args.dry_run:
                ok = cv2.imwrite(str(out_path), result)
                if not ok:
                    failed += 1
                else:
                    saved += 1
            else:
                saved += 1

            if saved % 50 == 0 or saved == 1:
                pct = frame_idx / max(total_frames, 1) * 100
                print(f"  [{pct:5.1f}%] frame {frame_idx:6d} → {out_path.name}  "
                      f"{'(dry run)' if args.dry_run else 'saved'}")

        frame_idx += 1

    cap.release()
    print("─" * 60)
    if args.dry_run:
        print(f"[DRY RUN] Would save {saved} frames to {out_dir}")
    else:
        print(f"[DONE] Saved {saved} frames  ({failed} failed) → {out_dir}")
        print(f"[NOTE] Use the same clip_limit={args.clip_limit} "
              f"tile_size={args.tile_size} in train.py and count-people.py")


# ─────────────────────── entry point ───────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir",   help="Path to CVAT export folder (images + labels)")
    src.add_argument("--video", help="Path to video file to extract frames from")

    # Video-only options
    p.add_argument("--out",   default="frames",
                   help="Output folder for extracted video frames (default: frames/)")
    p.add_argument("--every", default=1, type=int,
                   help="Save every N-th frame from video (default 1). "
                        "Use 30 to match train.py extract_frames() interval.")

    # Shared options
    p.add_argument("--clip-limit", default=DEFAULT_CLIP_LIMIT, type=float,
                   help=f"CLAHE clip limit (default {DEFAULT_CLIP_LIMIT})")
    p.add_argument("--tile-size",  default=DEFAULT_TILE_SIZE,  type=int,
                   help=f"CLAHE tile grid size (default {DEFAULT_TILE_SIZE})")
    p.add_argument("--dry-run",    action="store_true",
                   help="Preview without making any changes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.video:
        convert_video(args)
    else:
        convert_dir(args)