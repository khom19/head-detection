"""
merge_dataset.py
================
Append a newly-labeled camera into the existing accumulated head-detection
dataset (YOLO format, 1 class: head).

Layout convention (kept from the previous version):
    * flat directory  -> images and .txt labels live side by side
    * file naming      -> frame_%06d  (keeps the source image extension)

This job only MERGES.  It does not split train/val and does not write
data.yaml — train.py handles that.

Sources
-------
    BASE_DIR    merge_train_data/      existing dataset (111 frames, img+txt flat)
                                       also the OUTPUT: new frames are appended here
    NEW_IMG_DIR frames/                new camera images  (.jpg, no labels)
    NEW_LBL_DIR data/obj_train_data/   new camera labels  (.txt, no images, CVAT export)

Numbering
---------
    The highest frame number in BASE_DIR is found by scanning.  New-camera
    frames are renamed to continue from it in FRAME_STEP increments
    (e.g. base ends at 3300 -> new camera = 3330, 3360, ... ).

Usage
-----
    python merge_dataset.py --dry-run     # validate + preview, write nothing
    python merge_dataset.py               # prompt, then append into BASE_DIR
    python merge_dataset.py --yes         # skip the confirmation prompt
"""

import argparse
import re
import shutil
from pathlib import Path


# ========================================
# CONFIG
# ========================================

BASE_DIR    = r"merge_train_data"       # existing dataset + append target (flat img+txt)
NEW_IMG_DIR = r"frames"                 # new camera images
NEW_LBL_DIR = r"data/obj_train_data"    # new camera labels (CVAT export)

FRAME_STEP  = 30                        # numbering step (matches existing convention)

IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}
FRAME_RE    = re.compile(r"frame_(\d+)$")


# ========================================
# HELPERS
# ========================================

def frame_num(stem: str) -> int | None:
    """Extract the integer frame number from a 'frame_000123' style stem."""
    m = FRAME_RE.match(stem)
    return int(m.group(1)) if m else None


def find_image(folder: Path, stem: str) -> Path | None:
    """Find an image file by stem regardless of extension."""
    for ext in IMAGE_EXTS:
        p = folder / (stem + ext)
        if p.exists():
            return p
    return None


def validate_label(lbl_path: Path) -> tuple[int, list[str]]:
    """
    Validate a YOLO label file.

    Returns (n_instances, errors).  Each bbox line must be
        class_id cx cy w h
    with class_id == 0 and every coordinate in [0, 1].
    """
    errors: list[str] = []
    n = 0
    for i, raw in enumerate(lbl_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 fields, got {len(parts)}")
            continue
        cls, coords = parts[0], parts[1:]
        try:
            if int(cls) != 0:
                errors.append(f"line {i}: class_id != 0 ({cls})")
        except ValueError:
            errors.append(f"line {i}: class_id not an int ({cls!r})")
        try:
            vals = [float(x) for x in coords]
            if any(not (0.0 <= v <= 1.0) for v in vals):
                errors.append(f"line {i}: coord out of [0,1] -> {vals}")
        except ValueError:
            errors.append(f"line {i}: non-numeric coords -> {coords}")
        n += 1
    return n, errors


def summarize_range(nums: list[int]) -> str:
    """'frame_000000 .. frame_003300' for a list of frame numbers."""
    if not nums:
        return "(none)"
    return f"frame_{min(nums):06d} .. frame_{max(nums):06d}"


# ========================================
# SCAN
# ========================================

def scan_base(base_dir: Path):
    """
    Scan the existing dataset (flat img+txt).  Validates img<->label pairing
    and label contents, but never modifies anything here.

    Returns dict with: nums, png_nums, jpg_nums, instances, max_num, warnings.
    """
    nums, png_nums, jpg_nums = [], [], []
    instances = 0
    warnings: list[str] = []

    images = [p for p in base_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    for img in sorted(images):
        fn = frame_num(img.stem)
        if fn is None:
            warnings.append(f"unparseable image name: {img.name}")
            continue
        lbl = base_dir / f"{img.stem}.txt"
        if not lbl.exists():
            warnings.append(f"image without label: {img.name}")
            continue
        n, errs = validate_label(lbl)
        for e in errs:
            warnings.append(f"{lbl.name}: {e}")
        nums.append(fn)
        instances += n
        if img.suffix.lower() == ".png":
            png_nums.append(fn)
        else:
            jpg_nums.append(fn)

    return {
        "nums": nums,
        "png_nums": png_nums,
        "jpg_nums": jpg_nums,
        "instances": instances,
        "max_num": max(nums) if nums else -FRAME_STEP,
        "warnings": warnings,
    }


def scan_new(img_dir: Path, lbl_dir: Path):
    """
    Pair new-camera images (img_dir) with labels (lbl_dir) by stem.

    Returns (pairs, skipped, warnings) where pairs is a list of
    (src_num, img_path, lbl_path, n_instances) sorted by src_num.
    Unmatched or invalid frames are reported and skipped.
    """
    pairs: list[tuple[int, Path, Path, int]] = []
    skipped: list[str] = []
    warnings: list[str] = []

    images = [p for p in img_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    image_stems = {p.stem for p in images}

    # labels present without a matching image
    for lbl in sorted(lbl_dir.glob("*.txt")):
        if lbl.stem not in image_stems:
            skipped.append(f"label without image: {lbl.name}")

    for img in sorted(images):
        fn = frame_num(img.stem)
        if fn is None:
            skipped.append(f"unparseable image name: {img.name}")
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            skipped.append(f"image without label: {img.name}")
            continue
        n, errs = validate_label(lbl)
        if errs:
            skipped.append(f"{img.name}: invalid label -> " + "; ".join(errs))
            continue
        pairs.append((fn, img, lbl, n))

    pairs.sort(key=lambda t: t[0])
    return pairs, skipped, warnings


# ========================================
# MERGE
# ========================================

def merge(args):
    base_dir = Path(BASE_DIR)
    img_dir  = Path(NEW_IMG_DIR)
    lbl_dir  = Path(NEW_LBL_DIR)

    print("=" * 64)
    print("  Append new camera into existing dataset (flat directory)")
    print("=" * 64)

    for d, name in [(base_dir, "BASE_DIR"), (img_dir, "NEW_IMG_DIR"),
                    (lbl_dir, "NEW_LBL_DIR")]:
        if not d.exists():
            print(f"[ERROR] {name} not found: {d}")
            return

    # --- scan existing dataset ---------------------------------------------
    print(f"\n[1/3] Scanning base dataset ({base_dir})...")
    base = scan_base(base_dir)
    base_max = base["max_num"]
    print(f"   {len(base['nums'])} frames, {base['instances']} instances")
    print(f"   range: {summarize_range(base['nums'])}")
    print(f"     .png (cam1): {len(base['png_nums'])} frames  "
          f"{summarize_range(base['png_nums'])}")
    print(f"     .jpg (cam2): {len(base['jpg_nums'])} frames  "
          f"{summarize_range(base['jpg_nums'])}")
    for w in base["warnings"]:
        print(f"   [WARN] {w}")

    # --- scan + validate new camera ----------------------------------------
    print(f"\n[2/3] Scanning new camera "
          f"(img={img_dir}, lbl={lbl_dir})...")
    pairs, skipped, _ = scan_new(img_dir, lbl_dir)
    for s in skipped:
        print(f"   [SKIP] {s}")
    new_instances = sum(n for *_, n in pairs)
    print(f"   {len(pairs)} valid frames, {new_instances} instances "
          f"({len(skipped)} skipped)")

    if not pairs:
        print("\n[ERROR] No valid new frames to merge.")
        return

    # --- assign continued frame numbers (old convention: step offset) ------
    plan = []          # (img_path, lbl_path, out_num, out_stem, ext)
    collisions = []
    for rank, (src_num, img, lbl, _n) in enumerate(pairs, start=1):
        out_num  = base_max + FRAME_STEP * rank
        out_stem = f"frame_{out_num:06d}"
        ext      = img.suffix
        if (base_dir / f"{out_stem}{ext}").exists() or \
           (base_dir / f"{out_stem}.txt").exists():
            collisions.append(out_stem)
        plan.append((img, lbl, out_num, out_stem, ext))

    new_nums = [p[2] for p in plan]

    if collisions:
        print(f"\n[ERROR] {len(collisions)} target name(s) already exist in "
              f"{base_dir} (e.g. {collisions[0]}). Aborting to avoid overwrite.")
        return

    # --- summary -----------------------------------------------------------
    print("\n" + "-" * 64)
    print("  MERGE SUMMARY")
    print("-" * 64)
    print(f"  existing (base) : {len(base['nums']):>3} frames, "
          f"{base['instances']:>4} instances   {summarize_range(base['nums'])}")
    print(f"  new camera      : {len(plan):>3} frames, "
          f"{new_instances:>4} instances   {summarize_range(new_nums)}")
    print(f"  TOTAL           : {len(base['nums']) + len(plan):>3} frames, "
          f"{base['instances'] + new_instances:>4} instances")
    print()
    print("  >>> CUTOFF between cameras (record this):")
    print(f"        base last  = frame_{base_max:06d}")
    print(f"        new first  = frame_{new_nums[0]:06d}")
    print(f"        new last   = frame_{new_nums[-1]:06d}")

    if args.dry_run:
        print("\n[DRY RUN] Nothing written.")
        return

    # --- confirm before writing into the existing dataset ------------------
    if not args.yes:
        resp = input(f"\nAppend {len(plan)} new frames into existing "
                     f"{base_dir}/ ? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted. Nothing written.")
            return

    # --- copy (never move the sources) -------------------------------------
    print(f"\n[3/3] Copying {len(plan)} frames into {base_dir} ...")
    for img, lbl, _num, out_stem, ext in plan:
        shutil.copy(img, base_dir / f"{out_stem}{ext}")
        shutil.copy(lbl, base_dir / f"{out_stem}.txt")

    print(f"\n[DONE] {base_dir} now holds "
          f"{len(base['nums']) + len(plan)} frames "
          f"({base['instances'] + new_instances} instances). Sources untouched.")


# ========================================
# MAIN
# ========================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate and preview counts without writing anything")
    p.add_argument("--yes", action="store_true",
                   help="Skip the confirmation prompt")
    return p.parse_args()


if __name__ == "__main__":
    merge(parse_args())
