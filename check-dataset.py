from pathlib import Path

img_train = list(Path(r"dataset/dataset/images/train").glob("*.jpg"))
lbl_train = list(Path(r"dataset/dataset/labels/train").glob("*.txt"))

print(f"Images : {len(img_train)}")
print(f"Labels : {len(lbl_train)}")

# ดูว่า label มี annotation จริงไหม
total = 0
for f in lbl_train:
    lines = f.read_text().strip().splitlines()
    total += len(lines)
    print(f"{f.name}: {len(lines)} heads")

print(f"Total annotations: {total}")