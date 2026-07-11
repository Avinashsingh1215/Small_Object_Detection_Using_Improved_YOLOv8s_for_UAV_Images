#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

DATASET_ROOT = Path("/home/user1/Avinash/Small_Obj_Detection/Dataset")

def convert_split(split_name, source_folder):
    images_src = source_folder / "images"
    ann_src = source_folder / "annotations"
    if not images_src.exists() or not ann_src.exists():
        print(f"⚠️ Skipping {split_name}: {images_src} or {ann_src} not found")
        return

    images_dst = DATASET_ROOT / "images" / split_name
    labels_dst = DATASET_ROOT / "labels" / split_name
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_src.glob("*.txt"))
    print(f"Converting {split_name}: {len(ann_files)} annotations")

    for ann_file in tqdm(ann_files, desc=f"  {split_name}"):
        img_file = images_src / f"{ann_file.stem}.jpg"
        if not img_file.exists():
            img_file = images_src / f"{ann_file.stem}.png"
        if not img_file.exists():
            print(f"  Missing image for {ann_file.name}, skipping")
            continue

        try:
            with Image.open(img_file) as img:
                w, h = img.size
        except Exception as e:
            print(f"  Error reading {img_file}: {e}")
            continue

        shutil.copy(img_file, images_dst / img_file.name)

        with open(ann_file, "r") as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w_bbox, h_bbox = map(int, parts[:4])
                score = int(parts[4])
                cls = int(parts[5])
                if cls == 0 or score == 0:
                    continue
                x_center = (x + w_bbox / 2) / w
                y_center = (y + h_bbox / 2) / h
                width_norm = w_bbox / w
                height_norm = h_bbox / h
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width_norm = max(0, min(1, width_norm))
                height_norm = max(0, min(1, height_norm))
                yolo_cls = cls - 1
                yolo_lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            except Exception:
                continue

        if yolo_lines:
            label_file = labels_dst / f"{ann_file.stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_lines))

def main():
    print("Converting VisDrone dataset to YOLO format...")
    splits = {
        "train": DATASET_ROOT / "VisDrone2019-DET-train",
        "val": DATASET_ROOT / "VisDrone2019-DET-val",
        "test": DATASET_ROOT / "VisDrone2019-DET-test-dev",
    }
    for split_name, src in splits.items():
        if src.exists():
            convert_split(split_name, src)
        else:
            print(f"⚠️ Source folder not found: {src}")
    print("\n✅ Conversion complete.")
    print(f"YOLO dataset root: {DATASET_ROOT}")
    print("You can now train with:")
    print("  python train.py --config configs/my_config.yaml")

if __name__ == "__main__":
    main()
