#!/usr/bin/env python3
"""
Create 4-fold cross-validation datasets with presaved augmentations on disk.

- Reads images from `data/preprocessed_detection` (all splits).
- Deduplicates by base name (strips `.rf.<hash>` and `_aug` suffixes), keeping one file per base.
- Splits unique images into 4 groups: `group1`..`group4` under `outputs/cv_groups`.
- For each fold i (1..4):
    - val = group{i}
    - train = union of other groups
    - For every image in train: generate N augmented variants and save to
      `outputs/cv_folds/fold_{i}/train/images` and corresponding labels in `.../labels`.
    - Copy val images (no augmentation) to `.../val` folders.

This preserves label correctness by using Albumentations bbox transforms (YOLO format).
"""

from __future__ import annotations

import argparse
import pathlib
import random
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


ROOT = pathlib.Path(__file__).resolve().parent.parent
PREDET = ROOT / "data" / "preprocessed_detection"
OUT_GROUPS = ROOT / "outputs" / "cv_groups"
OUT_FOLDS = ROOT / "outputs" / "cv_folds"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_base_name(name: str) -> str:
    base = name
    if ".rf." in base:
        base = base.split('.rf.')[0]
    if "_aug" in base:
        base = base.split('_aug')[0]
    return base


def collect_unique_images(preproc_dir: pathlib.Path) -> List[pathlib.Path]:
    imgs = []
    for split in ("train", "val", "valid", "test"):
        d = preproc_dir / split / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in VALID_EXTS:
                imgs.append(p)

    # Keep one per base name, prefer non-aug when present
    seen = {}
    for p in sorted(imgs, key=lambda x: (1 if "_aug" in x.name else 0, x.name)):
        base = get_base_name(p.name)
        if base not in seen:
            seen[base] = p
        else:
            # prefer non-aug
            if ("_aug" not in p.name) and ("_aug" in seen[base].name):
                seen[base] = p
    return list(seen.values())


def read_yolo_labels(lbl_path: pathlib.Path) -> List[Tuple[int, float, float, float, float]]:
    # returns list of (cls, x,y,w,h) normalized
    boxes = []
    if not lbl_path.exists():
        return boxes
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            if len(parts) < 5:
                continue
            xc, yc, w, h = map(float, parts[1:5])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def write_yolo_labels(lbl_path: pathlib.Path, boxes: List[Tuple[int, float, float, float, float]]):
    with lbl_path.open("w", encoding="utf-8") as f:
        for (cls, xc, yc, w, h) in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def build_augmenter():
    return A.Compose(
        [
            A.Rotate(limit=10, p=0.6, border_mode=0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.Blur(blur_limit=3, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )


def create_groups(unique_images: List[pathlib.Path], n_splits: int = 4):
    idx = np.arange(len(unique_images))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    OUT_GROUPS.mkdir(parents=True, exist_ok=True)
    group_paths = []
    for i, inds in enumerate(folds, start=1):
        gdir = OUT_GROUPS / f"group{i}"
        img_dir = gdir / "images"
        lbl_dir = gdir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for j in inds:
            p = unique_images[j]
            # symlink image and label into group folder
            img_dst = img_dir / p.name
            lbl_src = p.parent.parent / "labels" / f"{p.stem}.txt"
            lbl_dst = lbl_dir / f"{p.stem}.txt"
            if not img_dst.exists():
                img_dst.symlink_to(p.resolve())
            if lbl_src.exists() and not lbl_dst.exists():
                lbl_dst.symlink_to(lbl_src.resolve())
        group_paths.append(gdir)
    return group_paths


def make_fold_dataset(groups: List[pathlib.Path], fold_idx: int, n_augs: int = 5):
    # val is groups[fold_idx], train is others
    val_group = groups[fold_idx]
    train_groups = [g for i, g in enumerate(groups) if i != fold_idx]

    fold_out = OUT_FOLDS / f"fold_{fold_idx + 1}"
    train_img_out = fold_out / "train" / "images"
    train_lbl_out = fold_out / "train" / "labels"
    val_img_out = fold_out / "val" / "images"
    val_lbl_out = fold_out / "val" / "labels"
    # clean
    if fold_out.exists():
        shutil.rmtree(fold_out)
    train_img_out.mkdir(parents=True, exist_ok=True)
    train_lbl_out.mkdir(parents=True, exist_ok=True)
    val_img_out.mkdir(parents=True, exist_ok=True)
    val_lbl_out.mkdir(parents=True, exist_ok=True)

    augmenter = build_augmenter()

    # copy val files (no augmentation)
    for p in sorted((val_group / "images").iterdir()):
        if p.suffix.lower() not in VALID_EXTS:
            continue
        src_img = p.resolve()
        src_lbl = (val_group / "labels" / f"{p.stem}.txt").resolve()
        dst_img = val_img_out / p.name
        if not dst_img.exists():
            dst_img.symlink_to(src_img)
        if src_lbl.exists():
            dst_lbl = val_lbl_out / src_lbl.name
            if not dst_lbl.exists():
                dst_lbl.symlink_to(src_lbl)

    # for training groups, copy original and create augmentations
    for g in train_groups:
        for p in sorted((g / "images").iterdir()):
            if p.suffix.lower() not in VALID_EXTS:
                continue
            src_img = p.resolve()
            src_lbl = (g / "labels" / f"{p.stem}.txt").resolve()
            # copy original
            orig_dst = train_img_out / p.name
            if not orig_dst.exists():
                orig_dst.symlink_to(src_img)
            if src_lbl.exists():
                orig_lbl_dst = train_lbl_out / f"{p.stem}.txt"
                if not orig_lbl_dst.exists():
                    orig_lbl_dst.symlink_to(src_lbl)
            # generate augmentations
            img = np.array(Image.open(src_img).convert('RGB'))
            boxes = read_yolo_labels(src_lbl) if src_lbl.exists() else []
            if not boxes:
                continue
            # Albumentations expects bboxes as list of tuples (cls, x,y,w,h) in 'yolo' with label list
            for k in range(n_augs):
                class_labels = [b[0] for b in boxes]
                bboxes_in = [tuple(b[1:]) for b in boxes]
                try:
                    augmented = augmenter(image=img, bboxes=bboxes_in, class_labels=class_labels)
                except Exception:
                    # fallback to original
                    augmented_img = img
                    augmented_bboxes = bboxes_in
                    augmented_labels = class_labels
                else:
                    augmented_img = augmented['image']
                    augmented_bboxes = augmented.get('bboxes', [])
                    augmented_labels = augmented.get('class_labels', [])

                if not augmented_bboxes:
                    # skip augmentation that removed boxes
                    continue

                out_name = f"{p.stem}_aug{k+1}.jpg"
                out_lbl_name = f"{p.stem}_aug{k+1}.txt"
                out_img_path = train_img_out / out_name
                out_lbl_path = train_lbl_out / out_lbl_name
                Image.fromarray(augmented_img).save(out_img_path, format='JPEG', quality=95)
                # write labels
                boxes_to_write = []
                for cls_lbl, bxy in zip(augmented_labels, augmented_bboxes):
                    xc, yc, w, h = bxy
                    boxes_to_write.append((int(cls_lbl), float(xc), float(yc), float(w), float(h)))
                write_yolo_labels(out_lbl_path, boxes_to_write)

    print(f"Created fold {fold_idx+1} at {fold_out} (train images: {len(list(train_img_out.iterdir()))}, val images: {len(list(val_img_out.iterdir()))})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-augs", type=int, default=5, help="Number of aug per image for training set")
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()

    unique = collect_unique_images(PREDET)
    print(f"Unique base images: {len(unique)}")
    groups = create_groups(unique, n_splits=args.n_folds)
    print(f"Created {len(groups)} groups under {OUT_GROUPS}")

    # create fold datasets
    for i in range(len(groups)):
        make_fold_dataset(groups, i, n_augs=args.n_augs)

    print("All folds created under:", OUT_FOLDS)


if __name__ == '__main__':
    main()
