#!/usr/bin/env python3
"""
Train Ultralytics YOLOv8 on 4-fold presaved CV datasets created by `create_cv_aug_datasets.py`.

This script expects folders under `outputs/cv_folds/fold_{i}/train` and `.../val` with
images/ and labels/ subfolders. It trains one model per fold and selects the best
by mAP (box.map) and copies the best weights to `checkpoints/best_detector/best_yolo_cv.pt`.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import numpy as np
from ultralytics import YOLO
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CV_FOLDS = ROOT / "outputs" / "cv_folds"
OUT_CHECKPOINT = ROOT / "checkpoints" / "best_detector"


def make_data_yaml(fold_dir: pathlib.Path) -> pathlib.Path:
    yaml_path = fold_dir / "data.yaml"
    data = {
        "path": str(fold_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "tooth"},
        "nc": 1,
    }
    with yaml_path.open("w") as f:
        yaml.dump(data, f)
    return yaml_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds-dir", type=pathlib.Path, default=CV_FOLDS)
    p.add_argument("--project", type=str, default="tooth_cv_4fold")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    folds = sorted([p for p in args.folds_dir.iterdir() if p.is_dir()])
    best_maps = []
    best_weights = []

    for i, fold in enumerate(folds, start=1):
        print(f"\n=== Training fold {i} @ {fold} ===")
        yaml_path = make_data_yaml(fold)
        model = YOLO("yolov8s.pt")
        res = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=f"fold_{i}",
            exist_ok=True,
        )
        # Evaluate
        metrics = model.val()
        try:
            m = float(metrics.box.map)
        except Exception:
            m = 0.0
        best_maps.append(m)
        best_weights.append(pathlib.Path(res.save_dir) / "weights" / "best.pt")
        print(f"Fold {i} mAP50-95 = {m:.4f}")

    best_idx = int(np.argmax(best_maps))
    print(f"Best fold: {best_idx+1} with mAP {best_maps[best_idx]:.4f}")
    OUT_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights[best_idx], OUT_CHECKPOINT / "best_yolo_cv.pt")
    print(f"Best model copied to {OUT_CHECKPOINT / 'best_yolo_cv.pt'}")


if __name__ == '__main__':
    main()
