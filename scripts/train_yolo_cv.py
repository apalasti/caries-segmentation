#!/usr/bin/env python3
import argparse
import os
import pathlib
import shutil
import yaml
import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with 3-Fold Cross Validation")
    parser.add_argument("--data-dir", type=pathlib.Path, default="data/preprocessed_detection", help="Path to YOLO dataset")
    parser.add_argument("--project", type=str, default="tooth_detection_cv", help="Project name for YOLO")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    return parser.parse_args()

def create_fold_yaml(data_dir, fold_idx, train_files, val_files, output_dir):
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # We use symlinks to avoid copying large amounts of data
    # Actually, Ultralytics prefers a list of files or a directory.
    # We'll create a temporary directory with symlinks to images.
    
    for split, files in [("train", train_files), ("val", val_files)]:
        split_img_dir = fold_dir / split / "images"
        split_lbl_dir = fold_dir / split / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in files:
            lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            (split_img_dir / img_path.name).symlink_to(img_path.resolve())
            if lbl_path.exists():
                (split_lbl_dir / lbl_path.name).symlink_to(lbl_path.resolve())

    data_yaml = {
        "path": str(fold_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "tooth"},
        "nc": 1
    }
    
    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)
    
    return yaml_path

def main():
    args = parse_args()
    
    # Collect all images from train/val (ignore test for CV)
    all_images = []
    for split in ["train", "val"]:
        img_dir = args.data_dir / split / "images"
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    
    all_images = np.array(sorted(all_images))
    print(f"Found {len(all_images)} images for cross-validation.")
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    output_cv_dir = pathlib.Path("cv_temp")
    if output_cv_dir.exists():
        shutil.rmtree(output_cv_dir)
    output_cv_dir.mkdir(parents=True, exist_ok=True)
    
    best_maps = []
    best_weights_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\n--- Starting Fold {fold_idx + 1} / 3 ---")
        train_files = all_images[train_idx]
        val_files = all_images[val_idx]
        
        yaml_path = create_fold_yaml(args.data_dir, fold_idx + 1, train_files, val_files, output_cv_dir)
        
        model = YOLO("yolov8s.pt")
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=f"fold_{fold_idx + 1}",
            exist_ok=True
        )
        
        # After training, get the best map50-95 from validation
        metrics = model.val()
        best_maps.append(metrics.box.map)
        best_weights_paths.append(pathlib.Path(results.save_dir) / "weights" / "best.pt")
        
    print("\n--- Cross-Validation Results ---")
    for i, m in enumerate(best_maps):
        print(f"Fold {i+1}: mAP50-95 = {m:.4f}")
    
    avg_map = np.mean(best_maps)
    print(f"Average mAP50-95: {avg_map:.4f}")
    
    best_fold = np.argmax(best_maps)
    print(f"Best model is from Fold {best_fold + 1} at {best_weights_paths[best_fold]}")
    
    # Copy best model to a final location
    final_dir = pathlib.Path("checkpoints/best_detector")
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights_paths[best_fold], final_dir / "best_yolo_cv.pt")
    print(f"Best model saved to {final_dir / 'best_yolo_cv.pt'}")

if __name__ == "__main__":
    main()
