#!/usr/bin/env python3
import argparse
import subprocess
import os
import toml
import sys

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Full Method 1 Pipeline: YOLO -> Crop -> UNet")
    parser.add_argument("--config", type=str, default="config.method1_full.toml", help="Path to full config")
    parser.add_argument("--skip-stage1", action="store_true", help="Skip YOLO training")
    parser.add_argument("--skip-export", action="store_true", help="Skip bbox export")
    parser.add_argument("--smoke-run", action="store_true", help="Run with minimal epochs/data for debugging")
    args = parser.parse_args()

    config = toml.load(args.config)

    # Stage 1: YOLO Training
    if not args.skip_stage1:
        s1 = config["stage1_detection"]
        cmd = [
            "uv", "run", "scripts/train_ultralytics_yolo.py",
            "--data-dir", s1["data_dir"],
            "--model", s1["model"],
            "--epochs", str(1 if args.smoke_run else s1["epochs"]),
            "--imgsz", str(s1["imgsz"]),
            "--batch", str(s1["batch"]),
            "--name", s1["name"]
        ]
        run_command(cmd)

    # Stage 1.5: Export Bboxes
    if not args.skip_export:
        se = config["stage1_export"]
        cmd = [
            "uv", "run", "scripts/export_ultralytics_bboxes_for_segmentation.py",
            "--weights", se["weights"],
            "--seg-root", se["seg_root"],
            "--splits", *se["splits"],
            "--output-dir", se["output_dir"],
            "--imgsz", str(se["imgsz"]),
            "--conf", str(se["conf"]),
            "--iou", str(se["iou"]),
            "--padding-ratio", str(se["padding_ratio"])
        ]
        run_command(cmd)

        # Stage 1.7: Materialize Crops
        print("Materializing crops for segmentation...")
        cmd = [
            "uv", "run", "scripts/export_segmentation_crops_from_bboxes.py",
            "--pred-root", se["output_dir"],
            "--output-dir", os.path.join(se["output_dir"], "crops"),
            "--splits", *se["splits"],
            "--image-size", str(se["imgsz"])
        ]
        run_command(cmd)

    # Stage 2: UNet Training
    cmd = [
        "uv", "run", "python", "-m", "src.train",
        "--config", args.config
    ]
    if args.smoke_run:
        # Override epochs via env var or we could add support in train.py for CLI overrides
        os.environ["SMOKE_RUN"] = "1"
    
    run_command(cmd)

if __name__ == "__main__":
    main()
