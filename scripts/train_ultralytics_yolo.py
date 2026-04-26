#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
from typing import Any

import yaml
from roboflow import Roboflow
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DETECTION_DIR = ROOT / "data" / "preprocessed_detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Ultralytics YOLOv8 for tooth detection"
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DEFAULT_DETECTION_DIR,
        help="Dataset root with train/val/test and images/labels",
    )
    parser.add_argument(
        "--data-yaml",
        type=pathlib.Path,
        default=None,
        help="Existing YOLO data.yaml. If not set, one is generated from --data-dir",
    )
    parser.add_argument(
        "--generated-data-yaml",
        type=pathlib.Path,
        default=DEFAULT_DETECTION_DIR / "data.ultralytics.yaml",
        help="Where to save generated data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Ultralytics model checkpoint name or path",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=ROOT / "runs" / "detect",
        help="Ultralytics runs root",
    )
    parser.add_argument("--name", type=str, default="tooth_ultralytics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split used for post-training validation",
    )

    parser.add_argument(
        "--download-roboflow",
        action="store_true",
        help="Download dataset from Roboflow and use its generated data.yaml",
    )
    parser.add_argument("--rf-workspace", type=str, default="")
    parser.add_argument("--rf-project", type=str, default="")
    parser.add_argument("--rf-version", type=int, default=1)
    parser.add_argument("--rf-format", type=str, default="yolov8")
    parser.add_argument(
        "--rf-api-key",
        type=str,
        default="",
        help="Roboflow API key. If omitted, ROBOFLOW_API_KEY env is used",
    )

    parser.add_argument(
        "--predict-source",
        type=str,
        default="",
        help="Optional inference source (image path, dir, glob, video, URL)",
    )
    parser.add_argument("--predict-conf", type=float, default=0.25)
    parser.add_argument("--predict-iou", type=float, default=0.45)
    parser.add_argument("--predict-max-det", type=int, default=300)
    parser.add_argument(
        "--predict-name",
        type=str,
        default="predict_after_train",
        help="Run name for optional predict step",
    )
    return parser.parse_args()


def create_data_yaml(data_dir: pathlib.Path, out_yaml: pathlib.Path) -> pathlib.Path:
    split_map = {
        "train": "train",
        "val": "val" if (data_dir / "val").exists() else "valid",
        "test": "test",
    }

    for canonical, real_split in split_map.items():
        split_path = data_dir / real_split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split directory for {canonical}: expected {split_path}"
            )
        images_dir = split_path / "images"
        labels_dir = split_path / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(
                f"Expected images/labels under {split_path}, got images={images_dir.exists()} labels={labels_dir.exists()}"
            )

    data = {
        "path": str(data_dir.resolve()),
        "train": f"{split_map['train']}/images",
        "val": f"{split_map['val']}/images",
        "test": f"{split_map['test']}/images",
        "names": {0: "tooth"},
        "nc": 1,
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_yaml


def download_from_roboflow(args: argparse.Namespace) -> pathlib.Path:
    api_key = args.rf_api_key or os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("Missing Roboflow API key. Set --rf-api-key or ROBOFLOW_API_KEY.")
    if not args.rf_workspace or not args.rf_project:
        raise ValueError("--rf-workspace and --rf-project are required with --download-roboflow.")

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(args.rf_workspace)
        .project(args.rf_project)
        .version(args.rf_version)
        .download(args.rf_format)
    )

    data_yaml = pathlib.Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Roboflow download did not include data.yaml: {data_yaml}")
    return data_yaml


def train_and_eval(args: argparse.Namespace, data_yaml: pathlib.Path) -> tuple[pathlib.Path, Any]:
    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "seed": args.seed,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = pathlib.Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best weights not found: {best_weights}")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(data=str(data_yaml), split=args.val_split)

    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    print(f"Validation split={args.val_split}: mAP50={map50:.4f}, mAP50-95={map5095:.4f}")
    print(f"Best weights: {best_weights}")

    return best_weights, best_model


def maybe_predict(args: argparse.Namespace, model: YOLO) -> None:
    if not args.predict_source:
        return

    predict_kwargs: dict[str, Any] = {
        "source": args.predict_source,
        "conf": args.predict_conf,
        "iou": args.predict_iou,
        "max_det": args.predict_max_det,
        "project": str(args.project),
        "name": args.predict_name,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "exist_ok": True,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    model.predict(**predict_kwargs)
    print(
        "Prediction export complete under "
        f"{args.project / args.predict_name} (images + labels + confidence)."
    )


def main() -> None:
    args = parse_args()

    if args.download_roboflow:
        data_yaml = download_from_roboflow(args)
        print(f"Using Roboflow data.yaml: {data_yaml}")
    else:
        if args.data_yaml is not None:
            data_yaml = args.data_yaml
            if not data_yaml.exists():
                raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
        else:
            data_yaml = create_data_yaml(args.data_dir, args.generated_data_yaml)
            print(f"Generated data.yaml: {data_yaml}")

    best_weights, best_model = train_and_eval(args, data_yaml)
    maybe_predict(args, best_model)

    print("Run complete.")
    print(f"Trained weights: {best_weights}")


if __name__ == "__main__":
    main()
