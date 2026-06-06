#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable

import numpy as np
from PIL import Image
from ultralytics import YOLO
import toml
from tqdm import tqdm

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config():
    """Load config from config.toml if available."""
    config_path = pathlib.Path("config.toml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return toml.load(f)
    return {}


def parse_args() -> argparse.Namespace:
    config = load_config()

    # Get defaults from config
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    # Default image size from config (assuming square, take first element)
    default_imgsz = 640
    if "images_size" in data_config and data_config["images_size"]:
        default_imgsz = int(data_config["images_size"][0])

    # Default seg_root from config
    default_seg_root = data_config.get("preprocessed_path", "data/preprocessed")

    # Default weights: try to use the best model from 4-fold CV training
    default_weights = pathlib.Path("checkpoints/best_detector/best_yolo_cv.pt")
    if not default_weights.exists():
        default_weights = None  # Will make it required if not found

    parser = argparse.ArgumentParser(
        description="Export YOLOv8 predicted bboxes for segmentation splits"
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=default_weights,
        help="Path to trained Ultralytics YOLO weights (.pt). Defaults to checkpoints/best_detector/best_yolo_cv.pt if exists.",
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=default_seg_root,
        help="Segmentation root with split/images and split/masks",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/yolo_pred_bboxes_segmentation_ultralytics"),
    )
    parser.add_argument("--imgsz", type=int, default=default_imgsz)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def iter_images(image_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.suffix.lower() in VALID_EXTS
    ]
    for path in paths:
        yield path


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return xc / w, yc / h, bw / w, bh / h


def check_has_caries(
    mask_path: pathlib.Path, x1: float, y1: float, x2: float, y2: float
) -> bool:
    if not mask_path.exists():
        return False
    with Image.open(mask_path) as m:
        mask = np.array(m.convert("L"))

    # Clip coordinates to mask boundaries
    h, w = mask.shape
    ix1, iy1 = int(max(0, x1)), int(max(0, y1))
    ix2, iy2 = int(min(w, x2)), int(min(h, y2))

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    crop = mask[iy1:iy2, ix1:ix2]
    return np.any(crop > 0)


def main() -> None:
    args = parse_args()

    # Check if weights are provided (either by arg or default)
    if args.weights is None:
        raise FileNotFoundError(
            "Weights not specified and default weights not found at 'checkpoints/best_detector/best_yolo_cv.pt'. "
            "Please specify --weights explicitly."
        )

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        image_dir = args.seg_root / split / "images"
        mask_dir = args.seg_root / split / "masks"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        labels_dir = args.output_dir / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        box_rows: list[dict[str, object]] = []
        total_boxes = 0

        # Get list of images for progress bar
        image_paths = list(iter_images(image_dir))
        for image_path in tqdm(image_paths, desc=f"Processing {split} split"):
            sample_id = image_path.stem
            mask_path = mask_dir / f"{sample_id}.png"

            with Image.open(image_path) as im:
                orig_w, orig_h = im.size

            predict_kwargs = {
                "source": str(image_path),
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
                "verbose": False,
            }
            if args.device:
                predict_kwargs["device"] = args.device

            result = model.predict(**predict_kwargs)[0]

            label_path = labels_dir / f"{sample_id}.txt"
            image_boxes = []

            for idx, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0].item())

                x1, y1, x2, y2 = xyxy
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_x = bw * args.padding_ratio
                pad_y = bh * args.padding_ratio

                x1 = clip(x1 - pad_x, 0.0, float(orig_w))
                y1 = clip(y1 - pad_y, 0.0, float(orig_h))
                x2 = clip(x2 + pad_x, 0.0, float(orig_w))
                y2 = clip(y2 + pad_y, 0.0, float(orig_h))

                if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                    continue

                has_caries = check_has_caries(mask_path, x1, y1, x2, y2)
                xc_n, yc_n, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)

                image_boxes.append(
                    f"0 {xc_n:.6f} {yc_n:.6f} {bw_n:.6f} {bh_n:.6f} {score:.6f}"
                )

                box_rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "box_index": idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": score,
                        "has_caries": has_caries,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path) if mask_path.exists() else "",
                    }
                )
                total_boxes += 1

            with label_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(image_boxes))

        meta_path = args.output_dir / f"{split}_box_metadata.csv"
        if box_rows:
            fields = list(box_rows[0].keys())
            with meta_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(box_rows)

        print(
            f"split={split}: images={len(list(iter_images(image_dir)))} total_boxes={total_boxes} "
            f"labels_dir={labels_dir} metadata={meta_path}"
        )


if __name__ == "__main__":
    main()
