#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Iterable

from PIL import Image
from ultralytics import YOLO

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 predicted bboxes for segmentation splits"
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        required=True,
        help="Path to trained Ultralytics YOLO weights (.pt)",
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=pathlib.Path("data/preprocessed"),
        help="Segmentation root with split/images",
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
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def iter_images(image_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in VALID_EXTS:
            yield path


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return xc / w, yc / h, bw / w, bh / h


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        image_dir = args.seg_root / split / "images"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        labels_dir = args.output_dir / split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        total_boxes = 0

        for image_path in iter_images(image_dir):
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                orig_w, orig_h = rgb.size

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

            label_path = labels_dir / f"{image_path.stem}.txt"
            image_box_count = 0
            with label_path.open("w", encoding="utf-8") as f:
                for box in result.boxes:
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

                    xc, yc, bw_n, bh_n = xyxy_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f} {score:.6f}\n")
                    total_boxes += 1
                    image_box_count += 1

            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "pred_label": str(label_path),
                    "num_boxes": str(image_box_count),
                }
            )

        meta_path = args.output_dir / f"{split}_metadata.csv"
        with meta_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "image", "pred_label", "num_boxes"],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"split={split}: images={len(rows)} total_boxes={total_boxes} "
            f"labels_dir={labels_dir} metadata={meta_path}"
        )


if __name__ == "__main__":
    main()
