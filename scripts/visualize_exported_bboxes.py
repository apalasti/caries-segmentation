#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize exported predicted YOLO bboxes on segmentation images"
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=pathlib.Path("data/preprocessed"),
        help="Segmentation dataset root",
    )
    parser.add_argument(
        "--pred-root",
        type=pathlib.Path,
        default=pathlib.Path("outputs/yolo_pred_bboxes_segmentation"),
        help="Predicted bbox label root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Rendering size (must match bbox label coordinate space)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="How many random samples per split to render",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Hide boxes below score threshold (if score present)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/bbox_visualizations"),
    )
    return parser.parse_args()


def parse_pred_boxes(label_path: pathlib.Path, image_size: int) -> List[Tuple[float, float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            xc, yc, bw, bh = map(float, parts[1:5])
            score = float(parts[5]) if len(parts) >= 6 else 1.0

            x1 = (xc - bw / 2.0) * image_size
            y1 = (yc - bh / 2.0) * image_size
            x2 = (xc + bw / 2.0) * image_size
            y2 = (yc + bh / 2.0) * image_size
            boxes.append((x1, y1, x2, y2, score))
    return boxes


def draw_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float, float]], score_threshold: float) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2, score in boxes:
        if score < score_threshold:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=(255, 180, 0), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{score:.2f}", fill=(255, 180, 0))
    return canvas


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        image_dir = args.seg_root / split / "images"
        pred_dir = args.pred_root / split / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {pred_dir}")

        split_out = args.output_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
        if not image_paths:
            continue

        count = min(args.num_samples, len(image_paths))
        idxs = rng.choice(len(image_paths), size=count, replace=False)

        for i in idxs:
            image_path = image_paths[int(i)]
            with Image.open(image_path) as im:
                im_resized = im.convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)

            boxes = parse_pred_boxes(pred_dir / f"{image_path.stem}.txt", args.image_size)
            rendered = draw_boxes(im_resized, boxes, args.score_threshold)
            rendered.save(split_out / f"{image_path.stem}.png")

        print(f"Saved {count} visualizations for split={split} -> {split_out}")


if __name__ == "__main__":
    main()
