#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List, Optional, Tuple

from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut tooth image segments from segmentation data using predicted YOLO bboxes"
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
        help="Resize size used before applying bbox coordinates",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Ignore predicted boxes with score below this threshold",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.05,
        help="Expand each bbox by this ratio of its own width/height",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum valid crop width/height in pixels",
    )
    parser.add_argument(
        "--fallback-full-image",
        action="store_true",
        help="If no valid box exists, emit one full-image crop",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Export only cropped tooth images (skip mask cropping)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/segmentation_crops_from_pred_bboxes"),
        help="Output folder for cropped segments",
    )
    return parser.parse_args()


def parse_pred_boxes(label_path: pathlib.Path, image_size: int, score_threshold: float) -> List[Tuple[float, float, float, float, float]]:
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
            if score < score_threshold:
                continue

            x1 = (xc - bw / 2.0) * image_size
            y1 = (yc - bh / 2.0) * image_size
            x2 = (xc + bw / 2.0) * image_size
            y2 = (yc + bh / 2.0) * image_size
            boxes.append((x1, y1, x2, y2, score))

    return boxes


def expand_and_clip(
    box: Tuple[float, float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
    min_size: int,
) -> Optional[Tuple[int, int, int, int, float]]:
    x1, y1, x2, y2, score = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    px = bw * padding_ratio
    py = bh * padding_ratio

    xi1 = max(0, int(round(x1 - px)))
    yi1 = max(0, int(round(y1 - py)))
    xi2 = min(width, int(round(x2 + px)))
    yi2 = min(height, int(round(y2 + py)))

    if xi2 - xi1 < min_size or yi2 - yi1 < min_size:
        return None

    return xi1, yi1, xi2, yi2, score


def write_metadata(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "split",
        "sample_id",
        "crop_id",
        "source_image",
        "source_mask",
        "pred_label",
        "crop_image",
        "crop_mask",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "crop_w",
        "crop_h",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_split(args: argparse.Namespace, split: str) -> Dict[str, object]:
    image_dir = args.seg_root / split / "images"
    mask_dir = args.seg_root / split / "masks"
    pred_dir = args.pred_root / split / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image dir for split={split}")
    if not args.images_only and not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir for split={split}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predicted labels for split={split}: {pred_dir}")

    out_img_dir = args.output_dir / split / "images"
    out_mask_dir = args.output_dir / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if not args.images_only:
        out_mask_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    total_images = 0
    no_box_samples = 0
    total_crops = 0

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS])

    for image_path in image_paths:
        total_images += 1
        sample_id = image_path.stem
        mask_path = mask_dir / f"{sample_id}.png"
        pred_label = pred_dir / f"{sample_id}.txt"

        if not args.images_only and not mask_path.exists():
            continue

        with Image.open(image_path) as im:
            im_r = im.convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
        if not args.images_only:
            with Image.open(mask_path) as mm:
                mask_r = mm.convert("L").resize((args.image_size, args.image_size), Image.NEAREST)

        boxes = parse_pred_boxes(pred_label, args.image_size, args.score_threshold)
        valid_boxes: List[Tuple[int, int, int, int, float]] = []
        for box in boxes:
            clipped = expand_and_clip(
                box,
                width=args.image_size,
                height=args.image_size,
                padding_ratio=args.padding_ratio,
                min_size=args.min_size,
            )
            if clipped is not None:
                valid_boxes.append(clipped)

        if not valid_boxes:
            no_box_samples += 1
            if args.fallback_full_image:
                valid_boxes = [(0, 0, args.image_size, args.image_size, -1.0)]
            else:
                continue

        for idx, (x1, y1, x2, y2, score) in enumerate(valid_boxes):
            crop_id = f"{sample_id}_crop{idx:03d}"
            crop_img = im_r.crop((x1, y1, x2, y2))

            crop_img_path = out_img_dir / f"{crop_id}.png"

            crop_img.save(crop_img_path)
            crop_mask_path = ""
            if not args.images_only:
                crop_mask = mask_r.crop((x1, y1, x2, y2))
                crop_mask_file = out_mask_dir / f"{crop_id}.png"
                crop_mask.save(crop_mask_file)
                crop_mask_path = str(crop_mask_file)
            total_crops += 1

            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "crop_id": crop_id,
                    "source_image": str(image_path),
                    "source_mask": "" if args.images_only else str(mask_path),
                    "pred_label": str(pred_label),
                    "crop_image": str(crop_img_path),
                    "crop_mask": crop_mask_path,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": float(score),
                    "crop_w": x2 - x1,
                    "crop_h": y2 - y1,
                }
            )

    meta_path = args.output_dir / split / "metadata.csv"
    write_metadata(meta_path, rows)

    return {
        "split": split,
        "total_images": total_images,
        "no_box_samples": no_box_samples,
        "total_crops": total_crops,
        "metadata": str(meta_path),
        "images_dir": str(out_img_dir),
        "masks_dir": "" if args.images_only else str(out_mask_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}
    for split in args.splits:
        s = process_split(args, split)
        summary[split] = s
        print(
            f"split={split}: images={s['total_images']} crops={s['total_crops']} "
            f"no_box={s['no_box_samples']}"
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
