#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute caries-mask coverage outside predicted bbox unions"
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=pathlib.Path("data/preprocessed"),
        help="Segmentation dataset root with split/images and split/masks",
    )
    parser.add_argument(
        "--pred-root",
        type=pathlib.Path,
        default=pathlib.Path("outputs/yolo_pred_bboxes_segmentation"),
        help="Predicted bbox export root with split/labels",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Evaluation size used for predicted YOLO labels",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs/bbox_outside_caries_stats.json"),
    )
    return parser.parse_args()


def parse_pred_boxes(label_path: pathlib.Path, image_size: int) -> List[Tuple[float, float, float, float]]:
    boxes: List[Tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2.0) * image_size
            y1 = (yc - bh / 2.0) * image_size
            x2 = (xc + bw / 2.0) * image_size
            y2 = (yc + bh / 2.0) * image_size
            boxes.append((x1, y1, x2, y2))
    return boxes


def component_boxes(mask_np: np.ndarray) -> List[Tuple[float, float, float, float]]:
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[float, float, float, float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def coverage(mask_np: np.ndarray, boxes: Iterable[Tuple[float, float, float, float]]) -> Dict[str, float]:
    gt = mask_np > 0.5
    gt_pixels = int(gt.sum())
    h, w = gt.shape

    if gt_pixels == 0:
        return {
            "gt_pixels": 0.0,
            "outside_pixels": 0.0,
            "coverage_pct": 100.0,
        }

    inside = np.zeros_like(gt, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xi1 = max(0, min(w, int(round(x1))))
        yi1 = max(0, min(h, int(round(y1))))
        xi2 = max(0, min(w, int(round(x2))))
        yi2 = max(0, min(h, int(round(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        inside[yi1:yi2, xi1:xi2] = True

    outside = int(np.logical_and(gt, np.logical_not(inside)).sum())
    return {
        "gt_pixels": float(gt_pixels),
        "outside_pixels": float(outside),
        "coverage_pct": float(100.0 * (1.0 - outside / max(1, gt_pixels))),
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    coverages = np.array([r["coverage_pct"] for r in rows], dtype=float)
    gt_pixels = np.array([r["gt_pixels"] for r in rows], dtype=float)
    outside = np.array([r["outside_pixels"] for r in rows], dtype=float)

    total_gt = float(gt_pixels.sum())
    total_out = float(outside.sum())

    weighted_coverage = 100.0 * (1.0 - total_out / max(1.0, total_gt))

    return {
        "images": int(len(rows)),
        "images_with_caries": int((gt_pixels > 0).sum()),
        "leak_samples": int((outside > 0).sum()),
        "mean_coverage_pct": float(coverages.mean()) if len(coverages) else 100.0,
        "median_coverage_pct": float(np.median(coverages)) if len(coverages) else 100.0,
        "p25_coverage_pct": float(np.percentile(coverages, 25)) if len(coverages) else 100.0,
        "p75_coverage_pct": float(np.percentile(coverages, 75)) if len(coverages) else 100.0,
        "total_caries_pixels": int(total_gt),
        "outside_caries_pixels": int(total_out),
        "outside_ratio_pct": float(100.0 * total_out / max(1.0, total_gt)),
        "weighted_coverage_pct": float(weighted_coverage),
    }


def main() -> None:
    args = parse_args()

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for split in args.splits:
        mask_dir = args.seg_root / split / "masks"
        pred_dir = args.pred_root / split / "labels"

        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing predicted label directory: {pred_dir}")

        rows_pred: List[Dict[str, float]] = []
        rows_gt: List[Dict[str, float]] = []

        for mask_path in sorted(mask_dir.glob("*.png")):
            with Image.open(mask_path) as m:
                arr = np.array(
                    m.convert("L").resize((args.image_size, args.image_size), Image.NEAREST),
                    dtype=np.float32,
                ) / 255.0

            pred_boxes = parse_pred_boxes(pred_dir / f"{mask_path.stem}.txt", args.image_size)
            gt_boxes = component_boxes(arr)

            rows_pred.append(coverage(arr, pred_boxes))
            rows_gt.append(coverage(arr, gt_boxes))

        summary[split] = {
            "predicted_boxes": aggregate(rows_pred),
            "gt_component_boxes_baseline": aggregate(rows_gt),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved stats to {args.output}")
    for split in args.splits:
        p = summary[split]["predicted_boxes"]
        g = summary[split]["gt_component_boxes_baseline"]
        print(
            f"{split}: pred weighted_coverage={p['weighted_coverage_pct']:.2f}% "
            f"outside_ratio={p['outside_ratio_pct']:.2f}% "
            f"leak_samples={p['leak_samples']}/{p['images']}"
        )
        print(
            f"{split}: gtbox weighted_coverage={g['weighted_coverage_pct']:.2f}% "
            f"outside_ratio={g['outside_ratio_pct']:.2f}% "
            f"leak_samples={g['leak_samples']}/{g['images']}"
        )


if __name__ == "__main__":
    main()
