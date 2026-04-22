#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all samples where GT caries pixels lie outside predicted bbox unions"
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
        help="Predicted YOLO bbox root",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Evaluation image size used for predicted labels",
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
        default=pathlib.Path("outputs/bbox_outside_examples"),
        help="Directory to save leak-example CSV/JSON files",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="If set, export all samples (not only leak samples)",
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


def coverage(mask_np: np.ndarray, boxes: Iterable[Tuple[float, float, float, float]]) -> Dict[str, float]:
    gt = mask_np > 0.5
    gt_pixels = int(gt.sum())
    h, w = gt.shape

    if gt_pixels == 0:
        return {
            "gt_pixels": 0.0,
            "outside_pixels": 0.0,
            "outside_ratio_pct": 0.0,
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
    outside_ratio_pct = 100.0 * outside / max(1, gt_pixels)
    coverage_pct = 100.0 - outside_ratio_pct

    return {
        "gt_pixels": float(gt_pixels),
        "outside_pixels": float(outside),
        "outside_ratio_pct": float(outside_ratio_pct),
        "coverage_pct": float(coverage_pct),
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        fieldnames = [
            "split",
            "sample_id",
            "image_path",
            "mask_path",
            "pred_label_path",
            "pred_box_count",
            "gt_pixels",
            "outside_pixels",
            "outside_ratio_pct",
            "coverage_pct",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}

    for split in args.splits:
        split_dir = args.seg_root / split
        mask_dir = split_dir / "masks"
        image_dir = split_dir / "images"
        pred_dir = args.pred_root / split / "labels"

        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing predicted labels directory: {pred_dir}")

        all_rows: List[Dict[str, object]] = []
        leak_rows: List[Dict[str, object]] = []

        for mask_path in sorted(mask_dir.glob("*.png")):
            sample_id = mask_path.stem
            image_path = image_dir / f"{sample_id}.png"
            pred_label_path = pred_dir / f"{sample_id}.txt"

            with Image.open(mask_path) as m:
                arr = np.array(
                    m.convert("L").resize((args.image_size, args.image_size), Image.NEAREST),
                    dtype=np.float32,
                ) / 255.0

            pred_boxes = parse_pred_boxes(pred_label_path, args.image_size)
            cov = coverage(arr, pred_boxes)

            row: Dict[str, object] = {
                "split": split,
                "sample_id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "pred_label_path": str(pred_label_path),
                "pred_box_count": len(pred_boxes),
                "gt_pixels": int(cov["gt_pixels"]),
                "outside_pixels": int(cov["outside_pixels"]),
                "outside_ratio_pct": float(cov["outside_ratio_pct"]),
                "coverage_pct": float(cov["coverage_pct"]),
            }

            all_rows.append(row)
            if row["outside_pixels"] > 0:
                leak_rows.append(row)

        export_rows = all_rows if args.include_all else leak_rows
        kind = "all" if args.include_all else "leak"

        csv_path = args.output_dir / f"{split}_{kind}_examples.csv"
        json_path = args.output_dir / f"{split}_{kind}_examples.json"

        write_csv(csv_path, export_rows)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(export_rows, f, indent=2)

        summary[split] = {
            "total_samples": len(all_rows),
            "leak_samples": len(leak_rows),
            "exported_rows": len(export_rows),
            "csv": str(csv_path),
            "json": str(json_path),
        }

        print(
            f"split={split}: total={len(all_rows)} leak={len(leak_rows)} "
            f"exported={len(export_rows)}"
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
