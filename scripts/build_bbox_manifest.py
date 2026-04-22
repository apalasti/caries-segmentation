#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Dict, List

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build portable CSV/JSON manifest from predicted YOLO bbox labels"
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=pathlib.Path("data/preprocessed"),
        help="Segmentation dataset root used to enumerate sample IDs",
    )
    parser.add_argument(
        "--pred-root",
        type=pathlib.Path,
        default=pathlib.Path("outputs/yolo_pred_bboxes_segmentation"),
        help="Predicted bbox root with split/labels/*.txt",
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
        help="Coordinate reference size used by predicted labels",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Drop boxes below this score while building manifest",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=pathlib.Path("outputs/bbox_manifest_predicted.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=pathlib.Path("outputs/bbox_manifest_predicted.csv"),
    )
    parser.add_argument(
        "--include-empty-samples",
        action="store_true",
        help="Include samples with zero predicted boxes in JSON manifest",
    )
    return parser.parse_args()


def parse_label_file(label_path: pathlib.Path, score_threshold: float) -> List[Dict[str, float]]:
    boxes: List[Dict[str, float]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            score = float(parts[5]) if len(parts) >= 6 else 1.0
            if score < score_threshold:
                continue
            boxes.append(
                {
                    "class_id": class_id,
                    "xc": xc,
                    "yc": yc,
                    "w": bw,
                    "h": bh,
                    "score": score,
                }
            )
    return boxes


def main() -> None:
    args = parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    samples_json: List[Dict[str, object]] = []
    rows_csv: List[Dict[str, object]] = []

    summary: Dict[str, Dict[str, int]] = {}

    for split in args.splits:
        image_dir = args.seg_root / split / "images"
        label_dir = args.pred_root / split / "labels"

        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in VALID_EXTS])

        split_samples = 0
        split_boxes = 0
        split_empty = 0

        for image_path in image_paths:
            sample_id = image_path.stem
            label_path = label_dir / f"{sample_id}.txt"
            boxes = parse_label_file(label_path, args.score_threshold)

            if not boxes:
                split_empty += 1

            if boxes or args.include_empty_samples:
                samples_json.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "image_rel": f"{split}/images/{sample_id}.png",
                        "mask_rel": f"{split}/masks/{sample_id}.png",
                        "pred_label_rel": f"{split}/labels/{sample_id}.txt",
                        "box_count": len(boxes),
                        "boxes": boxes,
                    }
                )

            for idx, b in enumerate(boxes):
                rows_csv.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "box_index": idx,
                        "class_id": b["class_id"],
                        "xc": b["xc"],
                        "yc": b["yc"],
                        "w": b["w"],
                        "h": b["h"],
                        "score": b["score"],
                        "image_rel": f"{split}/images/{sample_id}.png",
                        "mask_rel": f"{split}/masks/{sample_id}.png",
                        "pred_label_rel": f"{split}/labels/{sample_id}.txt",
                        "box_count": len(boxes),
                    }
                )

            split_samples += 1
            split_boxes += len(boxes)

        summary[split] = {
            "samples": split_samples,
            "boxes": split_boxes,
            "empty_samples": split_empty,
        }

    manifest = {
        "schema_version": "1.0",
        "kind": "predicted_tooth_bboxes",
        "source": "yolo_pred_bboxes_segmentation",
        "coordinate_format": "yolo_xywh_normalized",
        "reference_image_size": args.image_size,
        "no_augmentation": True,
        "splits": args.splits,
        "score_threshold": args.score_threshold,
        "summary": summary,
        "samples": samples_json,
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    fields = [
        "split",
        "sample_id",
        "box_index",
        "class_id",
        "xc",
        "yc",
        "w",
        "h",
        "score",
        "image_rel",
        "mask_rel",
        "pred_label_rel",
        "box_count",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_csv)

    print(f"Saved JSON manifest: {args.output_json}")
    print(f"Saved CSV manifest: {args.output_csv}")
    for split in args.splits:
        s = summary[split]
        print(
            f"split={split}: samples={s['samples']} boxes={s['boxes']} "
            f"empty_samples={s['empty_samples']}"
        )


if __name__ == "__main__":
    main()
