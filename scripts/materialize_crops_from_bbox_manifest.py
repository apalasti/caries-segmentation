#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize image+mask crops from portable bbox CSV/JSON manifest"
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        required=True,
        help="Path to bbox manifest (.json or .csv)",
    )
    parser.add_argument(
        "--seg-root",
        type=pathlib.Path,
        default=pathlib.Path("data/preprocessed"),
        help="Segmentation dataset root prepared by preprocess.py",
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
        default=None,
        help="Override reference size for normalized bboxes; defaults to manifest metadata if json",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Drop boxes below this score",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.05,
        help="Expand each bbox by this ratio",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum crop width/height",
    )
    parser.add_argument(
        "--fallback-full-image",
        action="store_true",
        help="Emit full-image crop if sample has no valid boxes",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/segmentation_crops_from_manifest"),
    )
    return parser.parse_args()


def _open_text(path: pathlib.Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _manifest_kind(path: pathlib.Path) -> str:
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes:
        raise ValueError("Manifest must have .json/.csv (optionally .gz)")
    if suffixes[-1] == ".gz":
        if len(suffixes) < 2:
            raise ValueError("Compressed manifest must be .json.gz or .csv.gz")
        return suffixes[-2]
    return suffixes[-1]


def load_manifest_json(path: pathlib.Path) -> Tuple[Dict[str, object], int]:
    with _open_text(path) as f:
        d = json.load(f)

    reference_size = int(d.get("reference_image_size", 640))
    by_sample: Dict[str, object] = {}

    for sample in d.get("samples", []):
        split = str(sample["split"])
        sample_id = str(sample["sample_id"])
        boxes = []
        for b in sample.get("boxes", []):
            boxes.append((float(b["xc"]), float(b["yc"]), float(b["w"]), float(b["h"]), float(b.get("score", 1.0))))
        by_sample[f"{split}:{sample_id}"] = {
            "split": split,
            "sample_id": sample_id,
            "boxes": boxes,
        }

    return by_sample, reference_size


def load_manifest_csv(path: pathlib.Path) -> Tuple[Dict[str, object], int]:
    by_sample: Dict[str, object] = defaultdict(lambda: {"split": "", "sample_id": "", "boxes": []})

    with _open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row["split"])
            sample_id = str(row["sample_id"])
            key = f"{split}:{sample_id}"

            box = (
                float(row["xc"]),
                float(row["yc"]),
                float(row["w"]),
                float(row["h"]),
                float(row.get("score", 1.0)),
            )

            by_sample[key]["split"] = split
            by_sample[key]["sample_id"] = sample_id
            by_sample[key]["boxes"].append(box)

    return dict(by_sample), 640


def yolo_to_xyxy(box: Tuple[float, float, float, float, float], image_size: int) -> Tuple[float, float, float, float, float]:
    xc, yc, bw, bh, score = box
    x1 = (xc - bw / 2.0) * image_size
    y1 = (yc - bh / 2.0) * image_size
    x2 = (xc + bw / 2.0) * image_size
    y2 = (yc + bh / 2.0) * image_size
    return x1, y1, x2, y2, score


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


def main() -> None:
    args = parse_args()

    manifest_kind = _manifest_kind(args.manifest)

    if manifest_kind == ".json":
        by_sample, default_size = load_manifest_json(args.manifest)
    elif manifest_kind == ".csv":
        by_sample, default_size = load_manifest_csv(args.manifest)
    else:
        raise ValueError("Manifest must be .json, .csv, .json.gz, or .csv.gz")

    image_size = int(args.image_size) if args.image_size is not None else int(default_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, object]] = {}

    for split in args.splits:
        out_img_dir = args.output_dir / split / "images"
        out_mask_dir = args.output_dir / split / "masks"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        total_images = 0
        total_crops = 0
        no_box_samples = 0

        split_image_dir = args.seg_root / split / "images"
        for image_path in sorted(split_image_dir.glob("*.png")):
            total_images += 1
            sample_id = image_path.stem
            key = f"{split}:{sample_id}"
            mask_path = args.seg_root / split / "masks" / f"{sample_id}.png"

            if key in by_sample:
                boxes_norm = by_sample[key]["boxes"]
            else:
                boxes_norm = []

            boxes_xyxy: List[Tuple[int, int, int, int, float]] = []
            for b in boxes_norm:
                if float(b[4]) < args.score_threshold:
                    continue
                raw = yolo_to_xyxy(b, image_size)
                clipped = expand_and_clip(raw, image_size, image_size, args.padding_ratio, args.min_size)
                if clipped is not None:
                    boxes_xyxy.append(clipped)

            if not boxes_xyxy:
                no_box_samples += 1
                if args.fallback_full_image:
                    boxes_xyxy = [(0, 0, image_size, image_size, -1.0)]
                else:
                    continue

            with Image.open(image_path) as im:
                im_r = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            with Image.open(mask_path) as mm:
                mask_r = mm.convert("L").resize((image_size, image_size), Image.NEAREST)

            for idx, (x1, y1, x2, y2, score) in enumerate(boxes_xyxy):
                crop_id = f"{sample_id}_crop{idx:03d}"
                crop_img_path = out_img_dir / f"{crop_id}.png"
                crop_mask_path = out_mask_dir / f"{crop_id}.png"

                im_r.crop((x1, y1, x2, y2)).save(crop_img_path)
                mask_r.crop((x1, y1, x2, y2)).save(crop_mask_path)
                total_crops += 1

                rows.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "crop_id": crop_id,
                        "source_image": str(image_path),
                        "source_mask": str(mask_path),
                        "crop_image": str(crop_img_path),
                        "crop_mask": str(crop_mask_path),
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

        summary[split] = {
            "total_images": total_images,
            "total_crops": total_crops,
            "no_box_samples": no_box_samples,
            "metadata": str(meta_path),
            "images_dir": str(out_img_dir),
            "masks_dir": str(out_mask_dir),
        }

        print(
            f"split={split}: images={total_images} crops={total_crops} no_box={no_box_samples}"
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
