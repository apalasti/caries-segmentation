import argparse
import csv
import hashlib
import json
import pathlib
import re
import shutil
from dataclasses import dataclass
import logging

from PIL import Image
from tqdm import tqdm


ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "preprocessed_detection"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORMALIZATION_RANGE = [0.0, 1.0]


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess tooth object detection dataset (YOLO labels)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=None,
        help="Path to dataset root with train/valid/test folders. Defaults to latest tooth-p8snt-* under data/raw.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Optional max side length; >0 rescales large images while preserving aspect ratio.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=0.005,
        help="Drop YOLO boxes with normalized width/height smaller than this threshold.",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.02,
        help="Expand each normalized box by this ratio of its width/height before clipping.",
    )
    return parser.parse_args()


def infer_latest_dataset_dir() -> pathlib.Path:
    def has_yolo_layout(root: pathlib.Path) -> bool:
        split_names = ["train", "valid", "val", "test"]
        found_splits = 0
        for split in split_names:
            split_dir = root / split
            if not split_dir.exists():
                continue
            if (split_dir / "images").exists() and (split_dir / "labels").exists():
                found_splits += 1
        return found_splits >= 2

    explicit_patterns = [
        "tooth-p8snt-*",
        "tooth-*",
        "Tooth-*",
        "teeth_*",
    ]
    candidates: list[pathlib.Path] = []
    for pattern in explicit_patterns:
        candidates.extend([p for p in RAW_DIR.glob(pattern) if p.is_dir() and has_yolo_layout(p)])

    # Also include any other Roboflow download folder that follows YOLO split layout.
    for p in RAW_DIR.iterdir():
        if p.is_dir() and has_yolo_layout(p):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "No YOLO detection dataset directory found under data/raw with train/val|valid/test images+labels layout."
        )

    def version_key(path: pathlib.Path) -> int:
        match = re.search(r"-(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    # De-duplicate, then prefer highest version suffix and latest modification time.
    unique_candidates = list({p.resolve(): p for p in candidates}.values())
    return sorted(
        unique_candidates,
        key=lambda p: (version_key(p), p.stat().st_mtime),
    )[-1]


def parse_yolo_labels(label_path: pathlib.Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logging.warning(f"Invalid label line (not enough parts): {line}")
                continue

            class_id = int(float(parts[0]))

            # YOLO bbox format: class xc yc w h
            if len(parts) == 5:
                _, x_center, y_center, width, height = parts
                boxes.append(
                    YoloBox(
                        class_id=class_id,
                        x_center=float(x_center),
                        y_center=float(y_center),
                        width=float(width),
                        height=float(height),
                    )
                )
                continue

            # YOLO polygon format: class x1 y1 x2 y2 ... -> convert polygon to bbox
            coords = [float(v) for v in parts[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                logging.warning(
                    f"Invalid polygon label (not enough points or odd number of coords): {line}"
                )
                continue

            xs = coords[::2]
            ys = coords[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            x_center = (min_x + max_x) / 2.0
            y_center = (min_y + max_y) / 2.0
            width = max_x - min_x
            height = max_y - min_y

            boxes.append(
                YoloBox(
                    class_id=class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            )
    return boxes


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sanitize_boxes(
    boxes: list[YoloBox], min_box_size: float, box_padding: float
) -> list[YoloBox]:
    cleaned: list[YoloBox] = []
    for b in boxes:
        w = clip(b.width, 0.0, 1.0)
        h = clip(b.height, 0.0, 1.0)
        w = clip(w * (1.0 + box_padding), 0.0, 1.0)
        h = clip(h * (1.0 + box_padding), 0.0, 1.0)

        xc = clip(b.x_center, 0.0, 1.0)
        yc = clip(b.y_center, 0.0, 1.0)
        if w < min_box_size or h < min_box_size:
            continue
        cleaned.append(
            YoloBox(
                class_id=0,
                x_center=xc,
                y_center=yc,
                width=w,
                height=h,
            )
        )
    return cleaned


def resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, Image.BILINEAR)


def write_labels(label_path: pathlib.Path, boxes: list[YoloBox]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for b in boxes:
            f.write(
                f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}\n"
            )


def write_metadata(
    output_dir: pathlib.Path,
    dataset_dir: pathlib.Path,
    max_side: int,
    min_box_size: float,
    box_padding: float,
    split_stats: dict[str, dict[str, int]],
) -> None:
    metadata = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_dir.name,
        "resize": {
            "policy": "preserve_aspect_ratio_max_side",
            "max_side": max_side,
        },
        "normalization": {
            "range": NORMALIZATION_RANGE,
            "method": "tensor_scaling_after_pillow_load",
        },
        "label_filtering": {
            "min_box_size": min_box_size,
            "box_padding": box_padding,
        },
        "splits": split_stats,
    }

    metadata_path = output_dir / "preprocessing_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Wrote preprocessing metadata: {metadata_path}")


def process_split(
    dataset_dir: pathlib.Path,
    src_split: str,
    dst_split: str,
    max_side: int,
    min_box_size: float,
    box_padding: float,
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    src_images = dataset_dir / src_split / "images"
    src_labels = dataset_dir / src_split / "labels"

    dst_images = OUT_DIR / dst_split / "images"
    dst_labels = OUT_DIR / dst_split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    kept_boxes = 0

    image_paths = sorted(
        [p for p in src_images.iterdir() if p.suffix.lower() in VALID_EXTS]
    )
    for image_path in tqdm(image_paths, desc=f"Preprocessing {src_split}"):
        label_path = src_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            logging.warning(f"Label file not found for image: {image_path}")
            continue

        boxes = sanitize_boxes(parse_yolo_labels(label_path), min_box_size, box_padding)
        if not boxes:
            skipped += 1
            logging.info(f"No valid boxes for image (skipping): {image_path}")
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            out_img = resize_keep_aspect(img, max_side)

        out_img_path = dst_images / f"{image_path.stem}.jpg"
        out_lbl_path = dst_labels / f"{image_path.stem}.txt"

        out_img.save(out_img_path, format="JPEG", quality=95)
        write_labels(out_lbl_path, boxes)

        rows.append(
            {
                "split": dst_split,
                "image": str(out_img_path.relative_to(ROOT)),
                "label": str(out_lbl_path.relative_to(ROOT)),
                "num_boxes": str(len(boxes)),
                "width": str(out_img.size[0]),
                "height": str(out_img.size[1]),
            }
        )

        processed += 1
        kept_boxes += len(boxes)

    return processed, skipped, kept_boxes


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir else infer_latest_dataset_dir()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    # Determinisztikus split logika a fájlnévből kiindulva (Ahogy a preprocess.py-ban)
    def deterministic_split(id_str: str, train: float = 0.7, val: float = 0.15) -> str:
        u = int(id_str[:8], 16) / (16**8)
        if u < train:
            return "train"
        if u < train + val:
            return "val"
        return "test"

    totals = {"train": [0, 0, 0], "val": [0, 0, 0], "test": [0, 0, 0]}
    split_stats: dict[str, dict[str, int]] = {}

    src_splits = ["train", "valid", "val", "test"]
    all_images = []
    for s in src_splits:
        img_dir = dataset_dir / s / "images"
        if img_dir.exists():
            all_images.extend(
                p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS
            )

    for dst in totals.keys():
        (OUT_DIR / dst / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / dst / "labels").mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(all_images, desc="Preprocessing images"):
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # Hash split kiszámítása a fájlnév (stem) alapján
        hash_val = hashlib.md5(image_path.name.encode()).hexdigest()
        dst_split = deterministic_split(hash_val)

        boxes = sanitize_boxes(
            parse_yolo_labels(label_path), args.min_box_size, args.box_padding
        )
        if not boxes:
            totals[dst_split][1] += 1
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            out_img = resize_keep_aspect(img, args.max_side)

        dst_images = OUT_DIR / dst_split / "images"
        dst_labels = OUT_DIR / dst_split / "labels"

        out_img_path = dst_images / f"{image_path.stem}.jpg"
        out_lbl_path = dst_labels / f"{image_path.stem}.txt"

        out_img.save(out_img_path, format="JPEG", quality=95)
        write_labels(out_lbl_path, boxes)

        rows.append(
            {
                "split": dst_split,
                "image": str(out_img_path.relative_to(ROOT)),
                "label": str(out_lbl_path.relative_to(ROOT)),
                "num_boxes": str(len(boxes)),
                "width": str(out_img.size[0]),
                "height": str(out_img.size[1]),
            }
        )

        totals[dst_split][0] += 1
        totals[dst_split][2] += len(boxes)

    metadata_path = OUT_DIR / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "image", "label", "num_boxes", "width", "height"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Detection preprocessing complete: {OUT_DIR}")
    for split, (processed, skipped, boxes) in totals.items():
        print(f"- {split}: processed={processed}, skipped={skipped}, boxes={boxes}")

    for split, (processed, skipped, boxes) in totals.items():
        split_stats[split] = {
            "processed": processed,
            "skipped": skipped,
            "boxes": boxes,
        }

    write_metadata(
        output_dir=OUT_DIR,
        dataset_dir=dataset_dir,
        max_side=args.max_side,
        min_box_size=args.min_box_size,
        box_padding=args.box_padding,
        split_stats=split_stats,
    )


if __name__ == "__main__":
    main()
