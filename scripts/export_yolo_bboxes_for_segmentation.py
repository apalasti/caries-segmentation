#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import Any, Optional, Sequence, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.train_tooth_detection_model import build_detector_model, resolve_device

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO predicted boxes for segmentation dataset splits"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("config.toml"),
        help="TOML config file",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
        help="Segmentation dataset splits to export predictions for",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/yolo_pred_bboxes_segmentation"),
        help="Destination directory for predicted labels and metadata",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional score threshold override",
    )
    return parser.parse_args()


def _parse_anchors(raw_anchors: Any) -> Optional[Sequence[Tuple[float, float]]]:
    if raw_anchors is None:
        return None
    anchors: list[Tuple[float, float]] = []
    for pair in raw_anchors:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        w, h = float(pair[0]), float(pair[1])
        if w > 0 and h > 0:
            anchors.append((w, h))
    return anchors if anchors else None


def _resolve_detector_checkpoint(config: dict[str, Any]) -> pathlib.Path:
    det_cfg = config.get("tooth_detection", {})
    explicit = det_cfg.get("detector_checkpoint")
    if explicit:
        return pathlib.Path(explicit)
    return pathlib.Path(det_cfg.get("output_dir", "checkpoints/detection")) / "detector_best.pt"


def _xyxy_to_yolo(box: torch.Tensor, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v.item()) for v in box]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return xc / width, yc / height, bw / width, bh / height


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))

    data_cfg = config.get("data", {})
    det_cfg = config.get("tooth_detection", {})

    preprocessed_path = pathlib.Path(data_cfg.get("preprocessed_path", "data/preprocessed"))
    image_size = int(det_cfg.get("image_size", 640))
    device = resolve_device(str(det_cfg.get("device", "auto")))

    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(det_cfg.get("score_threshold", 0.25))
    )

    detector = build_detector_model(
        device=device,
        anchors=_parse_anchors(det_cfg.get("anchors")),
        conf_threshold=score_threshold,
        nms_iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(det_cfg.get("max_detections", 300)),
    )

    checkpoint_path = _resolve_detector_checkpoint(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    detector.load_state_dict(torch.load(checkpoint_path, map_location=device))
    detector.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for split in args.splits:
            img_dir = preprocessed_path / split / "images"
            if not img_dir.exists():
                raise FileNotFoundError(f"Missing image directory: {img_dir}")

            out_labels = args.output_dir / split / "labels"
            out_labels.mkdir(parents=True, exist_ok=True)

            rows: list[dict[str, str]] = []
            image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS])

            for image_path in image_paths:
                with Image.open(image_path) as img:
                    resized = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)

                tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
                output = detector(tensor)[0]

                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                label_path = out_labels / f"{image_path.stem}.txt"
                with label_path.open("w", encoding="utf-8") as f:
                    for box, score in zip(pred_boxes, pred_scores):
                        xc, yc, bw, bh = _xyxy_to_yolo(box, image_size, image_size)
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {float(score.item()):.6f}\n")

                rows.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "pred_label": str(label_path),
                        "num_boxes": str(int(pred_boxes.shape[0])),
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
                f"Exported predicted YOLO boxes for split={split}: "
                f"images={len(rows)} labels_dir={out_labels} metadata={meta_path}"
            )


if __name__ == "__main__":
    main()
