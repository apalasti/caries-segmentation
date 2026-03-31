#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.bbox.yolo_unet_conjunction import YOLOUNetConjunction
from src.models.lightning_model import SegmentationLightningModule
from src.train_tooth_detection_model import build_detector_model, resolve_device

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes and optional U-Net refinement masks"
    )
    parser.add_argument("--config", type=pathlib.Path, default=pathlib.Path("config.toml"))
    parser.add_argument("--image", type=pathlib.Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs/prediction_steps.png"))
    parser.add_argument("--with-unet", action="store_true", help="Enable YOLO + U-Net conjunction")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override score threshold used in visualization",
    )
    return parser.parse_args()


def select_image_path(args: argparse.Namespace, config: dict) -> pathlib.Path:
    if args.image is not None:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        return args.image

    detection_cfg = config.get("tooth_detection", {})
    data_dir = pathlib.Path(detection_cfg.get("data_dir", "data/preprocessed_detection"))
    images_dir = data_dir / args.split / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    idx = max(0, min(args.index, len(image_paths) - 1))
    return image_paths[idx]


def load_gt_boxes(image_path: pathlib.Path, width: int, height: int) -> List[Tuple[float, float, float, float]]:
    labels_dir = image_path.parent.parent / "labels"
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []

    boxes: List[Tuple[float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    color: Tuple[int, int, int],
    score_threshold: float,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for box, score in zip(boxes, scores):
        score_val = float(score.item())
        if score_val < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v.item()) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{score_val:.2f}", fill=color)

    return canvas


def draw_gt_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    return canvas


def overlay_mask(image: Image.Image, mask: torch.Tensor, color: Tuple[int, int, int] = (255, 80, 80)) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_bin = (mask_np > 0.5).astype(np.float32)

    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    alpha = 0.45
    blended = base * (1.0 - alpha * mask_bin[..., None]) + overlay * (alpha * mask_bin[..., None])
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def caption_panel(image: Image.Image, text: str) -> Image.Image:
    w, h = image.size
    out = Image.new("RGB", (w, h + 28), (20, 20, 20))
    out.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((8, h + 6), text, fill=(230, 230, 230))
    return out


def concat_horizontal(images: List[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), (0, 0, 0))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    detection_cfg = config.get("tooth_detection", {})

    image_size = int(detection_cfg.get("image_size", 640))
    device = resolve_device(str(detection_cfg.get("device", "auto")))

    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(detection_cfg.get("score_threshold", 0.25))
    )

    anchors = detection_cfg.get("anchors")
    parsed_anchors = None
    if anchors:
        parsed_anchors = []
        for pair in anchors:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                parsed_anchors.append((float(pair[0]), float(pair[1])))

    detector = build_detector_model(
        device=device,
        anchors=parsed_anchors,
        conf_threshold=score_threshold,
        nms_iou_threshold=float(detection_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(detection_cfg.get("max_detections", 300)),
    )

    ckpt_dir = pathlib.Path(detection_cfg.get("output_dir", "checkpoints/detection"))
    det_ckpt = ckpt_dir / "detector_best.pt"
    if not det_ckpt.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {det_ckpt}")

    state = torch.load(det_ckpt, map_location=device)
    detector.load_state_dict(state)
    detector.eval()

    image_path = select_image_path(args, config)

    with Image.open(image_path) as pil_img:
        rgb = pil_img.convert("RGB")
        resized = rgb.resize((image_size, image_size), Image.BILINEAR)

    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        det_out = detector(tensor)[0]

    pred_boxes = det_out["boxes"].detach().cpu()
    pred_scores = det_out["scores"].detach().cpu()

    gt_boxes = load_gt_boxes(image_path, image_size, image_size)

    panels: List[Image.Image] = []
    panels.append(caption_panel(resized, "Step 1: Resized Input"))
    if gt_boxes:
        panels.append(caption_panel(draw_gt_boxes(resized, gt_boxes), "Step 2: Ground Truth Boxes"))

    det_panel = draw_boxes(resized, pred_boxes, pred_scores, (255, 200, 0), score_threshold)
    panels.append(caption_panel(det_panel, "Step 3: YOLO Predicted Boxes"))

    if args.with_unet:
        unet_ckpt = pathlib.Path(detection_cfg.get("unet_checkpoint", ""))
        if not unet_ckpt.exists():
            raise FileNotFoundError(f"U-Net checkpoint not found: {unet_ckpt}")

        unet_module = SegmentationLightningModule.load_from_checkpoint(
            str(unet_ckpt),
            config=config,
        )
        unet_module.eval()

        unet_input_size = config.get("data", {}).get("images_size", [256, 256])
        conj = YOLOUNetConjunction(
            detector=detector,
            segmenter=unet_module.model_instance,
            unet_input_size=(int(unet_input_size[0]), int(unet_input_size[1])),
            mask_threshold=float(detection_cfg.get("unet_mask_threshold", 0.5)),
            crop_padding_ratio=float(detection_cfg.get("unet_crop_padding", 0.05)),
        ).to(device)
        conj.eval()

        with torch.no_grad():
            conj_out = conj(tensor)

        refined_mask = conj_out["masks"][0].detach().cpu()
        overlay = overlay_mask(resized, refined_mask)
        panels.append(caption_panel(overlay, "Step 4: U-Net Refinement In YOLO Regions"))

    composed = concat_horizontal(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    composed.save(args.output)

    print(f"Saved visualization: {args.output}")
    print(f"Source image: {image_path}")


if __name__ == "__main__":
    main()
