with open("scripts/visualize_prediction_steps.py", "w") as f:
    f.write("""#!/usr/bin/env python3
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
    parser.add_argument("--dataset", type=str, default="segmentation", choices=["detection", "segmentation"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("outputs/prediction_steps.png"))
    parser.add_argument("--score-threshold", type=float, default=None)
    return parser.parse_args()

def load_gt_boxes(label_path: pathlib.Path, width: int, height: int) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            _, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

def draw_boxes(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, color: Tuple[int, int, int], threshold: float) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box, score in zip(boxes, scores):
        s = float(score.item())
        if s < threshold: continue
        x1, y1, x2, y2 = [float(v.item()) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{s:.2f}", fill=color)
    return canvas

def draw_gt_boxes(image: Image.Image, boxes: List[Tuple[float, float, float, float]]) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    return canvas

def overlay_mask(image: Image.Image, mask: torch.Tensor, color: Tuple[int, int, int]) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask_np = mask.squeeze().detach().cpu().numpy() if torch.is_tensor(mask) else mask
    mask_bin = (mask_np > 0.5).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color
    alpha = 0.5
    blended = base * (1.0 - alpha * mask_bin[..., None]) + overlay * (alpha * mask_bin[..., None])
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def overlay_mask_pil(image: Image.Image, mask_img: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
    mask_np = np.array(mask_img.convert("L")) / 255.0
    return overlay_mask(image, mask_np, color)

def caption_panel(image: Image.Image, text: str) -> Image.Image:
    w, h = image.size
    out = Image.new("RGB", (w, h + 28), (20, 20, 20))
    out.paste(image.convert("RGB"), (0, 0))
    ImageDraw.Draw(out).text((8, h + 6), text, fill=(230, 230, 230))
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
    det_cfg = config.get("tooth_detection", {})

    image_size = int(det_cfg.get("image_size", 640))
    device = resolve_device(str(det_cfg.get("device", "auto")))
    score_threshold = args.score_threshold if args.score_threshold else float(det_cfg.get("score_threshold", 0.25))

    # Resolve Dataset Directory
    if args.dataset == "detection":
        data_dir = ROOT / "data" / "preprocessed_detection"
    else:
        data_dir = ROOT / "data" / "preprocessed"

    img_dir = data_dir / args.split / "images"
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    if not image_paths: raise FileNotFoundError(f"No images found: {img_dir}")
    image_path = image_paths[max(0, min(args.index, len(image_paths) - 1))]

    # Load Source Image
    with Image.open(image_path) as pil_img:
        resized = pil_img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)

    # Detector
    anchors = det_cfg.get("anchors")
    parsed_anchors = [(float(p[0]), float(p[1])) for p in anchors] if anchors else None
    detector = build_detector_model(
        device=device, anchors=parsed_anchors, conf_threshold=score_threshold,
        nms_iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(det_cfg.get("max_detections", 300))
    )
    det_ckpt = pathlib.Path(det_cfg.get("output_dir", "checkpoints/detection")) / "detector_best.pt"
    if det_ckpt.exists():
        detector.load_state_dict(torch.load(det_ckpt, map_location=device))
    detector.eval()

    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        det_out = detector(tensor)[0]
    pred_boxes = det_out["boxes"].detach().cpu()
    pred_scores = det_out["scores"].detach().cpu()

    # Load UNet
    unet_ckpt = pathlib.Path(det_cfg.get("unet_checkpoint", "checkpoints/best_model.ckpt"))
    unet_module = None
    if unet_ckpt.exists():
        unet_module = SegmentationLightningModule.load_from_checkpoint(str(unet_ckpt), config=config)
        unet_module.eval()
        unet_input_size = config.get("data", {}).get("images_size", [256, 256])
        conj = YOLOUNetConjunction(
            detector=detector, segmenter=unet_module.model_instance,
            unet_input_size=(int(unet_input_size[0]), int(unet_input_size[1])),
            mask_threshold=float(det_cfg.get("unet_mask_threshold", 0.5)),
            crop_padding_ratio=float(det_cfg.get("unet_crop_padding", 0.05)),
        ).to(device)
        conj.eval()
        with torch.no_grad():
            refined_mask = conj(tensor)["masks"][0].detach().cpu()
    else:
        refined_mask = None

    # Load GT
    gt_box_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    gt_mask_path = image_path.parent.parent / "masks" / f"{image_path.stem}.png"
    
    panels: List[Image.Image] = []
    
    # Panel 1: Original
    panels.append(caption_panel(resized, "Step 1: Input"))

    # Panel 2: Ground Truths (Boxes & True Pixels)
    gt_panel = resized.copy()
    has_gt = False
    if gt_mask_path.exists():
        with Image.open(gt_mask_path) as gt_m:
            gt_mask_img = gt_m.resize((image_size, image_size), Image.NEAREST)
        gt_panel = overlay_mask_pil(gt_panel, gt_mask_img, color=(0, 255, 0)) # Green GT Pixels
        has_gt = True

    gt_boxes = load_gt_boxes(gt_box_path, image_size, image_size)
    if gt_boxes:
        gt_panel = draw_gt_boxes(gt_panel, gt_boxes) # Green GT Boxes
        has_gt = True
        
    if has_gt:
        panels.append(caption_panel(gt_panel, "Step 2: Ground Truth (Green Pixels/Boxes)"))

    # Panel 3: Predictions (Predicted Boxes & Predicted Pixels)
    pred_panel = resized.copy()
    if refined_mask is not None:
        pred_panel = overlay_mask(pred_panel, refined_mask, color=(255, 60, 60)) # Red Predicted Pixels
    
    pred_panel = draw_boxes(pred_panel, pred_boxes, pred_scores, (255, 200, 0), score_threshold) # Yellow Pred Boxes
    panels.append(caption_panel(pred_panel, "Step 3: Predictions (Red Pixels, Yellow Boxes)"))
    
    composed = concat_horizontal(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    composed.save(args.output)
    print(f"Saved visualization: {args.output}")

if __name__ == '__main__':
    main()
""")
