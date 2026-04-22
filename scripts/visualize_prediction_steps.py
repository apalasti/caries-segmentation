#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.bbox.yolo import YOLOv5
from src.models.bbox.yolo_unet_conjunction import YOLOUNetConjunction
from src.models.end2end import EndToEndCariesModel
from src.models.lightning_model import SegmentationLightningModule
from src.models.unet import UNet
from src.train_end2end import _parse_anchors
from src.train_tooth_detection_model import build_detector_model, resolve_device

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic visualization samples for method1/method2 pipelines"
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=pathlib.Path("config.toml")
    )
    parser.add_argument(
        "--method",
        type=str,
        default="method1",
        choices=["method1", "method2"],
        help="method1: detector (+optional conjunction), method2: true end-to-end model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="segmentation",
        choices=["detection", "segmentation"],
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"]
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs/prediction_steps"),
    )
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save per-sample summary JSON with sanity metrics",
    )
    return parser.parse_args()


def load_gt_boxes(
    label_path: pathlib.Path, width: int, height: int
) -> List[Tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
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
    scores: Optional[torch.Tensor],
    color: Tuple[int, int, int],
    threshold: float,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if boxes.numel() == 0:
        return canvas

    if scores is None:
        score_values = [1.0] * int(boxes.shape[0])
    else:
        score_values = [float(s.item()) for s in scores]

    for box, s in zip(boxes, score_values):
        if s < threshold:
            continue
        x1, y1, x2, y2 = [float(v.item()) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if scores is not None:
            draw.text((x1 + 2, max(0, y1 - 14)), f"{s:.2f}", fill=color)
    return canvas


def draw_gt_boxes(
    image: Image.Image, boxes: List[Tuple[float, float, float, float]]
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    return canvas


def mask_to_boxes(mask_np: np.ndarray) -> List[Tuple[float, float, float, float]]:
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[float, float, float, float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def boxes_list_to_tensor(
    boxes: List[Tuple[float, float, float, float]],
    device: torch.device,
) -> torch.Tensor:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    return torch.tensor(boxes, dtype=torch.float32, device=device)


def mask_coverage_outside_union(mask_np: np.ndarray, boxes: torch.Tensor) -> Dict[str, float]:
    gt = mask_np > 0.5
    gt_pixels = int(gt.sum())
    if gt_pixels == 0:
        return {
            "coverage_pct": 100.0,
            "outside_pixels": 0.0,
            "gt_pixels": 0.0,
        }

    h, w = gt.shape
    inside = np.zeros_like(gt, dtype=np.bool_)

    if boxes.numel() > 0:
        for b in boxes.detach().cpu().numpy():
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            inside[y1:y2, x1:x2] = True

    outside_pixels = int(np.logical_and(gt, np.logical_not(inside)).sum())
    coverage_pct = 100.0 * (1.0 - outside_pixels / max(1, gt_pixels))
    return {
        "coverage_pct": float(coverage_pct),
        "outside_pixels": float(outside_pixels),
        "gt_pixels": float(gt_pixels),
    }


def overlay_mask(
    image: Image.Image, mask: torch.Tensor, color: Tuple[int, int, int]
) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask_np = mask.squeeze().detach().cpu().numpy() if torch.is_tensor(mask) else mask
    mask_bin = (mask_np > 0.5).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color
    alpha = 0.5
    blended = base * (1.0 - alpha * mask_bin[..., None]) + overlay * (
        alpha * mask_bin[..., None]
    )
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def overlay_mask_pil(
    image: Image.Image, mask_img: Image.Image, color: Tuple[int, int, int]
) -> Image.Image:
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


def _resolve_detection_checkpoint(config: Dict) -> pathlib.Path:
    det_cfg = config.get("tooth_detection", {})
    explicit = det_cfg.get("detector_checkpoint")
    if explicit:
        return pathlib.Path(explicit)
    return pathlib.Path(det_cfg.get("output_dir", "checkpoints/detection")) / "detector_best.pt"


def _load_detector(
    config: Dict,
    device: torch.device,
    score_threshold: float,
) -> YOLOv5:
    det_cfg = config.get("tooth_detection", {})
    anchors = _parse_anchors(det_cfg.get("anchors"))
    detector = build_detector_model(
        device=device,
        anchors=anchors,
        conf_threshold=score_threshold,
        nms_iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(det_cfg.get("max_detections", 300)),
    )
    checkpoint = _resolve_detection_checkpoint(config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint}")
    detector.load_state_dict(torch.load(checkpoint, map_location=device))
    detector.eval()
    return detector


def _load_end2end_model(config: Dict, device: torch.device) -> EndToEndCariesModel:
    model_cfg = config.get("model", {})
    det_cfg = config.get("tooth_detection", {})
    end_cfg = config.get("end2end", {})

    detector = YOLOv5(
        num_classes=1,
        anchors=_parse_anchors(det_cfg.get("anchors")),
        conf_threshold=float(det_cfg.get("score_threshold", 0.25)),
        iou_threshold=float(det_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(det_cfg.get("max_detections", 300)),
    )

    segmenter_in_channels = int(end_cfg.get("segmenter_in_channels", model_cfg.get("n_channels", 1)))
    segmenter = UNet(
        n_channels=segmenter_in_channels,
        n_classes=1,
        depth=int(model_cfg.get("depth", 4)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )

    unet_size_raw = end_cfg.get("unet_input_size", config.get("data", {}).get("images_size", [256, 256]))
    model = EndToEndCariesModel(
        detector=detector,
        segmenter=segmenter,
        unet_input_size=(int(unet_size_raw[0]), int(unet_size_raw[1])),
        train_with_gt_boxes_prob=float(end_cfg.get("train_with_gt_boxes_prob", 0.5)),
        mask_threshold=float(end_cfg.get("mask_threshold", 0.5)),
        crop_padding_ratio=float(end_cfg.get("crop_padding_ratio", 0.05)),
        min_crop_size=int(end_cfg.get("min_crop_size", 2)),
        segmenter_in_channels=segmenter_in_channels,
    ).to(device)

    checkpoint_path = pathlib.Path(end_cfg.get("checkpoint", "checkpoints/end2end/end2end_best.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"End-to-end checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def _load_method1_segmenter(config: Dict, checkpoint_path: pathlib.Path) -> torch.nn.Module:
    if checkpoint_path.suffix.lower() == ".ckpt":
        unet_module = SegmentationLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            config=config,
        )
        unet_module.eval()
        return unet_module.model_instance

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported segmenter checkpoint format: {checkpoint_path}")

    normalized_state: Dict[str, torch.Tensor] = {}
    max_legacy_down = 0
    for key, value in state.items():
        clean_key = key[6:] if key.startswith("model.") else key

        m_down = re.match(r"^down(\d+)\.(.+)$", clean_key)
        if m_down:
            down_idx = int(m_down.group(1)) - 1
            max_legacy_down = max(max_legacy_down, down_idx + 1)
            clean_key = f"downs.{down_idx}.{m_down.group(2)}"

        m_up = re.match(r"^up(\d+)\.(.+)$", clean_key)
        if m_up:
            up_idx = int(m_up.group(1)) - 1
            clean_key = f"ups.{up_idx}.{m_up.group(2)}"

        m_up_conv = re.match(r"^up_conv(\d+)\.(.+)$", clean_key)
        if m_up_conv:
            up_conv_idx = int(m_up_conv.group(1)) - 1
            clean_key = f"up_convs.{up_conv_idx}.{m_up_conv.group(2)}"

        normalized_state[clean_key] = value

    model_cfg = config.get("model", {})
    inferred_depth = max_legacy_down + 1 if max_legacy_down > 0 else None
    segmenter = UNet(
        n_channels=int(model_cfg.get("n_channels", 1)),
        n_classes=int(model_cfg.get("n_classes", 1)),
        depth=int(inferred_depth or model_cfg.get("depth", 4)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )
    segmenter.load_state_dict(normalized_state, strict=True)
    segmenter.eval()
    return segmenter


def _resolve_samples(
    items: List[Tuple[pathlib.Path, Optional[pathlib.Path]]],
    index: Optional[int],
    num_samples: int,
    seed: int,
) -> List[Tuple[int, pathlib.Path, Optional[pathlib.Path]]]:
    if not items:
        return []
    if index is not None:
        idx = max(0, min(index, len(items) - 1))
        image_path, aux = items[idx]
        return [(idx, image_path, aux)]

    count = max(1, min(num_samples, len(items)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(items), size=count, replace=False).tolist()
    return [(idx, items[idx][0], items[idx][1]) for idx in chosen]


def _resolve_output_path(
    output_arg: pathlib.Path,
    method: str,
    dataset: str,
    split: str,
    sample_index: int,
    image_stem: str,
    single_only: bool,
) -> pathlib.Path:
    output_ext = output_arg.suffix.lower()
    if single_only and output_ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        return output_arg

    output_dir = output_arg
    if output_ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        output_dir = output_arg.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{method}_{dataset}_{split}_{sample_index:03d}_{image_stem}.png"
    return output_dir / filename


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    det_cfg = config.get("tooth_detection", {})

    if args.method == "method2" and args.dataset != "segmentation":
        raise ValueError("method2 supports only --dataset segmentation")

    image_size = int(det_cfg.get("image_size", 640))
    device = resolve_device(str(det_cfg.get("device", "auto")))
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else float(det_cfg.get("score_threshold", 0.25))
    )

    # Resolve sample items: (image_path, aux_path[label or mask])
    if args.dataset == "detection":
        data_dir = ROOT / "data" / "preprocessed_detection" / args.split
        img_dir = data_dir / "images"
        lbl_dir = data_dir / "labels"
        items = [
            (p, lbl_dir / f"{p.stem}.txt")
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in VALID_EXTS
        ]
    else:
        data_dir = ROOT / "data" / "preprocessed" / args.split
        img_dir = data_dir / "images"
        mask_dir = data_dir / "masks"
        items = [
            (p, mask_dir / f"{p.stem}.png")
            for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in VALID_EXTS
        ]

    samples = _resolve_samples(items, args.index, args.num_samples, args.seed)
    if not samples:
        raise FileNotFoundError(f"No images found for dataset={args.dataset}, split={args.split}")

    detector = None
    conjunction = None
    end2end_model = None

    if args.method == "method1":
        detector = _load_detector(config, device, score_threshold)

        # Conjunction mask is optional and supports both .ckpt and .pth UNet checkpoints.
        preferred_ckpt = pathlib.Path(det_cfg.get("unet_checkpoint", "checkpoints/best_model.ckpt"))
        fallback_ckpt = pathlib.Path(det_cfg.get("unet_state_dict", "checkpoints/best_model.pth"))

        segmenter_model = None
        if preferred_ckpt.exists():
            segmenter_model = _load_method1_segmenter(config, preferred_ckpt)
        elif fallback_ckpt.exists():
            segmenter_model = _load_method1_segmenter(config, fallback_ckpt)

        if segmenter_model is not None:
            unet_input_size = config.get("data", {}).get("images_size", [256, 256])
            conjunction = YOLOUNetConjunction(
                detector=detector,
                segmenter=segmenter_model,
                unet_input_size=(int(unet_input_size[0]), int(unet_input_size[1])),
                mask_threshold=float(det_cfg.get("unet_mask_threshold", 0.5)),
                crop_padding_ratio=float(det_cfg.get("unet_crop_padding", 0.05)),
            ).to(device)
            conjunction.eval()
    else:
        end2end_model = _load_end2end_model(config, device)
        size_raw = config.get("data", {}).get("images_size", [256, 256])
        image_size = int(size_raw[0])

    records: List[Dict[str, float]] = []
    single_only = len(samples) == 1

    for sample_order, (sample_index, image_path, aux_path) in enumerate(samples):
        with Image.open(image_path) as pil_img:
            resized = pil_img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)

        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)

        if args.method == "method1":
            with torch.no_grad():
                if conjunction is not None:
                    out = conjunction(tensor)
                    det_out = out["detections"][0]
                    refined_mask = out["masks"][0].detach().cpu()
                else:
                    det_out = detector(tensor)[0]
                    refined_mask = None
        else:
            with torch.no_grad():
                out = end2end_model(tensor)
                det_out = out["detections"][0]
                refined_mask = out["masks"][0].detach().cpu()

        pred_boxes = det_out["boxes"].detach().cpu()
        pred_scores = det_out["scores"].detach().cpu()

        gt_boxes: List[Tuple[float, float, float, float]] = []
        gt_mask_img: Optional[Image.Image] = None
        gt_mask_np: Optional[np.ndarray] = None

        if args.dataset == "detection":
            gt_boxes = load_gt_boxes(aux_path, image_size, image_size) if aux_path else []
        else:
            if aux_path and aux_path.exists():
                with Image.open(aux_path) as gt_m:
                    gt_mask_img = gt_m.convert("L").resize((image_size, image_size), Image.NEAREST)
                gt_mask_np = np.array(gt_mask_img, dtype=np.float32) / 255.0
                gt_boxes = mask_to_boxes(gt_mask_np)

        panels: List[Image.Image] = []
        panels.append(caption_panel(resized, "Input"))

        gt_panel = resized.copy()
        if gt_mask_img is not None:
            gt_panel = overlay_mask_pil(gt_panel, gt_mask_img, color=(0, 255, 0))
        if gt_boxes:
            gt_panel = draw_gt_boxes(gt_panel, gt_boxes)
        panels.append(caption_panel(gt_panel, "Ground Truth (green)"))

        pred_panel = resized.copy()
        if refined_mask is not None:
            pred_panel = overlay_mask(pred_panel, refined_mask, color=(255, 60, 60))
        pred_panel = draw_boxes(
            pred_panel,
            pred_boxes,
            pred_scores,
            (255, 200, 0),
            score_threshold,
        )
        panels.append(caption_panel(pred_panel, "Prediction (red mask, yellow boxes)"))

        composed = concat_horizontal(panels)
        output_path = _resolve_output_path(
            output_arg=args.output,
            method=args.method,
            dataset=args.dataset,
            split=args.split,
            sample_index=sample_order,
            image_stem=image_path.stem,
            single_only=single_only,
        )
        composed.save(output_path)
        print(f"Saved visualization: {output_path}")

        record: Dict[str, float] = {
            "sample_index": float(sample_index),
            "pred_box_count": float(pred_boxes.shape[0]),
            "gt_box_count": float(len(gt_boxes)),
        }

        if gt_mask_np is not None:
            gt_boxes_tensor = boxes_list_to_tensor(gt_boxes, device=torch.device("cpu"))
            pred_cov = mask_coverage_outside_union(gt_mask_np, pred_boxes)
            gt_cov = mask_coverage_outside_union(gt_mask_np, gt_boxes_tensor)
            record.update(
                {
                    "pred_box_coverage_pct": pred_cov["coverage_pct"],
                    "pred_box_outside_pixels": pred_cov["outside_pixels"],
                    "gt_box_coverage_pct": gt_cov["coverage_pct"],
                    "gt_box_outside_pixels": gt_cov["outside_pixels"],
                }
            )
        records.append(record)

    if args.save_summary or args.num_samples > 1:
        output_ext = args.output.suffix.lower()
        summary_dir = args.output
        if output_ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            summary_dir = args.output.parent
        summary_dir.mkdir(parents=True, exist_ok=True)

        pred_covs = [r["pred_box_coverage_pct"] for r in records if "pred_box_coverage_pct" in r]
        summary = {
            "method": args.method,
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "num_samples": len(records),
            "mean_pred_box_coverage_pct": float(np.mean(pred_covs)) if pred_covs else None,
            "pred_box_leak_samples": float(sum(r.get("pred_box_outside_pixels", 0.0) > 0 for r in records)),
            "records": records,
        }
        summary_path = summary_dir / f"{args.method}_{args.dataset}_{args.split}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
