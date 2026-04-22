from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from .data.dataset import BaseKariesDataset, load_split_pairs
from .models.bbox.yolo import YOLOv5
from .models.end2end import EndToEndCariesModel
from .models.unet import UNet
from .train_tooth_detection_model import resolve_device

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "checkpoints" / "end2end"


def collate_end2end(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


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


def _move_targets_to_device(
    targets: list[Dict[str, torch.Tensor]],
    device: torch.device,
) -> list[Dict[str, torch.Tensor]]:
    moved: list[Dict[str, torch.Tensor]] = []
    for t in targets:
        moved.append({k: v.to(device) for k, v in t.items()})
    return moved


def _build_joint_model(config: Dict[str, Any], device: torch.device) -> EndToEndCariesModel:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    detection_cfg = config.get("tooth_detection", {})
    end2end_cfg = config.get("end2end", {})

    anchors = _parse_anchors(detection_cfg.get("anchors"))
    detector = YOLOv5(
        num_classes=1,
        anchors=anchors,
        conf_threshold=float(detection_cfg.get("score_threshold", 0.25)),
        iou_threshold=float(detection_cfg.get("nms_iou_threshold", 0.45)),
        max_detections=int(detection_cfg.get("max_detections", 300)),
    )

    segmenter_in_channels = int(end2end_cfg.get("segmenter_in_channels", model_cfg.get("n_channels", 1)))
    segmenter = UNet(
        n_channels=segmenter_in_channels,
        n_classes=1,
        depth=int(model_cfg.get("depth", 4)),
        base_channels=int(model_cfg.get("base_channels", 64)),
    )

    unet_size_raw = end2end_cfg.get("unet_input_size", config.get("data", {}).get("images_size", [256, 256]))
    unet_input_size = (int(unet_size_raw[0]), int(unet_size_raw[1]))

    model = EndToEndCariesModel(
        detector=detector,
        segmenter=segmenter,
        unet_input_size=unet_input_size,
        train_with_gt_boxes_prob=float(end2end_cfg.get("train_with_gt_boxes_prob", 0.5)),
        mask_threshold=float(end2end_cfg.get("mask_threshold", detection_cfg.get("unet_mask_threshold", 0.5))),
        crop_padding_ratio=float(end2end_cfg.get("crop_padding_ratio", detection_cfg.get("unet_crop_padding", 0.05))),
        min_crop_size=int(end2end_cfg.get("min_crop_size", 2)),
        segmenter_in_channels=segmenter_in_channels,
        max_rois_per_image=end2end_cfg.get("max_rois_per_image", 32),
    ).to(device)
    return model


def _evaluate_joint_losses(
    model: EndToEndCariesModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.train()

    loss_values: list[float] = []
    det_values: list[float] = []
    seg_values: list[float] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = _move_targets_to_device(targets, device)
            losses = model(images, targets)
            loss_values.append(float(losses["loss"].item()))
            det_values.append(float(losses.get("detector_loss", images.new_tensor(0.0)).item()))
            seg_values.append(float(losses.get("unet_loss", images.new_tensor(0.0)).item()))

    if not was_training:
        model.eval()

    return {
        "loss": float(np.mean(loss_values)) if loss_values else 0.0,
        "detector_loss": float(np.mean(det_values)) if det_values else 0.0,
        "unet_loss": float(np.mean(seg_values)) if seg_values else 0.0,
    }


def _load_checkpoint(model: EndToEndCariesModel, checkpoint_path: pathlib.Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return
    model.load_state_dict(state)


def train_end2end(
    config: Dict[str, Any],
    *,
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
) -> None:
    training_cfg = config.get("training", {})
    end2end_cfg = config.get("end2end", {})

    device = resolve_device(str(end2end_cfg.get("device", training_cfg.get("device", "auto"))))
    seed = int(training_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    size_raw = config.get("data", {}).get("images_size", [256, 256])
    size = (int(size_raw[0]), int(size_raw[1]))
    bbox_padding = float(end2end_cfg.get("bbox_padding", 10.0))

    train_ds = BaseKariesDataset(
        train_pairs,
        size=size,
        bbox_padding=bbox_padding,
        return_targets=True,
    )
    val_ds = BaseKariesDataset(
        val_pairs,
        size=size,
        bbox_padding=bbox_padding,
        return_targets=True,
    )

    if len(train_ds) == 0:
        raise ValueError("No train samples found for end-to-end training.")
    if len(val_ds) == 0:
        raise ValueError("No val samples found for end-to-end training.")

    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 0))
    shuffle = bool(training_cfg.get("shuffle_train", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_end2end,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_end2end,
    )

    model = _build_joint_model(config, device)

    learning_rate = float(training_cfg.get("learning_rate", 5e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epochs = int(training_cfg.get("epochs", 20))
    output_dir = pathlib.Path(end2end_cfg.get("output_dir", DEFAULT_OUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_det_losses: list[float] = []
        train_seg_losses: list[float] = []

        for images, targets in train_loader:
            images = images.to(device)
            targets = _move_targets_to_device(targets, device)

            losses = model(images, targets)
            loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_det_losses.append(float(losses.get("detector_loss", images.new_tensor(0.0)).item()))
            train_seg_losses.append(float(losses.get("unet_loss", images.new_tensor(0.0)).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_det = float(np.mean(train_det_losses)) if train_det_losses else 0.0
        train_seg = float(np.mean(train_seg_losses)) if train_seg_losses else 0.0

        val_metrics = _evaluate_joint_losses(model, val_loader, device)
        val_loss = val_metrics["loss"]

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_det={train_det:.4f} train_unet={train_seg:.4f} "
            f"val_loss={val_loss:.4f} val_det={val_metrics['detector_loss']:.4f} "
            f"val_unet={val_metrics['unet_loss']:.4f}"
        )

        last_ckpt = output_dir / "end2end_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            last_ckpt,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = output_dir / "end2end_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt,
            )
            print(f"Saved best checkpoint: {best_ckpt}")

    print(f"End-to-end training complete. Best val_loss={best_val:.4f}")


def _compute_image_detection_counts(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]

    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0, 0.0
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0]), 0.0
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0, 0.0

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    tp = 0
    iou_sum = 0.0

    order = torch.argsort(ious.max(dim=1).values, descending=True)
    for pred_idx in order.tolist():
        gt_idx = int(torch.argmax(ious[pred_idx]).item())
        best_iou = float(ious[pred_idx, gt_idx].item())
        if best_iou >= iou_threshold and gt_idx not in matched_gt:
            matched_gt.add(gt_idx)
            tp += 1
            iou_sum += best_iou

    fp = int(pred_boxes.shape[0]) - tp
    fn = int(gt_boxes.shape[0]) - tp
    return tp, fp, fn, iou_sum


def _mask_bbox_coverage(gt_mask: torch.Tensor, boxes: torch.Tensor) -> Dict[str, float]:
    gt = (gt_mask > 0.5).detach().cpu().numpy().astype(np.bool_)
    gt_pixels = int(gt.sum())
    if gt_pixels == 0:
        return {
            "coverage_pct": 100.0,
            "outside_pixels": 0,
            "gt_pixels": 0,
        }

    h, w = gt.shape
    inside = np.zeros_like(gt, dtype=np.bool_)

    if boxes.numel() > 0:
        for box in boxes.detach().cpu().numpy():
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            inside[y1:y2, x1:x2] = True

    outside_pixels = int(np.logical_and(gt, np.logical_not(inside)).sum())
    coverage_pct = 100.0 * (1.0 - (outside_pixels / max(1, gt_pixels)))
    return {
        "coverage_pct": float(coverage_pct),
        "outside_pixels": outside_pixels,
        "gt_pixels": gt_pixels,
    }


def evaluate_end2end(
    config: Dict[str, Any],
    *,
    data_pairs: list[tuple[str, str]],
    split: str,
) -> Dict[str, float]:
    training_cfg = config.get("training", {})
    detection_cfg = config.get("tooth_detection", {})
    end2end_cfg = config.get("end2end", {})

    device = resolve_device(str(end2end_cfg.get("device", training_cfg.get("device", "auto"))))

    size_raw = config.get("data", {}).get("images_size", [256, 256])
    size = (int(size_raw[0]), int(size_raw[1]))
    bbox_padding = float(end2end_cfg.get("bbox_padding", 10.0))

    dataset = BaseKariesDataset(
        data_pairs,
        size=size,
        bbox_padding=bbox_padding,
        return_targets=True,
    )
    if len(dataset) == 0:
        raise ValueError(f"No samples found for split '{split}' in end-to-end evaluation.")

    dataloader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collate_end2end,
    )

    model = _build_joint_model(config, device)

    checkpoint_path = pathlib.Path(end2end_cfg.get("checkpoint", pathlib.Path(end2end_cfg.get("output_dir", DEFAULT_OUT_DIR)) / "end2end_best.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"End-to-end checkpoint not found: {checkpoint_path}. "
            "Train with training.task='yolo_unet_conjunction' or 'end2end_joint' first."
        )
    _load_checkpoint(model, checkpoint_path, device)
    model.eval()

    score_threshold = float(detection_cfg.get("score_threshold", 0.25))
    iou_threshold = float(detection_cfg.get("eval_iou_threshold", 0.5))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0

    seg_tp = 0
    seg_fp = 0
    seg_fn = 0

    sanity_records: list[Dict[str, Any]] = []

    with torch.no_grad():
        sample_index = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = _move_targets_to_device(targets, device)

            outputs = model(images)
            detections = outputs["detections"]
            pred_masks = outputs["masks"]

            for b, (det, target) in enumerate(zip(detections, targets)):
                pred_boxes = det["boxes"].detach().cpu()
                pred_scores = det["scores"].detach().cpu()
                gt_boxes = target["boxes"].detach().cpu()

                tp, fp, fn, iou_sum = _compute_image_detection_counts(
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_iou_sum += iou_sum

                pred_mask_bin = pred_masks[b, 0].detach().cpu() > 0.5
                gt_mask_bin = target["masks"][0].detach().cpu() > 0.5

                seg_tp += int(torch.logical_and(pred_mask_bin, gt_mask_bin).sum().item())
                seg_fp += int(torch.logical_and(pred_mask_bin, torch.logical_not(gt_mask_bin)).sum().item())
                seg_fn += int(torch.logical_and(torch.logical_not(pred_mask_bin), gt_mask_bin).sum().item())

                pred_cov = _mask_bbox_coverage(target["masks"][0].detach().cpu(), pred_boxes)
                gt_cov = _mask_bbox_coverage(target["masks"][0].detach().cpu(), gt_boxes)
                sanity_records.append(
                    {
                        "sample_index": sample_index,
                        "pred_box_count": int(pred_boxes.shape[0]),
                        "gt_box_count": int(gt_boxes.shape[0]),
                        "pred_box_coverage_pct": pred_cov["coverage_pct"],
                        "pred_box_outside_pixels": pred_cov["outside_pixels"],
                        "gt_box_coverage_pct": gt_cov["coverage_pct"],
                        "gt_box_outside_pixels": gt_cov["outside_pixels"],
                    }
                )
                sample_index += 1

    det_precision = total_tp / max(1, total_tp + total_fp)
    det_recall = total_tp / max(1, total_tp + total_fn)
    det_f1 = 2.0 * det_precision * det_recall / max(1e-8, det_precision + det_recall)
    det_mean_iou_tp = total_iou_sum / max(1, total_tp)

    seg_precision = seg_tp / max(1, seg_tp + seg_fp)
    seg_recall = seg_tp / max(1, seg_tp + seg_fn)
    seg_dice = 2.0 * seg_tp / max(1, 2 * seg_tp + seg_fp + seg_fn)
    seg_iou = seg_tp / max(1, seg_tp + seg_fp + seg_fn)

    pred_covs = [r["pred_box_coverage_pct"] for r in sanity_records]
    gt_covs = [r["gt_box_coverage_pct"] for r in sanity_records]

    metrics: Dict[str, float] = {
        "det_precision": float(det_precision),
        "det_recall": float(det_recall),
        "det_f1": float(det_f1),
        "det_mean_iou_tp": float(det_mean_iou_tp),
        "det_tp": float(total_tp),
        "det_fp": float(total_fp),
        "det_fn": float(total_fn),
        "seg_precision": float(seg_precision),
        "seg_recall": float(seg_recall),
        "seg_dice": float(seg_dice),
        "seg_iou": float(seg_iou),
        "sanity_pred_box_mean_coverage_pct": float(np.mean(pred_covs)) if pred_covs else 100.0,
        "sanity_pred_box_leak_samples": float(sum(r["pred_box_outside_pixels"] > 0 for r in sanity_records)),
        "sanity_gt_box_mean_coverage_pct": float(np.mean(gt_covs)) if gt_covs else 100.0,
        "sanity_gt_box_leak_samples": float(sum(r["gt_box_outside_pixels"] > 0 for r in sanity_records)),
    }

    if bool(end2end_cfg.get("save_sanity_report", True)):
        default_report = pathlib.Path("outputs") / f"end2end_sanity_{split}.json"
        report_path = pathlib.Path(end2end_cfg.get("sanity_report_path", default_report))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "split": split,
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "samples": sanity_records,
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        metrics["sanity_report_path"] = str(report_path)

    print(
        "End-to-end eval "
        f"(split={split}) -> det_f1={metrics['det_f1']:.4f}, "
        f"seg_dice={metrics['seg_dice']:.4f}, "
        f"pred_box_mean_coverage={metrics['sanity_pred_box_mean_coverage_pct']:.2f}%"
    )
    return metrics


def train_from_config(config: Dict[str, Any]) -> None:
    data_cfg = config.get("data", {})
    sources = data_cfg.get("sources", [])
    preprocessed_path = str(data_cfg.get("preprocessed_path", ROOT / "data" / "preprocessed"))

    train_pairs = load_split_pairs(preprocessed_path, "train", sources)
    val_pairs = load_split_pairs(preprocessed_path, "val", sources)

    train_end2end(config, train_pairs=train_pairs, val_pairs=val_pairs)


def evaluate_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    data_cfg = config.get("data", {})
    detection_cfg = config.get("tooth_detection", {})

    sources = data_cfg.get("sources", [])
    preprocessed_path = str(data_cfg.get("preprocessed_path", ROOT / "data" / "preprocessed"))
    split = str(detection_cfg.get("eval_split", "test"))

    data_pairs = load_split_pairs(preprocessed_path, split, sources)
    return evaluate_end2end(config, data_pairs=data_pairs, split=split)
