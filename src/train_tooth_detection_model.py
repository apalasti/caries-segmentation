import argparse
import pathlib
from typing import Any, Dict, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .data.dataset import load_split_pairs
from .models.lightning_model import SegmentationLightningModule
from .models.bbox.yolo import YOLOv5
from .models.bbox.yolo_unet_conjunction import YOLOUNetConjunction


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "preprocessed_detection"
DEFAULT_OUT_DIR = ROOT / "checkpoints" / "detection"


class ToothYoloDataset(Dataset):
    def __init__(self, split_dir: pathlib.Path, image_size: int = 640) -> None:
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.image_size = image_size

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f"Missing images/labels directory in {split_dir}")

        self.image_paths = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.image_size > 0:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            width, height = img.size
            image_tensor = F.to_tensor(img)

        boxes = []
        labels = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }
        return image_tensor, target


class YoloGuidedSegmentationDataset(Dataset):
    """Segmentation dataset cropped using YOLO predicted boxes.

    Each sample is resized to detector_size, cropped around the highest-confidence
    predicted box (with optional padding), and then resized to unet_size.
    """

    def __init__(
        self,
        data_pairs: list[tuple[str, str]],
        detector: YOLOv5,
        detector_size: tuple[int, int],
        unet_size: tuple[int, int],
        score_threshold: float = 0.25,
        crop_padding_ratio: float = 0.05,
        device: torch.device | None = None,
    ) -> None:
        self.data_pairs = data_pairs
        self.detector = detector
        self.detector_size = detector_size
        self.unet_size = unet_size
        self.score_threshold = score_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self.device = device or torch.device("cpu")
        self.crop_boxes = self._predict_crop_boxes()

    def __len__(self) -> int:
        return len(self.data_pairs)

    def _predict_crop_boxes(self) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        det_h, det_w = self.detector_size

        self.detector.eval()
        with torch.no_grad():
            for image_path, _ in self.data_pairs:
                with Image.open(image_path) as img:
                    gray = img.convert("L")
                    resized_gray = gray.resize((det_w, det_h), Image.BILINEAR)

                rgb = resized_gray.convert("RGB")
                image_tensor = F.to_tensor(rgb).unsqueeze(0).to(self.device)
                output = self.detector(image_tensor)[0]

                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                keep = pred_scores >= self.score_threshold

                if keep.any():
                    kept_scores = pred_scores[keep]
                    kept_boxes = pred_boxes[keep]
                    best_idx = int(torch.argmax(kept_scores).item())
                    x1, y1, x2, y2 = [float(v.item()) for v in kept_boxes[best_idx]]
                else:
                    x1, y1, x2, y2 = 0.0, 0.0, float(det_w), float(det_h)

                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                px = bw * self.crop_padding_ratio
                py = bh * self.crop_padding_ratio

                xi1 = max(0, int(round(x1 - px)))
                yi1 = max(0, int(round(y1 - py)))
                xi2 = min(det_w, int(round(x2 + px)))
                yi2 = min(det_h, int(round(y2 + py)))

                if xi2 <= xi1 or yi2 <= yi1:
                    xi1, yi1, xi2, yi2 = 0, 0, det_w, det_h

                boxes.append((xi1, yi1, xi2, yi2))

        return boxes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.data_pairs[index]
        xi1, yi1, xi2, yi2 = self.crop_boxes[index]
        det_h, det_w = self.detector_size
        unet_h, unet_w = self.unet_size

        with Image.open(image_path) as img:
            gray = img.convert("L")
            gray = gray.resize((det_w, det_h), Image.BILINEAR)
            gray = gray.crop((xi1, yi1, xi2, yi2))
            gray = gray.resize((unet_w, unet_h), Image.BILINEAR)

        with Image.open(mask_path) as mask:
            m = mask.convert("L")
            m = m.resize((det_w, det_h), Image.NEAREST)
            m = m.crop((xi1, yi1, xi2, yi2))
            m = m.resize((unet_w, unet_h), Image.NEAREST)

        image_tensor = F.to_tensor(gray)
        mask_tensor = F.to_tensor(m)
        mask_tensor = (mask_tensor > 0.5).float()
        return image_tensor, mask_tensor


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tooth detection model on preprocessed YOLO labels"
    )
    parser.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def build_detector_model(
    device: torch.device,
    anchors: Optional[Sequence[Tuple[float, float]]] = None,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> torch.nn.Module:
    model = YOLOv5(
        num_classes=1,
        anchors=anchors,
        conf_threshold=conf_threshold,
        iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )
    model.to(device)
    return model


def evaluate_val_loss(model, dataloader, device: torch.device) -> float:
    model.train()
    losses = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack([img.to(device) for img in images], dim=0)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses.append(loss_dict["loss"].item())
    return sum(losses) / max(1, len(losses))


def train_tooth_detection(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    image_size: int,
    device_arg: str,
    anchors: Optional[Sequence[Tuple[float, float]]] = None,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> None:
    device = resolve_device(device_arg)

    train_ds = ToothYoloDataset(data_dir / "train", image_size=image_size)
    val_ds = ToothYoloDataset(data_dir / "val", image_size=image_size)

    if len(train_ds) == 0:
        raise ValueError(
            f"No training samples found in {data_dir / 'train'}. "
            "Run scripts/preprocess_tooth_detection.py and verify labels are generated."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"No validation samples found in {data_dir / 'val'}. "
            "Run scripts/preprocess_tooth_detection.py and verify labels are generated."
        )

    print(f"Loaded samples: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_detector_model(
        device,
        anchors=anchors,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for images, targets in train_loader:
            images = torch.stack([img.to(device) for img in images], dim=0)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = evaluate_val_loss(model, val_loader, device)

        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        last_ckpt = out_dir / "detector_last.pt"
        torch.save(model.state_dict(), last_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = out_dir / "detector_best.pt"
            torch.save(model.state_dict(), best_ckpt)
            print(f"Saved best checkpoint: {best_ckpt}")

    print(f"Training done. Best val_loss={best_val:.4f}")


def train_unet_with_yolo_boxes(
    config: Dict[str, Any],
    *,
    detector_checkpoint: pathlib.Path,
) -> None:
    training_cfg = config.get("training", {})
    detection_cfg = config.get("tooth_detection", {})

    device = resolve_device(str(detection_cfg.get("device", training_cfg.get("device", "auto"))))
    detector_size = int(detection_cfg.get("image_size", 640))
    detector_hw = (detector_size, detector_size)

    unet_size_raw = config.get("data", {}).get("images_size", [256, 256])
    unet_size = (int(unet_size_raw[0]), int(unet_size_raw[1]))

    anchors = _parse_anchors(detection_cfg.get("anchors"))
    score_threshold = float(detection_cfg.get("score_threshold", 0.25))
    nms_iou_threshold = float(detection_cfg.get("nms_iou_threshold", 0.45))
    max_detections = int(detection_cfg.get("max_detections", 300))
    crop_padding = float(detection_cfg.get("unet_crop_padding", 0.05))

    detector = build_detector_model(
        device=device,
        anchors=anchors,
        conf_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )

    if not detector_checkpoint.exists():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {detector_checkpoint}. "
            "Train YOLO first with training.task='tooth_detection'."
        )

    detector_state = torch.load(detector_checkpoint, map_location=device)
    detector.load_state_dict(detector_state)
    detector.eval()

    preprocessed_path = str(config["data"]["preprocessed_path"])
    sources = config.get("data", {}).get("sources", [])

    train_pairs = load_split_pairs(preprocessed_path, "train", sources)
    val_pairs = load_split_pairs(preprocessed_path, "val", sources)

    if len(train_pairs) == 0:
        raise ValueError("No train samples found in segmentation preprocessed dataset.")
    if len(val_pairs) == 0:
        raise ValueError("No val samples found in segmentation preprocessed dataset.")

    print("Preparing YOLO-guided crop boxes for U-Net training...")
    train_ds = YoloGuidedSegmentationDataset(
        train_pairs,
        detector=detector,
        detector_size=detector_hw,
        unet_size=unet_size,
        score_threshold=score_threshold,
        crop_padding_ratio=crop_padding,
        device=device,
    )
    val_ds = YoloGuidedSegmentationDataset(
        val_pairs,
        detector=detector,
        detector_size=detector_hw,
        unet_size=unet_size,
        score_threshold=score_threshold,
        crop_padding_ratio=crop_padding,
        device=device,
    )

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    shuffle = bool(training_cfg.get("shuffle_train", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = SegmentationLightningModule(config)

    output_dir = pathlib.Path(training_cfg.get("output_dir", ROOT / "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if training_cfg.get("checkpointing", True):
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="best_model_yolo_guided",
                monitor="val/dice_loss",
                mode="min",
                save_top_k=1,
                verbose=True,
            )
        )
    if training_cfg.get("early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor="val/dice_loss",
                patience=int(training_cfg.get("early_stopping_patience", 10)),
                mode="min",
                verbose=True,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    logger = WandbLogger(
        project=config["wandb"]["project"],
        config=config,
    )

    trainer = pl.Trainer(
        max_epochs=int(training_cfg.get("epochs", 20)),
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.experiment.finish()


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


def evaluate_tooth_detection(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    device_arg: str,
    split: str,
    score_threshold: float,
    iou_threshold: float,
    anchors: Optional[Sequence[Tuple[float, float]]] = None,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 300,
    conjunction_model: Optional[YOLOUNetConjunction] = None,
) -> Dict[str, float]:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split '{split}' not found under {data_dir}")

    dataset = ToothYoloDataset(split_dir, image_size=image_size)
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {split_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    device = resolve_device(device_arg)
    model = build_detector_model(
        device,
        anchors=anchors,
        conf_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )

    checkpoint_path = out_dir / "detector_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train detection first."
        )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if conjunction_model is not None:
        conjunction_model.detector.load_state_dict(state)
        conjunction_model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_refined_area = 0.0
    total_pixels = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack([img.to(device) for img in images], dim=0)
            if conjunction_model is None:
                outputs = model(images)
                refined_masks = None
            else:
                conj_out = conjunction_model(images)
                outputs = conj_out["detections"]
                refined_masks = conj_out["masks"]

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
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

            if refined_masks is not None:
                total_refined_area += float(refined_masks.sum().item())
                total_pixels += float(refined_masks.numel())

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    mean_iou_tp = total_iou_sum / max(1, total_tp)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou_tp": mean_iou_tp,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }

    if total_pixels > 0:
        metrics["refined_area_ratio"] = total_refined_area / total_pixels

    print(
        "Detection eval "
        f"(split={split}, score_thr={score_threshold}, iou_thr={iou_threshold}) -> "
        f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, mean_iou_tp={mean_iou_tp:.4f}, "
        f"tp={total_tp}, fp={total_fp}, fn={total_fn}"
    )
    return metrics


def train_from_config(config: Dict[str, Any]) -> None:
    training_cfg = config.get("training", {})
    detection_cfg = config.get("tooth_detection", {})

    data_dir = pathlib.Path(detection_cfg.get("data_dir", DEFAULT_DATA_DIR))
    out_dir = pathlib.Path(
        detection_cfg.get("output_dir", detection_cfg.get("out_dir", DEFAULT_OUT_DIR))
    )

    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 4))
    learning_rate = float(training_cfg.get("learning_rate", training_cfg.get("lr", 1e-4)))
    num_workers = int(training_cfg.get("num_workers", 2))
    image_size = int(detection_cfg.get("image_size", 640))
    device_arg = str(detection_cfg.get("device", training_cfg.get("device", "auto")))
    anchors = _parse_anchors(detection_cfg.get("anchors"))
    conf_threshold = float(detection_cfg.get("score_threshold", 0.25))
    nms_iou_threshold = float(detection_cfg.get("nms_iou_threshold", 0.45))
    max_detections = int(detection_cfg.get("max_detections", 300))

    train_tooth_detection(
        data_dir=data_dir,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        image_size=image_size,
        device_arg=device_arg,
        anchors=anchors,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )


def train_unet_with_yolo_boxes_from_config(config: Dict[str, Any]) -> None:
    detection_cfg = config.get("tooth_detection", {})
    out_dir = pathlib.Path(
        detection_cfg.get("output_dir", detection_cfg.get("out_dir", DEFAULT_OUT_DIR))
    )
    detector_checkpoint = pathlib.Path(detection_cfg.get("detector_checkpoint", out_dir / "detector_best.pt"))
    train_unet_with_yolo_boxes(config, detector_checkpoint=detector_checkpoint)


def evaluate_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    training_cfg = config.get("training", {})
    detection_cfg = config.get("tooth_detection", {})

    data_dir = pathlib.Path(detection_cfg.get("data_dir", DEFAULT_DATA_DIR))
    out_dir = pathlib.Path(
        detection_cfg.get("output_dir", detection_cfg.get("out_dir", DEFAULT_OUT_DIR))
    )

    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 2))
    image_size = int(detection_cfg.get("image_size", 640))
    device_arg = str(detection_cfg.get("device", training_cfg.get("device", "auto")))
    split = str(detection_cfg.get("eval_split", "test"))
    score_threshold = float(detection_cfg.get("score_threshold", 0.25))
    iou_threshold = float(detection_cfg.get("eval_iou_threshold", 0.5))
    nms_iou_threshold = float(detection_cfg.get("nms_iou_threshold", 0.45))
    max_detections = int(detection_cfg.get("max_detections", 300))
    anchors = _parse_anchors(detection_cfg.get("anchors"))

    conjunction_model = None
    if bool(detection_cfg.get("use_unet_conjunction", False)):
        checkpoint_path = detection_cfg.get("unet_checkpoint", "")
        if not checkpoint_path:
            raise ValueError(
                "tooth_detection.use_unet_conjunction=true requires tooth_detection.unet_checkpoint"
            )
        checkpoint = pathlib.Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"UNet checkpoint not found: {checkpoint}")

        unet_module = SegmentationLightningModule.load_from_checkpoint(
            str(checkpoint),
            config=config,
        )
        unet_module.eval()

        unet_input_size = config.get("data", {}).get("images_size", [256, 256])
        conjunction_model = YOLOUNetConjunction(
            detector=build_detector_model(
                resolve_device(device_arg),
                anchors=anchors,
                conf_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            ),
            segmenter=unet_module.model_instance,
            unet_input_size=(int(unet_input_size[0]), int(unet_input_size[1])),
            mask_threshold=float(detection_cfg.get("unet_mask_threshold", 0.5)),
            crop_padding_ratio=float(detection_cfg.get("unet_crop_padding", 0.05)),
        ).to(resolve_device(device_arg))

    return evaluate_tooth_detection(
        data_dir=data_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        device_arg=device_arg,
        split=split,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        anchors=anchors,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        conjunction_model=conjunction_model,
    )


def main() -> None:
    args = parse_args()
    train_tooth_detection(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()