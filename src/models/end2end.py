from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import roi_align

class EndToEndCariesModel(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        segmenter: nn.Module,
        unet_input_size: Tuple[int, int] = (64, 64),
        train_with_gt_boxes_prob: float = 0.5,
        mask_threshold: float = 0.5,
        crop_padding_ratio: float = 0.05,
        min_crop_size: int = 2,
        segmenter_in_channels: int | None = None,
        max_rois_per_image: int | None = 32,
    ):
        super().__init__()
        self.detector = detector
        self.segmenter = segmenter
        self.unet_input_size = unet_input_size
        self.train_with_gt_boxes_prob = train_with_gt_boxes_prob
        self.mask_threshold = mask_threshold
        self.crop_padding_ratio = crop_padding_ratio
        self.min_crop_size = max(1, int(min_crop_size))
        self.max_rois_per_image = None if max_rois_per_image is None else max(1, int(max_rois_per_image))
        if segmenter_in_channels is None:
            segmenter_in_channels = int(getattr(segmenter, "n_channels", 1))
        self.segmenter_in_channels = max(1, int(segmenter_in_channels))

    def _decode_detector_outputs(self, images: Tensor) -> tuple[Tensor, List[Dict[str, Tensor]]]:
        raw_pred = self.detector._forward_raw(images)
        detections = self.detector.decode_predictions(raw_pred, images.shape[-2:])
        return raw_pred, detections

    def _pad_and_clip_boxes(self, boxes: Tensor, height: int, width: int) -> Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))

        b = boxes.clone()
        widths = (b[:, 2] - b[:, 0]).clamp(min=1.0)
        heights = (b[:, 3] - b[:, 1]).clamp(min=1.0)
        pad_x = widths * self.crop_padding_ratio
        pad_y = heights * self.crop_padding_ratio

        b[:, 0] = (b[:, 0] - pad_x).clamp(min=0, max=width)
        b[:, 1] = (b[:, 1] - pad_y).clamp(min=0, max=height)
        b[:, 2] = (b[:, 2] + pad_x).clamp(min=0, max=width)
        b[:, 3] = (b[:, 3] + pad_y).clamp(min=0, max=height)

        valid = (b[:, 2] - b[:, 0] >= self.min_crop_size) & (b[:, 3] - b[:, 1] >= self.min_crop_size)
        return b[valid]

    def _select_boxes(
        self,
        detections: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]] | None,
        height: int,
        width: int,
    ) -> tuple[List[Tensor], List[Tensor]]:
        boxes_to_use: List[Tensor] = []
        batch_inds: List[Tensor] = []

        use_gt_boxes = False
        if self.training and targets is not None:
            use_gt_boxes = torch.rand(1).item() < self.train_with_gt_boxes_prob

        for b in range(len(detections)):
            if use_gt_boxes and targets is not None:
                candidate_boxes = targets[b].get("boxes", detections[b]["boxes"])
            else:
                candidate_boxes = detections[b]["boxes"]

            clipped = self._pad_and_clip_boxes(candidate_boxes, height=height, width=width)
            if clipped.numel() == 0:
                continue

            if self.max_rois_per_image is not None and clipped.size(0) > self.max_rois_per_image:
                areas = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
                keep = torch.argsort(areas, descending=True)[: self.max_rois_per_image]
                clipped = clipped[keep]

            boxes_to_use.append(clipped)
            batch_inds.append(torch.full((clipped.size(0),), b, device=clipped.device, dtype=torch.long))

        return boxes_to_use, batch_inds

    def _adapt_crop_channels(self, crops: Tensor) -> Tensor:
        if crops.shape[1] == self.segmenter_in_channels:
            return crops
        if self.segmenter_in_channels == 1:
            return crops.mean(dim=1, keepdim=True)
        if crops.shape[1] == 1:
            return crops.repeat(1, self.segmenter_in_channels, 1, 1)
        return crops[:, : self.segmenter_in_channels]

    def _compute_unet_loss(
        self,
        seg_logits: Tensor,
        roi_boxes: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Tensor:
        losses: List[Tensor] = []

        for i, roi in enumerate(roi_boxes):
            b_idx = int(roi[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in roi[1:].tolist()]

            gt_mask = targets[b_idx]["masks"]
            if gt_mask.dim() == 3:
                gt_mask = gt_mask[0]

            h, w = gt_mask.shape[-2], gt_mask.shape[-1]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                continue

            gt_crop = gt_mask[y1:y2, x1:x2]
            if gt_crop.numel() == 0:
                continue

            gt_crop = gt_crop.unsqueeze(0).unsqueeze(0).float()
            gt_resized = F.interpolate(gt_crop, size=seg_logits.shape[-2:], mode="nearest")
            losses.append(F.binary_cross_entropy_with_logits(seg_logits[i : i + 1], gt_resized))

        if not losses:
            return seg_logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _paste_masks(self, full_masks: Tensor, seg_bin: Tensor, roi_boxes: Tensor) -> None:
        _, _, height, width = full_masks.shape
        for i, roi in enumerate(roi_boxes):
            b_idx = int(roi[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in roi[1:].tolist()]

            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                continue

            patch = F.interpolate(seg_bin[i : i + 1], size=(y2 - y1, x2 - x1), mode="nearest")
            full_masks[b_idx, 0, y1:y2, x1:x2] = torch.maximum(
                full_masks[b_idx, 0, y1:y2, x1:x2],
                patch[0, 0],
            )

    def forward(
        self, images: Tensor, targets: List[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        bsz, _, height, width = images.shape
        final_masks = torch.zeros((bsz, 1, height, width), device=images.device)

        raw_pred, detections = self._decode_detector_outputs(images)

        detector_losses: Dict[str, Tensor] = {}
        if self.training and targets is not None:
            detector_losses = self.detector.compute_loss(raw_pred, targets, images.shape[-2:])

        boxes_to_use, batch_inds = self._select_boxes(
            detections=detections,
            targets=targets,
            height=height,
            width=width,
        )

        if not boxes_to_use:
            if self.training and targets is not None:
                losses = {f"detector_{k}": v for k, v in detector_losses.items()}
                unet_loss = images.new_tensor(0.0)
                det_total = detector_losses.get("loss", images.new_tensor(0.0))
                losses["unet_loss"] = unet_loss
                losses["loss"] = det_total + unet_loss
                return losses
            return {"detections": detections, "masks": final_masks}

        flat_boxes = torch.cat(boxes_to_use, dim=0)
        flat_inds = torch.cat(batch_inds, dim=0).view(-1, 1).float()
        roi_boxes = torch.cat([flat_inds, flat_boxes], dim=1)

        crops = roi_align(
            images,
            roi_boxes,
            output_size=self.unet_input_size,
            spatial_scale=1.0,
            aligned=True,
        )
        crops = self._adapt_crop_channels(crops)
        seg_logits = self.segmenter(crops)

        if self.training and targets is not None:
            unet_loss = self._compute_unet_loss(seg_logits, roi_boxes, targets)
            det_total = detector_losses.get("loss", images.new_tensor(0.0))

            losses = {f"detector_{k}": v for k, v in detector_losses.items()}
            losses["unet_loss"] = unet_loss
            losses["loss"] = det_total + unet_loss
            return losses

        seg_probs = torch.sigmoid(seg_logits)
        seg_bin = (seg_probs >= self.mask_threshold).float()
        self._paste_masks(final_masks, seg_bin, roi_boxes)
        return {"detections": detections, "masks": final_masks}
