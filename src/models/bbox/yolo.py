from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import box_iou, nms


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.SiLU(inplace=True),
	)


class YOLOv5(nn.Module):
	"""A compact YOLO-style detector with decoding and optional training loss."""

	def __init__(
		self,
		num_classes: int,
		anchors: Optional[Sequence[Tuple[float, float]]] = None,
		conf_threshold: float = 0.25,
		iou_threshold: float = 0.45,
		max_detections: int = 300,
	) -> None:
		super().__init__()
		if num_classes < 1:
			raise ValueError("num_classes must be >= 1")

		self.num_classes = num_classes
		self.conf_threshold = conf_threshold
		self.iou_threshold = iou_threshold
		self.max_detections = max_detections

		anchor_values = anchors or ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0))
		self.num_anchors = len(anchor_values)
		self.register_buffer("anchors", torch.tensor(anchor_values, dtype=torch.float32))

		self.backbone = nn.Sequential(
			_conv_block(3, 32, stride=1),
			_conv_block(32, 64, stride=2),
			_conv_block(64, 64, stride=1),
			_conv_block(64, 128, stride=2),
			_conv_block(128, 128, stride=1),
			_conv_block(128, 256, stride=2),
			_conv_block(256, 256, stride=1),
			_conv_block(256, 512, stride=2),
			_conv_block(512, 512, stride=1),
			_conv_block(512, 1024, stride=2),
			_conv_block(1024, 512, stride=1),
		)

		out_channels = self.num_anchors * (5 + self.num_classes)
		self.head = nn.Conv2d(512, out_channels, kernel_size=1)

	def forward(
		self,
		images: Tensor,
		targets: Optional[List[Dict[str, Tensor]]] = None,
	) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
		raw = self._forward_raw(images)
		if self.training and targets is not None:
			return self.compute_loss(raw, targets, images.shape[-2:])
		return self.decode_predictions(raw, images.shape[-2:])

	def _forward_raw(self, images: Tensor) -> Tensor:
		feat = self.backbone(images)
		pred = self.head(feat)
		bsz, _, h, w = pred.shape
		pred = pred.view(bsz, self.num_anchors, 5 + self.num_classes, h, w)
		pred = pred.permute(0, 1, 3, 4, 2).contiguous()
		return pred

	def decode_predictions(
		self,
		raw_pred: Tensor,
		image_size: Tuple[int, int],
	) -> List[Dict[str, Tensor]]:
		bsz, _, h, w, _ = raw_pred.shape
		stride_y = image_size[0] / h
		stride_x = image_size[1] / w

		gy, gx = torch.meshgrid(
			torch.arange(h, device=raw_pred.device),
			torch.arange(w, device=raw_pred.device),
			indexing="ij",
		)
		grid = torch.stack((gx, gy), dim=-1).float().view(1, 1, h, w, 2)

		anchor_grid = self.anchors.view(1, self.num_anchors, 1, 1, 2)
		pred_xy = (torch.sigmoid(raw_pred[..., 0:2]) + grid) * torch.tensor(
			[stride_x, stride_y], device=raw_pred.device
		)
		pred_wh = torch.exp(raw_pred[..., 2:4]).clamp(max=1e4) * anchor_grid
		pred_obj = torch.sigmoid(raw_pred[..., 4:5])
		pred_cls = torch.sigmoid(raw_pred[..., 5:])

		x1y1 = pred_xy - pred_wh / 2.0
		x2y2 = pred_xy + pred_wh / 2.0
		boxes = torch.cat([x1y1, x2y2], dim=-1).view(bsz, -1, 4)

		scores_all = (pred_obj * pred_cls).view(bsz, -1, self.num_classes)
		outputs: List[Dict[str, Tensor]] = []

		for b in range(bsz):
			cls_scores, cls_labels = scores_all[b].max(dim=1)
			keep = cls_scores >= self.conf_threshold
			if keep.sum() == 0:
				outputs.append(
					{
						"boxes": boxes[b].new_zeros((0, 4)),
						"scores": cls_scores.new_zeros((0,)),
						"labels": cls_labels.new_zeros((0,), dtype=torch.long),
					}
				)
				continue

			kept_boxes = boxes[b][keep]
			kept_scores = cls_scores[keep]
			kept_labels = cls_labels[keep] + 1

			selected = nms(kept_boxes, kept_scores, self.iou_threshold)
			selected = selected[: self.max_detections]

			outputs.append(
				{
					"boxes": kept_boxes[selected],
					"scores": kept_scores[selected],
					"labels": kept_labels[selected],
				}
			)

		return outputs

	def compute_loss(
		self,
		raw_pred: Tensor,
		targets: List[Dict[str, Tensor]],
		image_size: Tuple[int, int],
	) -> Dict[str, Tensor]:
		bsz, _, h, w, _ = raw_pred.shape
		device = raw_pred.device
		dtype = raw_pred.dtype

		target_obj = torch.zeros((bsz, self.num_anchors, h, w), device=device, dtype=dtype)
		target_box = torch.zeros((bsz, self.num_anchors, h, w, 4), device=device, dtype=dtype)
		target_cls = torch.zeros(
			(bsz, self.num_anchors, h, w, self.num_classes),
			device=device,
			dtype=dtype,
		)

		stride_y = image_size[0] / h
		stride_x = image_size[1] / w

		for b, target in enumerate(targets):
			gt_boxes = target.get("boxes")
			gt_labels = target.get("labels")
			if gt_boxes is None or gt_labels is None or gt_boxes.numel() == 0:
				continue

			gt_boxes = gt_boxes.to(device=device, dtype=dtype)
			gt_labels = gt_labels.to(device=device, dtype=torch.long)

			widths = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
			heights = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
			centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
			centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5

			gi = (centers_x / stride_x).long().clamp(min=0, max=w - 1)
			gj = (centers_y / stride_y).long().clamp(min=0, max=h - 1)

			gt_wh = torch.stack([widths, heights], dim=1)
			anchor_boxes = torch.cat(
				[torch.zeros_like(self.anchors), self.anchors],
				dim=1,
			)
			gt_boxes_wh = torch.cat([torch.zeros_like(gt_wh), gt_wh], dim=1)
			ious = box_iou(gt_boxes_wh, anchor_boxes)
			best_anchors = ious.argmax(dim=1)

			for i in range(gt_boxes.shape[0]):
				a = best_anchors[i]
				xg = gi[i]
				yg = gj[i]

				target_obj[b, a, yg, xg] = 1.0

				tx = centers_x[i] / stride_x - xg.float()
				ty = centers_y[i] / stride_y - yg.float()
				tw = torch.log((widths[i] / self.anchors[a, 0]).clamp(min=1e-6))
				th = torch.log((heights[i] / self.anchors[a, 1]).clamp(min=1e-6))
				target_box[b, a, yg, xg] = torch.stack([tx, ty, tw, th])

				cls_idx = int(gt_labels[i].item()) - 1
				if 0 <= cls_idx < self.num_classes:
					target_cls[b, a, yg, xg, cls_idx] = 1.0

		obj_logits = raw_pred[..., 4]
		box_logits = raw_pred[..., 0:4]
		cls_logits = raw_pred[..., 5:]

		obj_loss = F.binary_cross_entropy_with_logits(obj_logits, target_obj)

		positive_mask = target_obj > 0
		if positive_mask.any():
			box_loss = F.smooth_l1_loss(box_logits[positive_mask], target_box[positive_mask])
			cls_loss = F.binary_cross_entropy_with_logits(
				cls_logits[positive_mask],
				target_cls[positive_mask],
			)
		else:
			zero = raw_pred.new_tensor(0.0)
			box_loss = zero
			cls_loss = zero

		total_loss = 5.0 * box_loss + obj_loss + cls_loss
		return {
			"loss": total_loss,
			"loss_box": box_loss,
			"loss_obj": obj_loss,
			"loss_cls": cls_loss,
		}



