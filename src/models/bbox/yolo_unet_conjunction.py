from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .yolo import YOLOv5


class YOLOUNetConjunction(nn.Module):
    def __init__(
        self,
        detector: YOLOv5,
        segmenter: nn.Module,
        unet_input_size: Tuple[int, int] = (256, 256),
        mask_threshold: float = 0.5,
        crop_padding_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.segmenter = segmenter
        self.unet_input_size = unet_input_size
        self.mask_threshold = mask_threshold
        self.crop_padding_ratio = crop_padding_ratio

    def forward(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]] | None = None,
    ) -> Dict[str, List[Dict[str, Tensor]] | Tensor] | Dict[str, Tensor]:
        if self.training and targets is not None:
            return self.detector(images, targets)

        detections = self.detector(images)
        masks = self._refine_with_unet(images, detections)
        return {
            "detections": detections,
            "masks": masks,
        }

    def _refine_with_unet(self, images: Tensor, detections: List[Dict[str, Tensor]]) -> Tensor:
        bsz, _, height, width = images.shape
        full_masks = torch.zeros((bsz, 1, height, width), device=images.device)

        self.segmenter.eval()
        with torch.no_grad():
            for b in range(bsz):
                image = images[b : b + 1]
                for box in detections[b]["boxes"]:
                    x1, y1, x2, y2 = [float(v.item()) for v in box]

                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    px = bw * self.crop_padding_ratio
                    py = bh * self.crop_padding_ratio

                    xi1 = max(0, int(round(x1 - px)))
                    yi1 = max(0, int(round(y1 - py)))
                    xi2 = min(width, int(round(x2 + px)))
                    yi2 = min(height, int(round(y2 + py)))

                    if xi2 <= xi1 or yi2 <= yi1:
                        continue

                    crop = image[:, :, yi1:yi2, xi1:xi2]
                    if crop.shape[-1] < 2 or crop.shape[-2] < 2:
                        continue

                    if crop.shape[1] != 1:
                        crop = crop.mean(dim=1, keepdim=True)

                    resized = F.interpolate(
                        crop,
                        size=self.unet_input_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    seg_logits = self.segmenter(resized)
                    seg_probs = torch.sigmoid(seg_logits)
                    seg_bin = (seg_probs >= self.mask_threshold).float()

                    seg_crop = F.interpolate(
                        seg_bin,
                        size=(yi2 - yi1, xi2 - xi1),
                        mode="nearest",
                    )

                    full_masks[b : b + 1, :, yi1:yi2, xi1:xi2] = torch.maximum(
                        full_masks[b : b + 1, :, yi1:yi2, xi1:xi2],
                        seg_crop,
                    )

        return full_masks