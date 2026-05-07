from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cropping import (
    apply_patch_crop,
    bbox_covering_patch_top_left,
    clamp_yolo_bbox,
    normed_box_to_pixels,
)


def _load_resized_arrays(
    image_path: str, mask_path: str, size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    if size is not None:
        img = img.resize(size, resample=Image.BILINEAR)
        mask = mask.resize(size, resample=Image.NEAREST)

    img_np = np.array(img).astype(np.float32) / 255.0
    mask_np = (np.array(mask) > 0).astype(np.float32)
    return img_np, mask_np


def _arrays_to_tensors(
    img_np: np.ndarray, mask_np: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    img = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_t = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)
    mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).float().unsqueeze(0)
    return img_t, mask_t


class FullImageDataset(Dataset):
    """Yields a full resized image and mask, optionally augmented.

    Item: ``{"id": str, "image": Tensor[1, H, W], "mask": Tensor[1, H, W]}``.
    """

    def __init__(
        self,
        images_df: pd.DataFrame,
        transform: Any = None,
    ):
        self.images_df = images_df.reset_index(drop=True).copy()
        self.images_df = self.images_df.set_index("id", drop=False)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.images_df.iloc[i]
        img_np, mask_np = _load_resized_arrays(row["image_path"], row["mask_path"])

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented["image"]
            mask_np = augmented["mask"]

        img_t, mask_t = _arrays_to_tensors(img_np, mask_np)
        return {"id": str(row["id"]), "image": img_t, "mask": mask_t}


class BboxPatchDataset(Dataset):
    """Per-bbox patch dataset for training.

    One dataset item corresponds to a single bbox: the source image is loaded,
    optionally augmented (with the target bbox flowing through albumentations),
    then a ``patch_size`` window covering the augmented bbox is cut out.

    Item: ``{"id": str, "image": Tensor[1, ph, pw], "mask": Tensor[1, ph, pw]}``.
    """

    def __init__(
        self,
        images_df: pd.DataFrame,
        bboxes_df: pd.DataFrame,
        transform: Any = None,
        patch_size: int = 128,
    ):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.images_df = images_df.reset_index(drop=True).copy()
        self.images_df = self.images_df.set_index("id", drop=False)

        self.transform = transform
        self.patch_size = int(patch_size)

        bbox_df = bboxes_df[bboxes_df["id"].isin(self.images_df.index)].copy()
        self.bbox_df = bbox_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.bbox_df)

    def _augment(
        self,
        img_np: np.ndarray,
        mask_np: np.ndarray,
        bbox_yolo: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        if self.transform is None:
            return img_np, mask_np, bbox_yolo

        try:
            augmented = self.transform(
                image=img_np, mask=mask_np, bboxes=[bbox_yolo]
            )
        except TypeError:
            augmented = self.transform(image=img_np, mask=mask_np)
            return augmented["image"], augmented["mask"], bbox_yolo

        bboxes_aug = augmented.get("bboxes", [])
        if len(bboxes_aug) == 0:
            return augmented["image"], augmented["mask"], bbox_yolo

        raw = bboxes_aug[0]
        b = clamp_yolo_bbox(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return augmented["image"], augmented["mask"], b

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.bbox_df.iloc[i]
        image_id = str(row["id"])
        img_np, mask_np = _load_resized_arrays(
            self.images_df.loc[image_id, "image_path"],
            self.images_df.loc[image_id, "mask_path"],
            None,
        )

        bbox_yolo = clamp_yolo_bbox(
            float(row["xc"]),
            float(row["yc"]),
            float(row["w"]),
            float(row["h"]),
        )
        img_np, mask_np, bbox_yolo = self._augment(img_np, mask_np, bbox_yolo)

        h, w = int(img_np.shape[0]), int(img_np.shape[1])
        ph = pw = self.patch_size
        if h < ph or w < pw:
            raise ValueError(
                f"image {image_id} has shape ({h}, {w}) smaller than patch_size {ph}"
            )
        min_r, min_c, max_r, max_c = normed_box_to_pixels(
            bbox_yolo[0], bbox_yolo[1], bbox_yolo[2], bbox_yolo[3], h, w
        )
        y0, x0 = bbox_covering_patch_top_left(h, w, ph, pw, min_r, min_c, max_r, max_c)
        patch_img_np, patch_mask_np = apply_patch_crop(img_np, mask_np, y0, x0, ph, pw)

        img_t, mask_t = _arrays_to_tensors(patch_img_np, patch_mask_np)
        return {"id": image_id, "image": img_t, "mask": mask_t}


class BboxEvalDataset(FullImageDataset):
    """Full-image eval dataset that exposes patch metadata for stitched inference.

    Yields the full resized image and mask (no augmentation), plus the patch
    origins derived from the image's bboxes. Variable ``N`` per item, so the
    DataLoader must use ``batch_size=1``.

    Item: ``{
        "id": str,
        "image": Tensor[1, H, W],
        "mask":  Tensor[1, H, W],
        "origins_yx": Tensor[N, 2],
        "patch_hw":   Tensor[2],
    }``.
    """

    def __init__(
        self,
        images_df: pd.DataFrame,
        bboxes_df: pd.DataFrame,
        patch_size: int = 128,
    ):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        super().__init__(images_df=images_df, transform=None)
        self.patch_size = int(patch_size)

        bbox_df = bboxes_df[bboxes_df["id"].isin(self.images_df.index)].copy()
        self.bboxes_grouped = bbox_df.groupby("id", sort=False)[
            ["xc", "yc", "w", "h"]
        ]

    def _origins_for_id(self, image_id: str, h: int, w: int) -> torch.Tensor:
        gb = self.bboxes_grouped
        if image_id not in gb.groups:
            return torch.zeros((0, 2), dtype=torch.long)
        sub = gb.get_group(image_id)

        ph = pw = self.patch_size
        if h < ph or w < pw:
            raise ValueError(
                f"image {image_id} has shape ({h}, {w}) smaller than patch_size {ph}"
            )
        origins: list[tuple[int, int]] = []
        for xc, yc, bw, bh in zip(sub["xc"], sub["yc"], sub["w"], sub["h"]):
            xc, yc, bw, bh = clamp_yolo_bbox(float(xc), float(yc), float(bw), float(bh))
            min_r, min_c, max_r, max_c = normed_box_to_pixels(xc, yc, bw, bh, h, w)
            y0, x0 = bbox_covering_patch_top_left(
                h, w, ph, pw, min_r, min_c, max_r, max_c
            )
            origins.append((y0, x0))
        return torch.tensor(origins, dtype=torch.long)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = super().__getitem__(i)
        _, h, w = item["image"].shape
        item["origins_yx"] = self._origins_for_id(item["id"], int(h), int(w))
        item["patch_hw"] = torch.tensor(
            [self.patch_size, self.patch_size], dtype=torch.long
        )
        return item
