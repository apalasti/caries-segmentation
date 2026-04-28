import os
import csv
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .cropping import (
    apply_patch_crop,
    center_patch_top_left,
    eval_patch_top_left_for_bbox,
    focused_patch_top_left,
    iter_tile_top_lefts,
    pad_to_tile_grid,
    random_patch_top_left,
    yolo_norm_box_to_pixels,
)


def load_image_mask_arrays(image_path, mask_path, size=(256, 256), transform=None):
    """Image and mask after resize and optional Albumentations; float32, mask binarized."""
    img = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    img = img.resize(size, resample=Image.BILINEAR)
    mask = mask.resize(size, resample=Image.NEAREST)

    img = np.array(img).astype(np.float32)
    img /= 255
    mask = np.array(mask)
    mask = (mask > 0).astype(np.float32)

    if transform is not None:
        augmented = transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

    return img, mask


def image_array_to_tensor(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if isinstance(img, np.ndarray):
        return torch.from_numpy(img).float().unsqueeze(0)
    return img


def mask_array_to_tensor(mask):
    if isinstance(mask, np.ndarray):
        return torch.from_numpy(mask.copy()).float().unsqueeze(0)
    return mask


class BaseKariesDataset(Dataset):
    def __init__(
        self,
        data_pairs,
        size=(256, 256),
        transform=None,
        patch_size=None,
        focused_crop_prob=0.5,
        center_crop=False,
    ):
        self.data_pairs = data_pairs
        self.size = size
        self.transform = transform
        self.patch_size = patch_size
        self.focused_crop_prob = focused_crop_prob
        self.center_crop = center_crop
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.data_pairs)

    def load_augmented_arrays(self, i):
        img_path, mask_path = self.data_pairs[i]
        return load_image_mask_arrays(img_path, mask_path, self.size, self.transform)

    def __getitem__(self, i):
        img, mask = self.load_augmented_arrays(i)

        if self.patch_size is not None:
            ph = pw = int(self.patch_size)
            h, w = img.shape[:2]
            if self.center_crop:
                y0, x0 = center_patch_top_left(h, w, ph, pw)
            elif self._rng.random() < self.focused_crop_prob:
                y0, x0 = focused_patch_top_left(mask, ph, pw, self._rng)
            else:
                y0, x0 = random_patch_top_left(h, w, ph, pw, self._rng)
            img, mask = apply_patch_crop(img, mask, y0, x0, ph, pw)

        return image_array_to_tensor(img), mask_array_to_tensor(mask)


def load_split_pairs(preprocessed_path, split, sources=None):
    csv_path = os.path.join(preprocessed_path, "data.csv")

    pairs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue

            if sources:
                if row["source"] not in sources:
                    continue

            img_id = row["id"]
            img_path = os.path.join(preprocessed_path, split, "images", f"{img_id}.png")
            mask_path = os.path.join(preprocessed_path, split, "masks", f"{img_id}.png")
            pairs.append((img_path, mask_path))

    return pairs


@dataclass(frozen=True)
class BboxCsvRow:
    image_rel: str
    mask_rel: str
    xc: float
    yc: float
    w: float
    h: float
    sample_id: str
    box_index: int
    split: str
    score: float


def read_bbox_csv_rows(csv_path: str) -> list[BboxCsvRow]:
    rows: list[BboxCsvRow] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                BboxCsvRow(
                    image_rel=row["image_rel"],
                    mask_rel=row["mask_rel"],
                    xc=float(row["xc"]),
                    yc=float(row["yc"]),
                    w=float(row["w"]),
                    h=float(row["h"]),
                    sample_id=row.get("sample_id", ""),
                    box_index=int(row.get("box_index", 0)),
                    split=row.get("split", ""),
                    score=float(row.get("score", 0.0)),
                )
            )
    return rows


def crop_patch_from_bbox_row(
    img: np.ndarray,
    mask: np.ndarray,
    row: BboxCsvRow,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    h, w = img.shape[:2]
    min_r, min_c, max_r, max_c = yolo_norm_box_to_pixels(row.xc, row.yc, row.w, row.h, h, w)
    y0, x0 = eval_patch_top_left_for_bbox(h, w, patch_size, patch_size, min_r, min_c, max_r, max_c)
    img_crop, mask_crop = apply_patch_crop(img, mask, y0, x0, patch_size, patch_size)
    return img_crop, mask_crop, y0, x0


class BboxCsvCropEvalDataset(Dataset):
    """Per-row bbox crops for evaluation/debugging workflows."""

    def __init__(
        self,
        rows: list[BboxCsvRow],
        preprocessed_path: str,
        size=(256, 256),
        transform=None,
        patch_size: int = 128,
    ):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.rows = rows
        self.preprocessed_path = preprocessed_path
        self.size = size
        self.transform = transform
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image_path = os.path.join(self.preprocessed_path, row.image_rel)
        mask_path = os.path.join(self.preprocessed_path, row.mask_rel)
        img, mask = load_image_mask_arrays(image_path, mask_path, self.size, self.transform)
        img_crop, mask_crop, y0, x0 = crop_patch_from_bbox_row(
            img, mask, row, self.patch_size
        )
        return {
            "image": image_array_to_tensor(img_crop),
            "mask": mask_array_to_tensor(mask_crop),
            "y0": y0,
            "x0": x0,
            "image_rel": row.image_rel,
            "meta": cast_dict(row),
        }


def cast_dict(row: BboxCsvRow) -> dict[str, Any]:
    return {
        "image_rel": row.image_rel,
        "mask_rel": row.mask_rel,
        "xc": row.xc,
        "yc": row.yc,
        "w": row.w,
        "h": row.h,
        "sample_id": row.sample_id,
        "box_index": row.box_index,
        "split": row.split,
        "score": row.score,
    }


class TiledEvalKariesDataset(BaseKariesDataset):
    """Val/test: non-overlapping tile grid on the resized canvas; metrics stitch full logits."""

    def __init__(self, data_pairs, size=(256, 256), transform=None, tile_size=128):
        super().__init__(
            data_pairs,
            size,
            transform,
            patch_size=None,
            focused_crop_prob=0.0,
            center_crop=False,
        )
        self.tile_size = int(tile_size)

    def __getitem__(self, i):
        img, mask = self.load_augmented_arrays(i)
        ph = pw = self.tile_size
        img_p, mask_p, _ = pad_to_tile_grid(img, mask, ph, pw)
        H, W = img_p.shape

        tile_imgs = []
        tile_masks = []
        for y0, x0 in iter_tile_top_lefts(H, W, ph, pw):
            ti, tm = apply_patch_crop(img_p, mask_p, y0, x0, ph, pw)
            ti = (ti - ti.min()) / (ti.max() - ti.min() + 1e-8)
            tile_imgs.append(torch.from_numpy(ti).float().unsqueeze(0))
            tile_masks.append(torch.from_numpy(tm.copy()).float().unsqueeze(0))

        return torch.stack(tile_imgs), torch.stack(tile_masks)
