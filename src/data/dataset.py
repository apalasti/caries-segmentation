import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .cropping import (
    apply_patch_crop,
    center_patch_top_left,
    focused_patch_top_left,
    iter_tile_top_lefts,
    pad_to_tile_grid,
    random_patch_top_left,
)


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
        """Image and mask after resize and optional Albumentations; float32, mask binarized."""
        img_path, mask_path = self.data_pairs[i]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        img = np.array(img).astype(np.float32)
        img /= 255
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        return img, mask

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

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).unsqueeze(0)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


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
