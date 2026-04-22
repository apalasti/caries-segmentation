import os
import csv
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BaseKariesDataset(Dataset):
    def __init__(
        self,
        data_pairs,
        size=(256, 256),
        transform=None,
        bbox_padding=10.0,
        return_targets=False,
    ):
        self.data_pairs = data_pairs
        self.size = size
        self.transform = transform
        self.bbox_padding = bbox_padding
        self.return_targets = return_targets

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.data_pairs[i]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        # Normalize by intensity
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.float32)

        if self.transform:
            # Assumes albumentations format
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np = augmented['image']
            mask_np = augmented['mask']

        # Build detection-style targets from connected mask components.
        boxes = []
        labels = []
        
        mask_u8 = (mask_np * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        padding = int(round(self.bbox_padding))
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 2 or h < 2:
                continue
            
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.size[0], x + w + padding)
            y2 = min(self.size[1], y + h + padding)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(1)

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        if self.return_targets:
            # End-to-end detector path expects RGB-like 3-channel tensors.
            img_tensor = img_tensor.repeat(3, 1, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "masks": mask_tensor,
        }

        if self.return_targets:
            return img_tensor, target
        return img_tensor, mask_tensor


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
