import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class BaseKariesDataset(Dataset):
    def __init__(self, data_pairs, size=(256, 256)):
        self.data_pairs = data_pairs
        self.size = size

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.data_pairs[i]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        img = np.array(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
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
