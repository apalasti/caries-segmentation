import os
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .augmentations import get_train_transforms, get_val_transforms
from .dataset import (
    BboxEvalDataset,
    BboxPatchDataset,
    FullImageDataset,
    MixedCariesBboxPatchDataset,
)


def load_split_pairs(preprocessed_path, split: str, sources=None):
    csv_path = os.path.join(preprocessed_path, "dataset.csv")
    df = pd.read_csv(csv_path).astype(
        {
            "id": "string",
            "split": "string",
            "source": "string",
        }
    )
    df = df[df["split"] == split]
    if sources:
        df = df[df["source"].isin(sources)]

    df["image_path"] = preprocessed_path + "/" + split + "/images/" + df["id"] + ".png"
    df["mask_path"] = preprocessed_path + "/" + split + "/masks/" + df["id"] + ".png"
    return df


def load_bboxes_df(bboxes_csv_path: str) -> pd.DataFrame:
    return pd.read_csv(bboxes_csv_path).astype(
        {
            "id": "string",
            "w": "float64",
            "h": "float64",
            "xc": "float64",
            "yc": "float64",
            "box_index": "int64",
            "score": "float64",
            "has_caries": "bool"
        }
    )


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessed_path = config["data"]["preprocessed_path"]
        self.sources = config["data"].get("sources", [])
        self.batch_size = config["training"].get("batch_size", 32)
        self.num_workers = config["training"].get("num_workers", 4)
        self.size = tuple(config["data"].get("images_size", [256, 256]))

        raw_patch = config["data"].get("patch_size", None)
        self.patch_size = int(raw_patch) if raw_patch is not None else None
        self.bbox_mode = self.patch_size is not None
        self.bboxes_csv_path = config["data"].get(
            "bboxes_csv", os.path.join(self.preprocessed_path, "bboxes.csv")
        )

        if self.bbox_mode:
            if self.patch_size <= 0:
                raise ValueError("data.patch_size must be positive when set.")
            sh, sw = self.size
            if sh < self.patch_size or sw < self.patch_size:
                raise ValueError(
                    f"data.images_size {self.size} must be >= patch_size {self.patch_size} "
                    "on each axis."
                )

        aug_config = config.get("augmentation", {})
        self.augmentation_enabled = aug_config.get("enabled", True)
        self.train_transform = (
            get_train_transforms(aug_config, self.size, bbox_aware=self.bbox_mode)
            if self.augmentation_enabled
            else None
        )
        self.val_transform = get_val_transforms(self.size)

    def _dataloader_kwargs(self) -> dict:
        # Defaults chosen to avoid "silent hangs" on HPC/NFS when worker processes stall.
        # If you want max throughput, override via config.training.*
        training_cfg = self.config.get("training", {})
        num_workers = int(self.num_workers)

        pin_memory = bool(training_cfg.get("pin_memory", torch.cuda.is_available()))
        persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))
        prefetch_factor = training_cfg.get("prefetch_factor", 2 if num_workers > 0 else None)
        timeout = int(training_cfg.get("dataloader_timeout", 30))

        kwargs: dict = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        if num_workers > 0 and 0 < timeout:
            kwargs["timeout"] = timeout
        return kwargs

    def setup(self, stage=None):
        train_pairs = load_split_pairs(self.preprocessed_path, "train", self.sources)
        val_pairs = load_split_pairs(self.preprocessed_path, "val", self.sources)
        test_pairs = load_split_pairs(self.preprocessed_path, "test", self.sources)

        if self.bbox_mode:
            bboxes_df = load_bboxes_df(self.bboxes_csv_path)

            train_pos_fraction = self.config["data"].get("train_pos_fraction", None)
            if train_pos_fraction is not None:
                self.train_dataset = MixedCariesBboxPatchDataset(
                    images_df=train_pairs,
                    bboxes_df=bboxes_df,
                    transform=self.train_transform,
                    patch_size=self.patch_size,
                    pos_fraction=train_pos_fraction,
                )
            else:
                self.train_dataset = BboxPatchDataset(
                    images_df=train_pairs,
                    bboxes_df=bboxes_df,
                    transform=self.train_transform,
                    patch_size=self.patch_size,
                )
            self.val_dataset = BboxEvalDataset(
                images_df=val_pairs,
                bboxes_df=bboxes_df,
                patch_size=self.patch_size,
            )
            self.test_dataset = BboxEvalDataset(
                images_df=test_pairs,
                bboxes_df=bboxes_df,
                patch_size=self.patch_size,
            )
        else:
            self.train_dataset = FullImageDataset(
                train_pairs,
                transform=self.train_transform,
            )
            self.val_dataset = FullImageDataset(
                val_pairs,
                transform=self.val_transform,
            )
            self.test_dataset = FullImageDataset(
                test_pairs,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        shuffle = self.config["training"].get("shuffle_train", True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **self._dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1 if self.bbox_mode else self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1 if self.bbox_mode else self.batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )
