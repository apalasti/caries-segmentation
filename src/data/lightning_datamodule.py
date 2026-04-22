import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset import load_split_pairs, BaseKariesDataset, TiledEvalKariesDataset
from .augmentations import get_train_transforms, get_val_transforms


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

        self.focused_crop_prob = float(config["data"].get("focused_crop_prob", 0.5))
        if self.patch_size is not None:
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
            get_train_transforms(aug_config, self.size)
            if self.augmentation_enabled
            else None
        )
        self.val_transform = get_val_transforms(self.size)

    def setup(self, stage=None):
        self.train_dataset = BaseKariesDataset(
            load_split_pairs(self.preprocessed_path, "train", self.sources),
            size=self.size,
            transform=self.train_transform,
            patch_size=self.patch_size,
            focused_crop_prob=self.focused_crop_prob,
            center_crop=False,
        )
        val_pairs = load_split_pairs(self.preprocessed_path, "val", self.sources)
        test_pairs = load_split_pairs(self.preprocessed_path, "test", self.sources)
        if self.patch_size is not None:
            self.val_dataset = TiledEvalKariesDataset(
                val_pairs,
                size=self.size,
                transform=self.val_transform,
                tile_size=self.patch_size,
            )
            self.test_dataset = TiledEvalKariesDataset(
                test_pairs,
                size=self.size,
                transform=self.val_transform,
                tile_size=self.patch_size,
            )
        else:
            self.val_dataset = BaseKariesDataset(
                val_pairs,
                size=self.size,
                transform=self.val_transform,
                patch_size=None,
                focused_crop_prob=self.focused_crop_prob,
                center_crop=False,
            )
            self.test_dataset = BaseKariesDataset(
                test_pairs,
                size=self.size,
                transform=self.val_transform,
                patch_size=None,
                focused_crop_prob=self.focused_crop_prob,
                center_crop=False,
            )

    def train_dataloader(self):
        shuffle = self.config["training"].get("shuffle_train", True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
