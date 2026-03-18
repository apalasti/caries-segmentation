import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset import load_split_pairs, BaseKariesDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.preprocessed_path = config["data"]["preprocessed_path"]
        self.sources = config["data"].get("sources", [])
        self.batch_size = config["training"].get("batch_size", 32)
        self.num_workers = config["training"].get("num_workers", 4)
        self.size = tuple(config["data"].get("images_size", [256, 256]))

    def setup(self, stage=None):
        self.train_dataset = BaseKariesDataset(
            load_split_pairs(self.preprocessed_path, "train", self.sources),
            size=self.size,
        )
        self.val_dataset = BaseKariesDataset(
            load_split_pairs(self.preprocessed_path, "val", self.sources),
            size=self.size,
        )
        self.test_dataset = BaseKariesDataset(
            load_split_pairs(self.preprocessed_path, "test", self.sources),
            size=self.size,
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
