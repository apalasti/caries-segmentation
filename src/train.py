import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule


def train():
    config = load_config()

    wandb_logger = WandbLogger(
        project=config["wandb"]["project"],
        config=config,
    )

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    data_module = SegmentationDataModule(config)
    model = SegmentationLightningModule(config)

    callbacks = []

    if config["training"].get("checkpointing", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["training"]["output_dir"],
            filename="best_model",
            monitor="val/dice",
            mode="max",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    if config["training"].get("early_stopping", True):
        early_stop_callback = EarlyStopping(
            monitor="val/dice",
            patience=config["training"].get("early_stopping_patience", 10),
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    trainer = pl.Trainer(
        max_epochs=config["training"].get("epochs", 50),
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=config["training"]["output_dir"],
    )

    trainer.fit(model, datamodule=data_module)

    wandb_logger.experiment.finish()


if __name__ == "__main__":
    train()
