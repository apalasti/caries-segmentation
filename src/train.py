import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule


def train():
    config = load_config()

    seed = config["training"].get("seed", 42)
    pl.seed_everything(seed, workers=True)

    enable_logging = os.environ.get("WANDB_LOGGING", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    if enable_logging:
        logger = WandbLogger(
            project=config["wandb"]["project"],
            config=config,
        )
    else:
        logger = CSVLogger(
            save_dir=config["training"]["output_dir"],
            name="csv_logs",
        )

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    data_module = SegmentationDataModule(config)
    model = SegmentationLightningModule(config)

    callbacks = []

    if config["training"].get("checkpointing", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["training"]["output_dir"],
            filename="best_model",
            monitor="val/dice_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    if config["training"].get("early_stopping", True):
        early_stop_callback = EarlyStopping(
            monitor="val/dice_loss",
            patience=config["training"].get("early_stopping_patience", 10),
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    #callbacks.append(DeviceStatsMonitor())
    callbacks.append(RichProgressBar())

    trainer = pl.Trainer(
        max_epochs=config["training"].get("epochs", 50),
        limit_train_batches=config["training"].get("limit_train_batches", None),
        limit_val_batches=config["training"].get("limit_val_batches", None),
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=config["training"]["output_dir"],
    )

    trainer.fit(model, datamodule=data_module)

    if isinstance(logger, WandbLogger):
        logger.experiment.finish()


if __name__ == "__main__":
    train()
