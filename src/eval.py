import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule
from .train_tooth_detection_model import evaluate_from_config as evaluate_tooth_detection_from_config


def evaluate_segmentation(config):

    wandb_logger = WandbLogger(
        project=config["wandb"]["project"],
        config=config,
    )

    data_module = SegmentationDataModule(config)
    data_module.setup("test")

    model = SegmentationLightningModule(config)

    best_model_path = f"{config['training']['output_dir']}/best_model.ckpt"
    model = model.load_from_checkpoint(best_model_path, config=config)

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
    )

    results = trainer.test(model, dataloaders=data_module.test_dataloader())

    print(f"Test Results: {results}")

    wandb_logger.experiment.finish()


def evaluate():
    config = load_config()
    task = config.get("training", {}).get("task", "segmentation")

    if task == "segmentation":
        evaluate_segmentation(config)
        return

    if task in {"tooth_detection", "detection", "yolo_unet_conjunction"}:
        metrics = evaluate_tooth_detection_from_config(config)
        print(f"Detection Test Results: {metrics}")
        return

    raise ValueError(
        f"Unsupported training.task='{task}'. "
        "Use one of: segmentation, tooth_detection, yolo_unet_conjunction."
    )


if __name__ == "__main__":
    evaluate()
