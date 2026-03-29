import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule


def evaluate():
    config = load_config()

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


if __name__ == "__main__":
    evaluate()
