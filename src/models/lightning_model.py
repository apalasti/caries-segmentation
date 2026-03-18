import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import wandb

from .unet import UNet
from ..utils.metrics import DiceLoss, dice_coeff


indices = np.array(torch.randint(0, 1000, (3,)).tolist(), dtype=np.int32)


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        model_config = config.get("model", {})
        self.model = UNet(
            n_channels=model_config.get("n_channels", 1),
            n_classes=model_config.get("n_classes", 1),
            depth=model_config.get("depth", 4),
            base_channels=model_config.get("base_channels", 64),
        )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss()

        self.learning_rate = config["training"].get("learning_rate", 5e-4)
        self.weight_decay = config["training"].get("weight_decay", 1e-4)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, preds, targets):
        return self.bce_loss(preds, targets) + self.dice_loss_fn(preds, targets)

    def _log_predictions(self, images, masks, preds, prefix="train"):
        global indices

        indices = indices % images.size(0)
        images_np = images[indices, 0].cpu().numpy()
        masks_np = masks[indices, 0].cpu().numpy()
        preds_np = torch.sigmoid(preds[indices, 0]).detach().cpu().numpy()

        wandb_images = []
        for i, idx in enumerate(indices):
            img = images_np[i]
            mask = masks_np[i]
            pred = preds_np[i]

            wandb_images.append(
                wandb.Image(
                    np.concatenate([img, mask, pred], axis=1),
                    masks={
                        "ground_truth": {
                            "mask_data": mask > 0.5,
                            "class_labels": {0: "background", 1: "caries"},
                        },
                        "prediction": {
                            "mask_data": pred > 0.5,
                            "class_labels": {0: "background", 1: "caries"},
                        },
                    },
                    caption=f"{prefix}_sample_{idx}",
                )
            )

        self.logger.experiment.log({f"{prefix}/predictions": wandb_images})

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self._compute_loss(preds, masks)
        dice = dice_coeff(preds, masks)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        if self.current_epoch % 1 == 0 and batch_idx == 0:
            self._log_predictions(images, masks, preds, prefix="train")

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self._compute_loss(preds, masks)
        dice = dice_coeff(preds, masks)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_predictions(images, masks, preds, prefix="val")

        return {"val_loss": loss, "val_dice": dice}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self._compute_loss(preds, masks)
        dice = dice_coeff(preds, masks)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_dice": dice}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.hparams.get("training", {}).get("lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5
            )
            return [optimizer], [
                {"scheduler": scheduler, "monitor": "val/dice", "interval": "epoch"}
            ]
        return optimizer

    @property
    def model_instance(self):
        return self.model
