import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb

from .unet import UNet
from ..utils.metrics import DiceLoss, iou_coeff


LOGGED_IXS = np.array([0, 1, 2], dtype=np.int32)


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
            dropout=model_config.get("dropout", 0.0),
        )

        bce_pos_weight = config["training"].get("bce_pos_weight", 1.0)
        dice_weight = config["training"].get("dice_weight", [1.0, 1.0])

        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_pos_weight))
        self.dice_loss_fn = DiceLoss(weight=dice_weight)

        self.learning_rate = config["training"].get("learning_rate", 5e-4)
        self.weight_decay = config["training"].get("weight_decay", 1e-4)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, preds, targets):
        bce = self.bce_loss(preds, targets)
        dice = self.dice_loss_fn(preds, targets)
        return bce + dice, bce, dice

    def _log_predictions(self, images, masks, preds, prefix="train"):
        if not isinstance(self.logger, WandbLogger):
            return

        images_np = images[LOGGED_IXS, 0].cpu().numpy()
        masks_np = masks[LOGGED_IXS, 0].cpu().numpy()
        preds_np = torch.sigmoid(preds[LOGGED_IXS, 0]).detach().cpu().numpy()

        wandb_images = []
        for i, idx in enumerate(LOGGED_IXS):
            img = images_np[i]
            mask = masks_np[i]
            pred = preds_np[i]

            wandb_images.append(
                wandb.Image(
                    img,
                    masks={
                        "ground_truth": {
                            "mask_data": mask > 0.5,
                            "class_labels": {0: "background", 1: "caries"},
                        },
                        "predictions": {
                            "mask_data": pred > 0.5,
                            "class_labels": {0: "background", 1: "caries"},
                        },
                    },
                    caption=f"{prefix}_sample_{idx}",
                )
            )

        self.logger.experiment.log({f"{prefix}/predictions": wandb_images})

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        self.log("grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss, bce, dice = self._compute_loss(preds, masks)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/bce_loss", bce, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/dice_loss", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "lr",
            self.optimizers().optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        if self.current_epoch % 1 == 0 and batch_idx == 0:
            self._log_predictions(images, masks, preds, prefix="train")

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss, bce, dice = self._compute_loss(preds, masks)
        iou = iou_coeff(preds, masks)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bce_loss", bce, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/dice_loss", dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_predictions(images, masks, preds, prefix="val")

        return {"val_loss": loss, "val_dice_loss": dice, "val_iou": iou}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss, bce, dice = self._compute_loss(preds, masks)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bce_loss", bce, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/dice_loss", dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_dice_loss": dice}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.hparams.get("training", {}).get("lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, threshold=0.01
            )
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "monitor": "val/dice_loss",
                    "interval": "epoch",
                }
            ]
        return optimizer

    @property
    def model_instance(self):
        return self.model
