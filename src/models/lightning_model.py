import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from .unet import UNet
from ..utils.metrics import DiceLoss, dice_coeff, iou_coeff
from ..utils.focal_loss import FocalLoss


LOGGED_IXS = np.array([0, 1, 2], dtype=np.int32)


def _normalize_bce_pos_weight(value):
    """Return a list of floats, or None if value is None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(
        "training['bce_pos_weight'] must be a float or a list of floats, "
        f"got {type(value).__name__}"
    )


def _focal_alpha_tensor(value, n_classes: int):
    """Return float tensor of shape (n_classes,) or None if value is None."""
    if value is None:
        return None
    lst = _normalize_bce_pos_weight(value)
    if len(lst) != n_classes:
        raise ValueError(
            "training['focal_alpha'] must have length equal to model['n_classes'] "
            f"({n_classes}), got len={len(lst)}: {lst!r}"
        )
    return torch.tensor(lst, dtype=torch.float32)


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.test_preds, self.test_targets =  [], []

        model_config = config.get("model", {})

        n_classes = model_config.get("n_classes", 1)
        self.model = UNet(
            n_classes=n_classes,
            n_channels=model_config.get("n_channels", 1),
            depth=model_config.get("depth", 4),
            base_channels=model_config.get("base_channels", 64),
            dropout=model_config.get("dropout", 0.0),
        )

        training = config["training"]
        modules = {}
        for n in training.get("losses", ["bce", "dice"]):
            if n == "bce":
                bce_kwargs = {}
                pw = _normalize_bce_pos_weight(training.get("bce_pos_weight"))
                if pw is not None:
                    if len(pw) != n_classes:
                        raise ValueError(
                            "training['bce_pos_weight'] must have length equal to model['n_classes'] "
                            f"({n_classes}), got len={len(pw)}: {pw!r}"
                        )
                    bce_kwargs["pos_weight"] = torch.tensor(
                        pw, dtype=torch.float32
                    )
                modules["bce"] = nn.BCEWithLogitsLoss(**bce_kwargs)
            elif n == "focal":
                fa_raw = training.get("focal_alpha", 0.25)
                fa_tensor = _focal_alpha_tensor(fa_raw, n_classes)
                modules["focal"] = FocalLoss(
                    gamma=training.get("focal_gamma", 2.0),
                    alpha=fa_tensor,
                    task_type="binary",
                )
            elif n == "dice":
                dice_weight = training.get("dice_weight", [1.0, 1.0])
                modules["dice"] = DiceLoss(weight=dice_weight)

        if not modules:
            raise ValueError("No valid losses have been provided in config['training']['losses']. At least one loss must be specified.")

        self.loss_modules = nn.ModuleDict(modules)

        self.learning_rate = config["training"].get("learning_rate", 5e-4)
        self.weight_decay = config["training"].get("weight_decay", 1e-4)

    def forward(self, x):
        return self.model(x)

    def _bce_multiplier(self) -> float:
        training = self.hparams["training"]
        term = float(training.get("bce_term_weight", 1.0))
        ramp_epochs = int(training.get("bce_ramp_epochs", 0))
        if ramp_epochs <= 0:
            return term
        ramp = min(1.0, float(self.current_epoch) / float(ramp_epochs))
        return ramp * term

    def _focal_multiplier(self) -> float:
        training = self.hparams["training"]
        term = float(training.get("focal_term_weight", 1.0))
        ramp_epochs = int(training.get("focal_ramp_epochs", 0))
        if ramp_epochs <= 0:
            return term
        ramp = min(1.0, float(self.current_epoch) / float(ramp_epochs))
        return ramp * term

    def _compute_loss(self, preds, targets):
        parts = {}
        bce_mult = None
        focal_mult = None
        for name, mod in self.loss_modules.items():
            raw = mod(preds, targets)
            if name == "bce":
                bce_mult = self._bce_multiplier()
                parts[name] = raw * bce_mult
            elif name == "focal":
                focal_mult = self._focal_multiplier()
                parts[name] = raw * focal_mult
            else:
                parts[name] = raw
        total = sum(parts.values())
        return total, parts, bce_mult, focal_mult

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
        loss, parts, bce_mult, focal_mult = self._compute_loss(preds, masks)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if bce_mult is not None:
            self.log(
                "train/bce_lambda",
                bce_mult,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        if focal_mult is not None:
            self.log(
                "train/focal_lambda",
                focal_mult,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        for name, val in parts.items():
            self.log(
                f"train/{name}_loss",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
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
        loss, parts, *_ = self._compute_loss(preds, masks)
        iou = iou_coeff(preds, masks)
        dice_score = dice_coeff(preds, masks)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, val in parts.items():
            self.log(
                f"val/{name}_loss",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == "dice"),
            )
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/dice",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx == 0:
            self._log_predictions(images, masks, preds, prefix="val")

        return {"val_loss": loss, "val_iou": iou, "val_dice": dice_score}

    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss, parts, *_ = self._compute_loss(preds, masks)

        preds = torch.sigmoid(preds)

        preds_bin = (preds > 0.5).int()

        self.test_preds.append(preds_bin.cpu())
        self.test_targets.append(masks.int().cpu())

        loss = self._compute_loss(preds, masks)
        dice = dice_coeff(preds, masks)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, val in parts.items():
            self.log(
                f"test/{name}_loss",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == "dice"),
            )

        dice = parts.get(
            "dice", torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        )
        self.log(
            "test/dice",
            dice_coeff(preds, masks),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return {"test_loss": loss, "test_dice_loss": dice}

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.hparams.get("training", {}).get("lr_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=15, threshold=0.01
            )
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "monitor": "val/dice",
                    "interval": "epoch",
                }
            ]
        return optimizer

    @property
    def model_instance(self):
        return self.model

    def on_test_epoch_end(self):

        preds = torch.cat(self.test_preds).view(-1)
        targets = torch.cat(self.test_targets).view(-1)

        TP = ((preds == 1) & (targets == 1)).sum().item()
        TN = ((preds == 0) & (targets == 0)).sum().item()
        FP = ((preds == 1) & (targets == 0)).sum().item()
        FN = ((preds == 0) & (targets == 1)).sum().item()

        cm = [[TN, FP],
              [FN, TP]]



        plt.figure(figsize=(5, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Background", "Caries"],
            yticklabels=["Background", "Caries"],
        )

        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")

        plt.savefig("confusion_matrix.png")
        plt.close()


        #wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})