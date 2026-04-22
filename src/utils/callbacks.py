from pytorch_lightning.callbacks import Callback


class MetricsHistoryCallback(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_dice = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train/loss" in metrics:
            self.train_loss.append(metrics["train/loss"].cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val/loss" in metrics:
            self.val_loss.append(metrics["val/loss"].cpu().item())
        if "val/dice" in metrics:
            self.val_dice.append(metrics["val/dice"].cpu().item())