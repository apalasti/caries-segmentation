from pytorch_lightning.callbacks import Callback

class MetricsHistoryCallback(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_dice = []
        self.val_iou = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_loss.append(trainer.callback_metrics["train/loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_loss.append(trainer.callback_metrics["val/loss"].item())
        self.val_dice.append(trainer.callback_metrics["val/dice"].item())
        self.val_iou.append(trainer.callback_metrics["val/iou"].item())