import time

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger


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


class StepTimingCallback(Callback):
    """Log wall-clock train/val/test step and epoch durations under ``time/...``."""

    def __init__(self):
        self._state = {
            "train": {"step": self.empty_state(), "epoch": self.empty_state()},
            "val": {"step": self.empty_state(), "epoch": self.empty_state()},
            "test": {"step": self.empty_state(), "epoch": self.empty_state()},
        }

    def empty_state(self):
        return {"t0": 0, "sum": 0, "count": 0}

    def _log(self, trainer, pl_module: LightningModule, phase: str, kind: str):
        s = self._state[phase][kind]

        duration = time.perf_counter() - s["t0"]
        pl_module.log(
            f"time/{phase}/{kind}",
            duration,
            on_epoch=True,
            prog_bar=False,
            reduce_fx=torch.mean,
        )
        s["sum"] += duration
        s["count"] += 1

        log = trainer.logger
        if isinstance(log, WandbLogger) and 0 < s["count"]:
            summary = log.experiment.summary
            summary[f"time/{phase}/{kind}"] = s["sum"] / s["count"]

    # --- train ---
    def on_train_epoch_start(self, trainer, pl_module):
        self._state["train"]["epoch"]["t0"] = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._state["train"]["step"]["t0"] = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(trainer, pl_module, "train", "step")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "train", "epoch")

    # --- validation ---
    def on_validation_epoch_start(self, trainer, pl_module):
        self._state["val"]["epoch"]["t0"] = time.perf_counter()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._state["val"]["step"]["t0"] = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(trainer, pl_module, "val", "step")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "val", "epoch")

    # --- test ---
    def on_test_epoch_start(self, trainer, pl_module):
        self._state["test"]["epoch"]["t0"] = time.perf_counter()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._state["test"]["step"]["t0"] = time.perf_counter()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(trainer, pl_module, "test", "step")

    def on_test_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module, "test", "epoch")
