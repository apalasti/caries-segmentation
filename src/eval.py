import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule
from .utils.visualization import sample_test_predictions, visualize_prediction, plot_confusion_matrix, plot_metrics_bars


def evaluate(max_samples=5, threshold=0.5):
    config = load_config()

    seed = config["training"].get("seed", 42)
    pl.seed_everything(seed, workers=True)

    try:
        logger = WandbLogger(
            project=config["wandb"]["project"],
            config=config,
        )
    except Exception:
        logger = CSVLogger(
            save_dir=config["training"]["output_dir"],
            name="csv_logs",
        )

    data_module = SegmentationDataModule(config)
    data_module.setup("test")

    model = SegmentationLightningModule(config)

    best_model_path = f"{config['training']['output_dir']}/best_model.ckpt"
    model = SegmentationLightningModule.load_from_checkpoint(best_model_path, config=config)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Mappák létrehozása
    save_dir = Path(__file__).parent.parent / "docs/typst/figures/evaluation"
    print("saving eval to ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    samples = sample_test_predictions(model, data_module.test_dataloader(), max_samples=max_samples, device=device)
    for i, (img, gt_mask, pred_mask) in enumerate(samples):
        visualize_prediction(img, gt_mask, pred_mask, save_path=os.path.join(save_dir, f"sample_{i}.png"))

    TP = FP = TN = FN = 0
    for batch in data_module.test_dataloader():
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(images)) > threshold

        TP += ((preds == 1) & (masks == 1)).sum().item()
        TN += ((preds == 0) & (masks == 0)).sum().item()
        FP += ((preds == 1) & (masks == 0)).sum().item()
        FN += ((preds == 0) & (masks == 1)).sum().item()

    metrics = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": TP / (TP + FP + 1e-8),
        "recall": TP / (TP + FN + 1e-8),
        "dice": 2 * TP / (2 * TP + FP + FN + 1e-8),
        "iou": TP / (TP + FP + FN + 1e-8)
    }

    # 3. Confusion matrix ábrázolása és mentése
    plot_confusion_matrix(metrics, save_path=str(save_dir / "confusion_matrix.png"))
    plot_metrics_bars(metrics,save_dir)
    print("Evaluation finished. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if isinstance(logger, WandbLogger):
        logger.experiment.finish()


if __name__ == "__main__":
    evaluate()
