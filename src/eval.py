import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from .config import load_config
from .data.lightning_datamodule import SegmentationDataModule
from .models.lightning_model import SegmentationLightningModule
from .utils.metrics import compute_lesion_level_metrics_batch
from .utils.visualization import plot_confusion_matrix, plot_metrics_bars


def evaluate(checkpoint: str | None = None, threshold: float = 0.5):
    config = load_config()

    seed = config["training"].get("seed", 42)
    pl.seed_everything(seed, workers=True)

    if checkpoint is None:
        checkpoint = os.path.join(config["training"]["output_dir"], "best_model.ckpt")
    model = SegmentationLightningModule.load_from_checkpoint(checkpoint, config=config)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device", device)
    model.to(device)
    model.eval()

    save_dir = Path(__file__).parent.parent / "."
    print("saving eval to", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data_module = SegmentationDataModule(config)
    data_module.setup()
    loader = data_module.test_dataloader()

    lesion_threshold = config.get("evaluation", {}).get("lesion_detection_threshold", 0.5)

    TP = FP = TN = FN = 0
    lesion_TP = lesion_FP = lesion_FN = 0

    for batch in tqdm(loader, total=len(loader)):
        batch_on_device = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        with torch.no_grad():
            logits, _, masks = model._forward_full(batch_on_device)

        probs = torch.sigmoid(logits)
        preds = probs > threshold
        gts = masks > 0.5

        TP += ((preds == 1) & (gts == 1)).sum().item()
        TN += ((preds == 0) & (gts == 0)).sum().item()
        FP += ((preds == 1) & (gts == 0)).sum().item()
        FN += ((preds == 0) & (gts == 1)).sum().item()

        preds_bin = preds.float()
        lesion_metrics = compute_lesion_level_metrics_batch(
            preds_bin[:, 0], masks[:, 0], threshold=lesion_threshold
        )
        lesion_TP += lesion_metrics["true_positive"]
        lesion_FP += lesion_metrics["false_positive"]
        lesion_FN += lesion_metrics["false_negative"]

    lesion_recall = lesion_TP / (lesion_TP + lesion_FN + 1e-8)
    lesion_precision = lesion_TP / (lesion_TP + lesion_FP + 1e-8)
    lesion_f1 = 2 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall + 1e-8)

    metrics = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": TP / (TP + FP + 1e-8),
        "recall": TP / (TP + FN + 1e-8),
        "dice": 2 * TP / (2 * TP + FP + FN + 1e-8),
        "iou": TP / (TP + FP + FN + 1e-8),
        "lesion_recall": lesion_recall,
        "lesion_precision": lesion_precision,
        "lesion_f1": lesion_f1,
    }

    plot_confusion_matrix(metrics, save_path=str(save_dir / "confusion_matrix.png"))
    plot_metrics_bars(metrics, save_dir)

    print("Evaluation finished. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.ckpt). Defaults to <output_dir>/best_model.ckpt.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary prediction threshold (default: 0.5).",
    )
    args = parser.parse_args()
    evaluate(checkpoint=args.checkpoint, threshold=args.threshold)
