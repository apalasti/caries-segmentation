import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data.dataset import BboxEvalDataset
from .data.lightning_datamodule import load_bboxes_df, load_split_pairs
from .models.lightning_model import SegmentationLightningModule
from .utils.visualization import plot_confusion_matrix, plot_metrics_bars


def evaluate(max_samples=5, threshold=0.5, bboxes_csv=None):
    del max_samples
    config = load_config()

    seed = config["training"].get("seed", 42)
    pl.seed_everything(seed, workers=True)

    best_model_path = f"{config['training']['output_dir']}/pious-surf-49.ckpt"
    model = SegmentationLightningModule.load_from_checkpoint(best_model_path, config=config)
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
    print("saving eval to ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    preprocessed_path = config["data"]["preprocessed_path"]
    size = tuple(config["data"].get("images_size", [256, 256]))
    patch_size = int(config["data"].get("patch_size", min(size)))

    bboxes_csv_path = (
        Path(bboxes_csv)
        if bboxes_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "preprocessed" / "bboxes.csv"
    )
    test_images_df = load_split_pairs(
        preprocessed_path, "test", config["data"].get("sources", [])
    )
    dataset = BboxEvalDataset(
        images_df=test_images_df,
        bboxes_df=load_bboxes_df(str(bboxes_csv_path)),
        size=size,
        patch_size=patch_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    TP = FP = TN = FN = 0

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

    plot_confusion_matrix(metrics, save_path=str(save_dir / "confusion_matrix.png"))
    plot_metrics_bars(metrics, save_dir)
    print("Evaluation finished. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    evaluate()
