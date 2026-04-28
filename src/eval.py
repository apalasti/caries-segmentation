import os
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from .config import load_config
from .data.dataset import (
    crop_patch_from_bbox_row,
    image_array_to_tensor,
    load_image_mask_arrays,
    read_bbox_csv_rows,
)
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
    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    del num_workers  # kept for config parity

    bboxes_csv_path = (
        Path(bboxes_csv)
        if bboxes_csv is not None
        else Path(__file__).resolve().parents[1] / "data" / "preprocessed" / "bboxes.csv"
    )
    rows = read_bbox_csv_rows(str(bboxes_csv_path))
    rows_by_image: dict[str, list] = defaultdict(list)
    for row in rows:
        rows_by_image[row.image_rel].append(row)

    TP = FP = TN = FN = 0

    for _, image_rows in tqdm(rows_by_image.items(), total=len(rows_by_image)):
        first = image_rows[0]
        image_path = os.path.join(preprocessed_path, first.image_rel)
        mask_path = os.path.join(preprocessed_path, first.mask_rel)
        full_img, full_mask = load_image_mask_arrays(
            image_path,
            mask_path,
            size=size,
            transform=None,
        )
        h, w = full_img.shape[:2]
        if patch_size > h or patch_size > w:
            raise ValueError(
                f"patch_size={patch_size} must be <= resized shape ({h}, {w})"
            )
        canvas_probs = torch.zeros((h, w), device=device, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, len(image_rows), batch_size):
                chunk = image_rows[start : start + batch_size]
                batch_crops = []
                batch_origins = []
                for row in chunk:
                    img_crop, _, y0, x0 = crop_patch_from_bbox_row(
                        full_img, full_mask, row, patch_size
                    )
                    batch_crops.append(image_array_to_tensor(img_crop))
                    batch_origins.append((y0, x0))
                crops_tensor = torch.stack(batch_crops, dim=0).to(device)
                probs = torch.sigmoid(model(crops_tensor)).squeeze(1)
                for i, (y0, x0) in enumerate(batch_origins):
                    y1 = y0 + patch_size
                    x1 = x0 + patch_size
                    canvas_probs[y0:y1, x0:x1] = torch.maximum(
                        canvas_probs[y0:y1, x0:x1], probs[i]
                    )

        preds = canvas_probs > threshold
        masks = torch.from_numpy((full_mask > 0).astype(bool)).to(device)

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
    plot_metrics_bars(metrics, save_dir)
    print("Evaluation finished. Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    evaluate()
