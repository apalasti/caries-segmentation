import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lightning_model import SegmentationLightningModule
from src.data.lightning_datamodule import SegmentationDataModule
from src.utils.metrics import iou_coeff
from src.config import load_config


def compute_weighted_dice(preds, targets, weights=[1.0, 400.0], smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)

    weights_tensor = torch.tensor(weights, device=preds.device)
    targets_weighted = (
        targets * (weights_tensor[1] - weights_tensor[0]) + weights_tensor[0]
    )

    intersection = (preds * targets * targets_weighted).sum()
    dice = (2.0 * intersection + smooth) / (
        (preds * targets_weighted).sum() + (targets * targets_weighted).sum() + smooth
    )
    return dice


def compute_hard_dice(preds, targets, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(preds) > threshold).float()
    intersection = (pred * targets).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + targets.sum() + smooth)


def compute_soft_iou(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_to_iou(dice):
    return dice / (2 - dice) if dice < 2 else 0


def main():
    config = load_config("config.toml")

    print("Loading checkpoint...")
    model = SegmentationLightningModule.load_from_checkpoint(
        "best_model-v1.ckpt",
        config=config,
    )
    model.eval()
    model.freeze()

    print("Loading validation data...")
    data_module = SegmentationDataModule(config)
    data_module.setup(stage="val")
    val_loader = data_module.val_dataloader()

    print(f"Validation batches: {len(val_loader)}")

    if model.dice_loss_fn is None:
        raise ValueError(
            "config must include 'dice' in training.losses for this script "
            "(needed for model.dice_loss_fn)."
        )

    dice_loss_list = []
    weighted_dice_list = []
    hard_dice_list = []
    iou_hard_list = []
    soft_iou_list = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            preds = model(images)

            dice_loss_val = model.dice_loss_fn(preds, masks)
            weighted_dice_val = compute_weighted_dice(preds, masks)
            hard_dice_val = compute_hard_dice(preds, masks)
            iou_hard_val = iou_coeff(preds, masks)
            soft_iou_val = compute_soft_iou(preds, masks)

            dice_loss_list.append(dice_loss_val.item())
            weighted_dice_list.append(weighted_dice_val.item())
            hard_dice_list.append(hard_dice_val.item())
            iou_hard_list.append(iou_hard_val.item())
            soft_iou_list.append(soft_iou_val.item())

            print(f"  Batch {batch_idx + 1}/{len(val_loader)}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    avg_dice_loss = sum(dice_loss_list) / len(dice_loss_list)
    avg_weighted_dice = sum(weighted_dice_list) / len(weighted_dice_list)
    avg_hard_dice = sum(hard_dice_list) / len(hard_dice_list)
    avg_iou_hard = sum(iou_hard_list) / len(iou_hard_list)
    avg_soft_iou = sum(soft_iou_list) / len(soft_iou_list)

    print(f"\nDice Loss (from model):           {avg_dice_loss:.4f}")
    print(f"  -> Dice from loss:               {1 - avg_dice_loss:.4f}")
    print(f"\nWeighted Dice (like loss):         {avg_weighted_dice:.4f}")
    print(f"  -> Expected IoU (weighted):     {dice_to_iou(avg_weighted_dice):.4f}")
    print(f"\nHard Dice (threshold=0.5):         {avg_hard_dice:.4f}")
    print(f"  -> Expected IoU from hard Dice:  {dice_to_iou(avg_hard_dice):.4f}")
    print(f"\nIoU (hard predictions):            {avg_iou_hard:.4f}")
    print(f"IoU (soft predictions):            {avg_soft_iou:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\nWeighted Dice = {avg_weighted_dice:.4f}")
    print(f"Hard Dice (unweighted) = {avg_hard_dice:.4f}")
    print(f"\nIoU (hard predictions): {avg_iou_hard:.4f}")
    print(f"Expected IoU from hard Dice: {dice_to_iou(avg_hard_dice):.4f}")
    print(f"Ratio (actual/expected): {avg_iou_hard / dice_to_iou(avg_hard_dice):.4f}")


if __name__ == "__main__":
    main()
