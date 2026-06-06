import torch
import torch.nn as nn
from scipy import ndimage


def compute_lesion_level_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Compute lesion-level detection metrics (instance-level).

    A lesion/caries is considered "detected" if the intersection of the predicted
    mask with the ground truth lesion segment is > threshold * area_of_gt_lesion.

    This is a lesion-level metric common in medical imaging for tumor/lesion detection,
    evaluating at the instance level rather than pixel level.

    Args:
        pred_mask: Binary predicted mask, shape (H, W) or (1, H, W)
        gt_mask: Binary ground truth mask, shape (H, W) or (1, H, W)
        threshold: Minimum intersection ratio to count as detected (default 0.5 = 50%)

    Returns:
        dict with keys: true_positive, false_positive, false_negative, true_negative
    """
    # Ensure 2D arrays
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask.squeeze(0)

    # Convert to numpy if tensors
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()

    # Ensure binary
    pred_mask = (pred_mask > 0.5).astype(int)
    gt_mask = (gt_mask > 0.5).astype(int)

    # Label connected components in GT to get individual caries instances
    gt_labels, num_gt_caries = ndimage.label(gt_mask)

    # Label connected components in prediction
    pred_labels, num_pred_regions = ndimage.label(pred_mask)

    # For each GT caries, check if it's detected
    detected_caries = 0
    matched_pred_regions = set()

    for caries_id in range(1, num_gt_caries + 1):
        # Get mask for this specific caries
        caries_mask = (gt_labels == caries_id).astype(int)
        caries_area = caries_mask.sum()

        if caries_area == 0:
            continue

        # Calculate intersection with prediction
        intersection = (caries_mask * pred_mask).sum()
        intersection_ratio = intersection / caries_area

        if intersection_ratio > threshold:
            detected_caries += 1
            # Mark which predicted regions matched this caries
            pred_in_caries = pred_labels * caries_mask
            for region_id in range(1, num_pred_regions + 1):
                if (pred_in_caries == region_id).any():
                    matched_pred_regions.add(region_id)

    # False negatives: GT caries not detected
    missed_caries = num_gt_caries - detected_caries

    # False positives: predicted regions that didn't match any GT caries
    false_positive_regions = num_pred_regions - len(matched_pred_regions)

    # Calculate metrics
    tp = detected_caries
    fn = missed_caries
    fp = max(0, false_positive_regions)

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": 0  # Not applicable for instance-level detection
    }


def compute_lesion_level_metrics_batch(pred_masks, gt_masks, threshold=0.5):
    """
    Compute lesion-level detection metrics for a batch of masks.

    Args:
        pred_masks: Binary predicted masks, shape (B, H, W) or (B, 1, H, W)
        gt_masks: Binary ground truth masks, shape (B, H, W) or (B, 1, H, W)
        threshold: Minimum intersection ratio to count as detected (default 0.5 = 50%)

    Returns:
        dict with aggregated keys: true_positive, false_positive, false_negative, true_negative
    """
    # Ensure 3D arrays (B, H, W)
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.ndim == 4:
        gt_masks = gt_masks.squeeze(1)

    batch_size = pred_masks.shape[0]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for i in range(batch_size):
        metrics = compute_lesion_level_metrics(pred_masks[i], gt_masks[i], threshold)
        total_tp += metrics["true_positive"]
        total_fp += metrics["false_positive"]
        total_fn += metrics["false_negative"]
        total_tn += metrics["true_negative"]

    return {
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn
    }


def compute_metrics(preds, masks, threshold=0.5):
    preds_bin = (torch.sigmoid(preds) > threshold).int()
    masks_bin = masks.int()

    TP = (preds_bin * masks_bin).sum().item()
    TN = ((1 - preds_bin) * (1 - masks_bin)).sum().item()
    FP = (preds_bin * (1 - masks_bin)).sum().item()
    FN = ((1 - preds_bin) * masks_bin).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall,
            "dice": dice, "iou": iou}

def dice_coeff(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_coeff(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        return 1 - dice
