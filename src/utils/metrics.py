import torch
import torch.nn as nn


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
