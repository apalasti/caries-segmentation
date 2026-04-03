import torch
import torch.nn as nn


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
