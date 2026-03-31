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
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        if self.weight is not None:
            weights = torch.tensor(self.weight, device=inputs.device)
            targets_weighted = targets * (weights[1] - weights[0]) + weights[0]
            intersection = (inputs * targets * targets_weighted).sum()
            dice = (2.0 * intersection + smooth) / (
                (inputs * targets_weighted).sum()
                + (targets * targets_weighted).sum()
                + smooth
            )
        else:
            intersection = (inputs * targets).sum()
            dice = (2.0 * intersection + smooth) / (
                inputs.sum() + targets.sum() + smooth
            )

        return 1 - dice
