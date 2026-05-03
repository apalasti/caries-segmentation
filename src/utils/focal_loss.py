import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean", task_type="binary", num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor: None, scalar, list, or tensor. For binary/multi-label segmentation
            with shape (N, C, H, W), pass a tensor of shape (C,) (registered as a buffer).
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        self.has_alpha = False

        if alpha is None:
            return

        self.has_alpha = True

        if isinstance(alpha, torch.Tensor):
            t = alpha.detach().float().clone().reshape(-1)
        elif isinstance(alpha, (list, tuple)):
            t = torch.tensor([float(x) for x in alpha], dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            t = torch.tensor([float(alpha)], dtype=torch.float32)
        else:
            raise TypeError(
                "alpha must be None, float, list, tuple, or torch.Tensor, "
                f"got {type(alpha).__name__}"
            )

        if task_type == "multi-class":
            if num_classes is None:
                raise ValueError("num_classes must be specified for multi-class focal loss")
            if t.numel() != num_classes:
                raise ValueError(
                    f"alpha length ({t.numel()}) must equal num_classes ({num_classes})"
                )
        self.register_buffer("alpha", t)

    def forward(self, inputs, targets):
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        if self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        if self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        raise ValueError(
            f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
        )

    def _broadcast_alpha_binary(self, inputs):
        """(C,) -> (1, C, 1, 1) for broadcasting with (N, C, H, W)."""
        a = self.alpha
        if a.numel() == 1:
            return a.view(1, 1, 1, 1)
        return a.view(1, -1, 1, 1)

    def binary_focal_loss(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.has_alpha:
            af = self._broadcast_alpha_binary(inputs)
            alpha_t = af * targets + (1 - af) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        ce_loss = -targets_one_hot * torch.log(probs)

        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        if self.has_alpha:
            alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.has_alpha:
            af = self._broadcast_alpha_binary(inputs)
            alpha_t = af * targets + (1 - af) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
