import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross Entropy with Logits.
    Primary loss function used in WildfireSpreadTS benchmark (pos_class_weight ~ 10).
    """
    def __init__(self, pos_class_weight: float = 10.0):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_class_weight = pos_class_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == targets.ndim + 1:
            logits = logits.squeeze(1)
        pos_weight = torch.tensor([self.pos_class_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)


class BinaryFocalLoss(nn.Module):
    """
    Sigmoid Focal Loss to address extreme foreground-background class imbalance:
      FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == targets.ndim + 1:
            logits = logits.squeeze(1)
        targets = targets.float()
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Binary Dice Loss for spatial segmentation overlap.
    """
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == targets.ndim + 1:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        
        intersection = (probs * targets).sum(dim=(-2, -1))
        cardinality = (probs + targets).sum(dim=(-2, -1))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


def get_loss_function(name: str = "BCE", pos_class_weight: float = 10.0, **kwargs) -> nn.Module:
    """
    Factory function for loss functions.
    Supported: 'BCE', 'Focal', 'Dice'
    """
    name = name.upper()
    if name == "BCE":
        return WeightedBCEWithLogitsLoss(pos_class_weight=pos_class_weight)
    elif name == "FOCAL":
        alpha = kwargs.get("alpha", 1.0 / (1.0 + pos_class_weight))
        gamma = kwargs.get("gamma", 2.0)
        return BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif name == "DICE":
        return DiceLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}. Choose from 'BCE', 'Focal', 'Dice'.")
