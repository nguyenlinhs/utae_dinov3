import numpy as np
import torch


def compute_average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes Average Precision (Area Under Precision-Recall Curve),
    primary benchmark metric reported in WACV 2026 paper.
    """
    probs = torch.sigmoid(logits).detach().cpu().flatten().numpy()
    targets = targets.detach().cpu().flatten().numpy().astype(int)
    
    if np.sum(targets) == 0:
        return 0.0
        
    order = np.argsort(probs)[::-1]
    targets = targets[order]
    
    tp = np.cumsum(targets)
    fp = np.cumsum(1 - targets)
    
    recalls = tp / np.sum(targets)
    precisions = tp / (tp + fp)
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Computes Precision, Recall, F1-Score, and IoU.
    """
    probs = torch.sigmoid(logits).detach().cpu().flatten()
    targets = targets.detach().cpu().flatten().float()
    preds = (probs >= threshold).float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    ap = compute_average_precision(logits, targets)
    
    return {
        "AveragePrecision": ap,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "IoU": iou,
    }
