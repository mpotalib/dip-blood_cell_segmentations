import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks running averages for logging."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Computes mean Dice over all classes (including background)."""
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dims)
    cardinality = torch.sum(pred_one_hot + target_one_hot, dims)
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Computes mean IoU over all classes (including background)."""
    pred_one_hot = torch.nn.functional.one_hot(pred, num_classes=num_classes).permute(0, 3, 1, 2)
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    dims = (0, 2, 3)
    intersection = torch.sum(pred_one_hot * target_one_hot, dims)
    union = torch.sum(pred_one_hot + target_one_hot, dims) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def save_checkpoint(state: Dict, path: Path, is_best: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best.pt"
        torch.save(state, best_path)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
