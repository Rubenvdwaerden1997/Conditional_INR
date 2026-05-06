import math
from typing import List, Tuple

import numpy as np
import torch


@torch.no_grad()
def compute_dice(
    logits: torch.Tensor,   # [B, N, C]
    labels: torch.Tensor,   # [B, N]
    num_classes: int,
    ignore_index: int = 255,
    background_class: int = 0,
) -> Tuple[float, List[float]]:
    """Returns mean Dice (scalar) and per-class Dice list (nan for excluded classes).

    ignore_index pixels are excluded from evaluation entirely.
    background_class is included in the loss but excluded from the metric.
    """
    preds = logits.argmax(dim=-1).view(-1)
    lbls  = labels.view(-1)

    valid = lbls != ignore_index
    preds = preds[valid]
    lbls  = lbls[valid]

    dice_per_class = []
    for c in range(num_classes):
        if c == ignore_index or c == background_class:
            dice_per_class.append(float("nan"))
            continue
        pred_c = preds == c
        true_c = lbls  == c
        tp    = (pred_c & true_c).sum().float()
        denom = pred_c.sum().float() + true_c.sum().float()
        dice_per_class.append(float("nan") if denom == 0 else (2 * tp / denom).item())

    valid_dice = [v for v in dice_per_class if not math.isnan(v)]
    mean_dice = float(np.mean(valid_dice)) if valid_dice else 0.0
    return mean_dice, dice_per_class
