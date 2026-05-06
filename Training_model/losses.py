from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config


def dice_loss(
    logits: torch.Tensor,    # [M, C]  already filtered to valid points
    targets: torch.Tensor,   # [M]
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = targets != ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)

    logits  = logits[valid]
    targets = targets[valid]

    probs   = torch.softmax(logits, dim=-1)            # [M, C]
    one_hot = F.one_hot(targets, num_classes).float()  # [M, C]

    inter = (probs * one_hot).sum(dim=0)               # [C]
    union = probs.sum(dim=0) + one_hot.sum(dim=0)      # [C]
    dice_per_class = 1 - (2 * inter + eps) / (union + eps)

    # Exclude the ignore class — it has no true positives after filtering,
    # so its term is always ~1.0 early in training and drifts to 0 spuriously.
    fg_classes = [c for c in range(num_classes) if c != ignore_index]
    return dice_per_class[fg_classes].mean()


def smoothness_loss(
    logits_seq: torch.Tensor,   # [B, F, C]  consecutive-frame logits
    cfg: Config,
) -> torch.Tensor:
    """Penalise large changes in class distribution between consecutive frames.

    Device classes (stent, guidewire, …) get a relaxed weight because they
    can legitimately appear or disappear within 1–2 frames.
    """
    if logits_seq.shape[1] < 2:
        return torch.tensor(0.0, device=logits_seq.device)

    probs   = torch.softmax(logits_seq, dim=-1)        # [B, F, C]
    diff    = (probs[:, 1:] - probs[:, :-1]).abs()     # [B, F-1, C]

    weights = torch.ones(cfg.num_classes, device=logits_seq.device)
    for idx in cfg.device_class_indices:
        weights[idx] = cfg.device_smoothness_weight / cfg.smoothness_weight

    return (diff * weights.unsqueeze(0).unsqueeze(0)).mean()


def tv_loss(dense_pred: torch.Tensor) -> torch.Tensor:
    """Spatial total variation on dense logits [B, C, H, W].

    Penalises large probability differences between adjacent pixels, nudging the
    encoder toward spatially coherent features.  L1 TV promotes piecewise-flat
    regions (i.e. consistent class predictions) without blurring boundaries as
    aggressively as L2 would.
    """
    probs = torch.softmax(dense_pred, dim=1)           # [B, C, H, W]
    dh    = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
    dw    = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
    return dh + dw


def compute_loss(
    logits: torch.Tensor,                              # [B, N, C]
    labels: torch.Tensor,                              # [B, N]
    cfg: Config,
    smoothness_logits: Optional[torch.Tensor] = None,  # [B, F, C]
    dense_pred: Optional[torch.Tensor] = None,         # [B, C, H, W]
) -> Tuple[torch.Tensor, dict]:
    B, N, C   = logits.shape
    flat_logits = logits.view(B * N, C)
    flat_labels = labels.view(B * N)

    valid = flat_labels != cfg.ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=logits.device), {}

    ce = F.cross_entropy(flat_logits[valid], flat_labels[valid])
    dl = dice_loss(flat_logits, flat_labels, C, cfg.ignore_index)

    total = ce + cfg.dice_weight * dl
    log   = {"ce": ce.item(), "dice": dl.item()}

    if smoothness_logits is not None:
        sl    = smoothness_loss(smoothness_logits, cfg)
        total = total + cfg.smoothness_weight * sl
        log["smoothness"] = sl.item()

    if dense_pred is not None and cfg.tv_weight > 0:
        tvl   = tv_loss(dense_pred)
        total = total + cfg.tv_weight * tvl
        log["tv"] = tvl.item()

    return total, log
