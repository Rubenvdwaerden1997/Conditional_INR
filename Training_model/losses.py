from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from config import Config


def dice_loss(
    logits: torch.Tensor,    # [M, C]  already filtered to valid points
    targets: torch.Tensor,   # [M]
    num_classes: int,
    ignore_index: int = 255,
    ignore_background: bool = False,
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

    exclude = {ignore_index, *(([0] if ignore_background else []))}
    fg_classes = [c for c in range(num_classes) if c not in exclude]
    return dice_per_class[fg_classes].mean()


def smoothness_3d_loss(
    logits_seq: torch.Tensor,   # [B, F, C]  consecutive-frame logits
    cfg: Config,
) -> torch.Tensor:
    """3D temporal smoothness (inter-frame): penalise class-prob changes between consecutive z-frames.

    Improves pullback consistency — predictions should not flicker frame to frame.
    Device classes (stent, guidewire, …) get a relaxed weight because they
    can legitimately appear or disappear within 1–2 frames.
    NOT the right tool for within-frame (x,y) pixelation — use smoothness_2d_loss() for that.
    """
    if logits_seq.shape[1] < 2:
        return torch.tensor(0.0, device=logits_seq.device)

    probs   = torch.softmax(logits_seq, dim=-1)        # [B, F, C]
    diff    = (probs[:, 1:] - probs[:, :-1]).abs()     # [B, F-1, C]

    if cfg.smoothness_class_indices:
        weights = torch.zeros(cfg.num_classes, device=logits_seq.device)
        for idx in cfg.smoothness_class_indices:
            weights[idx] = 1.0
    else:
        weights = torch.ones(cfg.num_classes, device=logits_seq.device)

    return (diff * weights.unsqueeze(0).unsqueeze(0)).mean()


def smoothness_2d_loss(
    logits_A: torch.Tensor,   # [B, K, C]  logits at center coordinates
    logits_B: torch.Tensor,   # [B, K, C]  logits at 1-pixel-offset neighbors
    cfg: Config,
) -> torch.Tensor:
    """2D spatial smoothness (intra-frame): penalise adjacent (x,y) pixel disagreement on annotated frame.

    Directly targets per-pixel incoherence — neighbouring pixels predicting different
    classes despite similar encoder features.  Each pair (A, B) differs by exactly
    1 pixel in a random cardinal direction on the annotated frame.

    Different from smoothness_3d_loss() which is temporal (z-direction, between frames).
    Enable via cfg.smoothness_2d_weight before enabling cfg.smoothness_3d_weight.
    """
    probs_A = torch.softmax(logits_A, dim=-1)   # [B, K, C]
    probs_B = torch.softmax(logits_B, dim=-1)   # [B, K, C]
    diff    = (probs_A - probs_B).abs()          # [B, K, C]

    if cfg.smoothness_class_indices:
        weights = torch.zeros(cfg.num_classes, device=logits_A.device)
        for idx in cfg.smoothness_class_indices:
            weights[idx] = 1.0
    else:
        weights = torch.ones(cfg.num_classes, device=logits_A.device)

    return (diff * weights).mean()


def mse_onehot_loss(
    logits: torch.Tensor,    # [M, C]
    targets: torch.Tensor,   # [M]
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    valid = targets != ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)
    probs   = torch.softmax(logits[valid], dim=-1)
    one_hot = F.one_hot(targets[valid], num_classes).float()
    return F.mse_loss(probs, one_hot)


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
    smoothness_logits: Optional[torch.Tensor] = None,  # [B, F, C]  temporal/3D inter-frame
    dense_pred: Optional[torch.Tensor] = None,         # [B, C, H, W]
    spatial_logits_A: Optional[torch.Tensor] = None,   # [B, K, C]  within-frame pair A
    spatial_logits_B: Optional[torch.Tensor] = None,   # [B, K, C]  within-frame pair B
) -> Tuple[torch.Tensor, dict]:
    B, N, C   = logits.shape
    flat_logits = logits.view(B * N, C)
    flat_labels = labels.view(B * N)

    valid = flat_labels != cfg.ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=logits.device), {}

    if cfg.loss_type == "mse_onehot":
        total = mse_onehot_loss(flat_logits, flat_labels, C, cfg.ignore_index)
        log   = {"mse_onehot": total.item()}
    else:
        ce    = F.cross_entropy(flat_logits[valid], flat_labels[valid])
        dl    = dice_loss(flat_logits, flat_labels, C, cfg.ignore_index, cfg.ignore_background)
        total = ce + cfg.dice_weight * dl
        log   = {"ce": ce.item(), "dice": dl.item()}

    if smoothness_logits is not None and cfg.smoothness_3d_weight > 0:
        sl    = smoothness_3d_loss(smoothness_logits, cfg)
        total = total + cfg.smoothness_3d_weight * sl
        log["smooth_3d"] = sl.item()

    if spatial_logits_A is not None and spatial_logits_B is not None:
        ssl   = smoothness_2d_loss(spatial_logits_A, spatial_logits_B, cfg)
        total = total + cfg.smoothness_2d_weight * ssl
        log["smooth_2d"] = ssl.item()

    if dense_pred is not None and cfg.tv_weight > 0:
        tvl   = tv_loss(dense_pred)
        total = total + cfg.tv_weight * tvl
        log["tv"] = tvl.item()

    return total, log
