import logging
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import Config
from dataset import OCTPullbackDataset
from inference import predict_frame_2d, predict_pullback
from losses import compute_loss
from model import ConditionalINR
from utils import save_training_plots, setup_output_dir


# ---------------------------------------------------------------------------
# ITK-SnAP colormap (labels 0-11)
# ---------------------------------------------------------------------------

_ITKSNAP_COLORS = np.array([
    [  0,   0,   0],  # 0  Clear Label
    [255,   0,   0],  # 1  Lumen
    [ 63,  63,  63],  # 2  Guidewire
    [  0,   0, 255],  # 3  Intima
    [255, 255,   0],  # 4  Lipiden
    [210, 210, 210],  # 5  Calcium
    [255,   0, 255],  # 6  Media
    [255, 123,   0],  # 7  Sidebranch
    [230, 141, 230],  # 8  Thrombus
    [208, 190, 161],  # 9  Plaque rupture
    [  0, 255,   0],  # 10 Healed plaque
    [162, 162, 162],  # 11 Neovascularization
], dtype=np.float32) / 255.0

_ITKSNAP_LABELS = [
    "Clear Label", "Lumen", "Guidewire", "Intima",
    "Lipiden", "Calcium", "Media", "Sidebranch",
    "Thrombus", "Plaque rupture", "Healed plaque", "Neovascularization",
]

_ITKSNAP_CMAP = ListedColormap(_ITKSNAP_COLORS)

# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _remap_labels(seg: np.ndarray, mapping: Dict) -> np.ndarray:
    lut_size = max(int(seg.max()), max(mapping.keys())) + 1
    lut = np.arange(lut_size, dtype=np.int64)
    for old, new in mapping.items():
        lut[old] = new
    return lut[seg]


def _load_entry_for_vis(entry: Dict, label_mapping: Optional[Dict]):
    """Return (display_frame [H,W], gt_label [H,W]) for a dataset entry."""
    data = np.load(entry["file"])

    if entry.get("prebuilt"):
        frame = data["frame"].astype(np.float32)          # [C, H, W]
        label = data["label"].astype(np.int64)            # [H, W]
        img   = frame[frame.shape[0] // 2]                # middle channel for display
    else:
        vol  = data["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]
        seg  = data["Segmentation_image"]
        if seg.ndim == 4:
            seg = seg[0]
        z     = entry.get("frame_idx", 0)
        frame = vol[z].astype(np.float32)
        label = seg[z].astype(np.int64)
        img   = frame

    if label_mapping:
        label = _remap_labels(label, label_mapping)
    return frame, img, label


def _save_vis_predictions(
    vis_entries:   List[Dict],
    model:         ConditionalINR,
    cfg:           Config,
    device:        torch.device,
    epoch:         int,
    save_dir:      str,
    label_mapping: Optional[Dict] = None,
) -> None:
    """Save input / ground-truth / prediction figures for fixed val entries."""
    cmap    = _ITKSNAP_CMAP
    vis_dir = os.path.join(save_dir, f"vis_epoch_{epoch:04d}")
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    for entry in vis_entries:
        if cfg.training_mode == "2D":
            frame, img, gt_label = _load_entry_for_vis(entry, label_mapping)
            pred      = predict_frame_2d(model, frame, cfg, device)
            frame_idx = entry.get("frame_idx")
            tag       = (
                os.path.splitext(os.path.basename(entry["file"]))[0]  # prebuilt: stem encodes frame number
                if frame_idx is None
                else f"{entry['pullback']}_frame{frame_idx:04d}"
            )
        elif entry.get("prebuilt_3d"):
            data        = np.load(entry["file"], allow_pickle=False)
            patch       = data["patch"].astype(np.float32)   # [patch_z, H, W]
            gt_label    = data["label"].astype(np.int64)     # [H, W]  already remapped
            z_local     = int(data["z_local"])
            img         = patch[z_local]
            pred_patch  = predict_pullback(model, patch, cfg, device)
            pred        = pred_patch[z_local]
            tag         = os.path.splitext(os.path.basename(entry["file"]))[0]
        else:
            data = np.load(entry["file"], allow_pickle=False)
            vol  = data["Volume_input_image"]
            if vol.ndim == 4:
                vol = vol[0]
            pred_vol = predict_pullback(model, vol.astype(np.float32), cfg, device)
            ann      = entry["annotated_frames"]
            z        = ann[len(ann) // 2]       # middle annotated frame
            img      = vol[z]
            gt_raw   = data["Segmentation_image"]
            if gt_raw.ndim == 4:
                gt_raw = gt_raw[0]
            gt_label = gt_raw[z].astype(np.int64)
            if label_mapping:
                gt_label = _remap_labels(gt_label, label_mapping)
            pred = pred_vol[z]
            tag  = f"{entry['pullback']}_frame{z:04d}"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(gt_label, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1,
                       interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1,
                       interpolation="nearest")
        axes[2].set_title(f"Prediction (epoch {epoch})")
        axes[2].axis("off")

        legend_handles = [
            Patch(facecolor=_ITKSNAP_COLORS[i], edgecolor="gray",
                  label=f"{i}  {_ITKSNAP_LABELS[i]}")
            for i in range(len(_ITKSNAP_LABELS))
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.08), fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"{tag}.png"), dpi=100,
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _sample_smoothness_logits(
    model: ConditionalINR,
    feat_vol: torch.Tensor,    # [B, feat_ch, D, H, W] — pre-computed, detached
    skips: list,               # intermediate encoder maps, detached
    volume_shape,              # full volume shape (B, 1, D, H, W)
    cfg: Config,
    device: torch.device,
    n_spatial: int = 8,
) -> torch.Tensor:
    """Compute smoothness logits using pre-computed encoder features.

    Reuses the encoder output from the main forward pass (detached) so the
    encoder runs only once per batch and its gradient is not doubled.
    Returns [B, D, C] logits averaged over n_spatial random spatial positions.
    """
    B, _, D, H, W = volume_shape
    z_all = torch.arange(D, dtype=torch.float32, device=device)

    xs = torch.rand(n_spatial, device=device) * (W - 1)
    ys = torch.rand(n_spatial, device=device) * (H - 1)

    coords_list = []
    for k in range(n_spatial):
        coords_list.append(torch.stack([
            torch.full_like(z_all, xs[k]),
            torch.full_like(z_all, ys[k]),
            z_all,
        ], dim=-1))                                                 # [D, 3]
    coords_all   = torch.cat(coords_list, dim=0)                   # [n_spatial*D, 3]
    coords_batch = coords_all.unsqueeze(0).expand(B, -1, -1)       # [B, n_spatial*D, 3]

    logits_flat = model.decode_3d(feat_vol, skips, coords_batch, (D, H, W))
    logits      = logits_flat.view(B, n_spatial, D, cfg.num_classes)
    return logits.mean(dim=1)                                       # [B, D, C]


def train_one_epoch(
    model: ConditionalINR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> tuple:
    import time
    model.train()
    total_loss = 0.0
    log_accum  = {"ce": 0.0, "dice": 0.0, "smoothness": 0.0, "feat_sup": 0.0, "tv": 0.0}
    n_batches  = 0

    for volume, coords, labels, full_label in loader:
        t0         = time.time()
        volume     = volume.to(device)
        coords     = coords.to(device)
        labels     = labels.to(device)
        full_label = full_label.to(device)

        if cfg.training_mode == "3D":
            feat_vol, skips = model.encoder(volume)
            logits          = model.decode_3d(feat_vol, skips, coords, volume.shape[2:])
            smooth_logits   = _sample_smoothness_logits(
                model, feat_vol.detach(), [s.detach() for s in skips],
                volume.shape, cfg, device,
            )
            dp = None
        else:
            if cfg.feature_supervision:
                logits, dense_pred = model(volume, coords, return_feat_logits=True)
            else:
                logits = model(volume, coords)
            smooth_logits = None
            dp = dense_pred if cfg.feature_supervision else None
        loss, log = compute_loss(logits, labels, cfg, smooth_logits, dp)

        if cfg.feature_supervision and full_label.numel() > 0:
            # Dense CE over the full annotated frame [B, num_classes, H, W] vs [B, H, W]
            feat_loss = F.cross_entropy(dense_pred, full_label, ignore_index=cfg.ignore_index)
            loss = loss + cfg.feature_supervision_weight * feat_loss
            log["feat_sup"] = feat_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in log.items():
            log_accum[k] = log_accum.get(k, 0.0) + v
        n_batches += 1

        if n_batches == 1 and logger is not None:
            batch_time = time.time() - t0
            est_epoch  = batch_time * len(loader)
            logger.info(
                f"First batch done in {batch_time:.1f}s — "
                f"est. {est_epoch / 60:.1f} min/epoch "
                f"({len(loader)} batches)"
            )

    avg_loss = total_loss / max(n_batches, 1)
    avg_log  = {k: v / max(n_batches, 1) for k, v in log_accum.items()}
    return avg_loss, avg_log


@torch.no_grad()
def validate(
    model: ConditionalINR,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> tuple:
    """Pixel-weighted loss + global Dice (all val pixels pooled before averaging).

    Pooling predictions across all frames before computing Dice eliminates the
    per-frame averaging noise where a tiny frame with 50 pixels gets equal weight
    to a large frame with 30 000 pixels.
    """
    model.eval()
    total_weighted_loss = 0.0
    total_valid_pixels  = 0
    all_preds: List[torch.Tensor] = []
    all_lbls:  List[torch.Tensor] = []

    for volume, coords, labels, _ in loader:
        volume = volume.to(device)
        coords = coords.to(device)
        labels = labels.to(device)

        logits = model(volume, coords)
        loss, _ = compute_loss(logits, labels, cfg)

        flat_labels = labels.view(-1)
        valid       = flat_labels != cfg.ignore_index
        n_valid     = int(valid.sum().item())

        total_weighted_loss += loss.item() * n_valid
        total_valid_pixels  += n_valid

        preds = logits.argmax(dim=-1).view(-1)
        all_preds.append(preds[valid].cpu())
        all_lbls.append(flat_labels[valid].cpu())

    avg_loss = total_weighted_loss / max(total_valid_pixels, 1)

    # Global Dice: pool all valid pixels from all frames, then compute per-class Dice
    all_preds_t = torch.cat(all_preds)   # [N_total]
    all_lbls_t  = torch.cat(all_lbls)    # [N_total]  (no ignore_index values present)

    dice_per_class = []
    for c in range(cfg.num_classes):
        if c == cfg.ignore_index or c == 0:   # 0 = background: in loss, not in metric
            dice_per_class.append(float("nan"))
            continue
        pred_c = all_preds_t == c
        true_c = all_lbls_t  == c
        tp     = (pred_c & true_c).sum().float()
        denom  = pred_c.sum().float() + true_c.sum().float()
        dice_per_class.append(float("nan") if denom == 0 else (2 * tp / denom).item())

    valid_dice = [v for v in dice_per_class if not math.isnan(v)]
    mean_dice  = float(np.mean(valid_dice)) if valid_dice else 0.0

    return avg_loss, mean_dice, dice_per_class


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    train_ds:      OCTPullbackDataset,
    val_ds:        OCTPullbackDataset,
    cfg:           Config,
    device:        str = "cuda" if torch.cuda.is_available() else "cpu",
    logger:        Optional[logging.Logger] = None,
    config_path:   Optional[str] = None,
    label_mapping: Optional[Dict] = None,
) -> ConditionalINR:
    if logger is None:
        logger = setup_output_dir(cfg, config_path)

    device = torch.device(device)
    logger.info(f"Device: {device}")
    logger.info(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    model     = ConditionalINR(cfg).to(device)

    def _count_params(module):
        return sum(p.numel() for p in module.parameters())

    total_params   = _count_params(model)
    encoder_params = _count_params(model.encoder)
    inr_params     = _count_params(model.inr)
    logger.info(
        f"Model parameters — "
        f"total: {total_params:,}  |  "
        f"encoder: {encoder_params:,}  |  "
        f"INR: {inr_params:,}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)

    warmup = LinearLR(
        optimizer,
        start_factor = cfg.warmup_start_factor,
        end_factor   = 1.0,
        total_iters  = cfg.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max   = max(1, cfg.num_epochs - cfg.warmup_epochs),
        eta_min = cfg.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [cfg.warmup_epochs],
    )
    logger.info(
        f"LR schedule: warmup {cfg.warmup_epochs} ep "
        f"({cfg.lr * cfg.warmup_start_factor:.2e} → {cfg.lr:.2e}), "
        f"then cosine → {cfg.lr * 0.01:.2e}"
    )

    # Fixed validation entries used for periodic visual inspection
    vis_entries = val_ds.entries[:10]

    history: Dict = {
        "epoch": [], "train_loss": [], "ce_loss": [], "dice_loss": [], "smooth_loss": [],
        "feat_sup_loss": [], "tv_loss": [],
        "val_epoch": [], "val_loss": [], "val_dice_mean": [], "val_dice_per_class": [],
    }
    best_dice = 0.0
    logger.info("Starting training...")
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_log = train_one_epoch(model, train_loader, optimizer, cfg, device, logger)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["ce_loss"].append(train_log.get("ce", 0.0))
        history["dice_loss"].append(train_log.get("dice", 0.0))
        history["smooth_loss"].append(train_log.get("smoothness", 0.0))
        history["feat_sup_loss"].append(train_log.get("feat_sup", 0.0))
        history["tv_loss"].append(train_log.get("tv", 0.0))

        log_str = (
            f"Epoch {epoch:03d}/{cfg.num_epochs} | "
            f"Train loss: {train_loss:.4f} "
            f"(CE={train_log.get('ce', 0):.3f} "
            f"Dice={train_log.get('dice', 0):.3f} "
            f"Smooth={train_log.get('smoothness', 0):.3f}"
        )
        if cfg.feature_supervision:
            log_str += f" FeatSup={train_log.get('feat_sup', 0):.3f}(×{cfg.feature_supervision_weight})"
        if cfg.tv_weight > 0:
            log_str += f" TV={train_log.get('tv', 0):.3f}(×{cfg.tv_weight})"
        log_str += ")"

        if epoch % cfg.val_interval == 0 or epoch == cfg.num_epochs:
            val_loss, val_dice, val_dice_per_class = validate(model, val_loader, cfg, device)

            history["val_epoch"].append(epoch)
            history["val_loss"].append(val_loss)
            history["val_dice_mean"].append(val_dice)
            history["val_dice_per_class"].append(val_dice_per_class)

            log_str += f" | Val loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"

            if val_dice > best_dice:
                best_dice = val_dice
                ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.best_ckpt_name)
                torch.save({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_dice":  val_dice,
                    "cfg":       cfg,
                }, ckpt_path)
                log_str += f"  ✓ best (Dice={best_dice:.4f})"

            save_training_plots(history, cfg.checkpoint_dir, cfg.feature_supervision,
                                tv_weight=cfg.tv_weight,
                                class_names=_ITKSNAP_LABELS, class_colors=_ITKSNAP_COLORS)

        if epoch % max(1, cfg.num_epochs // 10) == 0 or epoch == 5:
            _save_vis_predictions(
                vis_entries, model, cfg, device, epoch,
                cfg.checkpoint_dir, label_mapping=label_mapping,
            )
            logger.info(f"Saved prediction visualisations → vis_epoch_{epoch:04d}/")
            model.train()   # restore train mode after eval inside _save_vis_predictions

        logger.info(log_str)

    logger.info(f"Training complete. Best val Dice: {best_dice:.4f}")
    return model
