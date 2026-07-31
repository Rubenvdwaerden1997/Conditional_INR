import logging
import math
import os
from typing import Dict, List, Optional, Tuple

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
from inference import predict_frame_unconditional, predict_pullback, predict_pullback_unconditional
from losses import compute_loss, dice_loss
from model import ConditionalINR, UnconditionalINR
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
    split:         str = "val",
) -> None:
    """Save input / ground-truth / prediction figures for fixed entries."""
    cmap    = _ITKSNAP_CMAP
    vis_dir = os.path.join(save_dir, f"vis_{split}_epoch_{epoch:04d}")
    os.makedirs(vis_dir, exist_ok=True)

    model.eval()

    # Unconditional mode: all vis_entries point to the same volume — show each
    # annotated frame once instead of repeating the same entry N times.
    if cfg.unconditional:
        entry    = vis_entries[0]
        data     = np.load(entry["file"], allow_pickle=False)
        vol      = data["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]
        segm = data["Segmentation_image"]
        if segm.ndim == 4:
            segm = segm[0]
        segm = segm.astype(np.int64)
        D, H, W  = vol.shape

        for z in entry["annotated_frames"]:
            gt_label = segm[z].astype(np.int64)
            if label_mapping:
                gt_label = _remap_labels(gt_label, label_mapping)
            img  = vol[z]
            pred = predict_frame_unconditional(model, (D, H, W), z, cfg, device)
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
        return

    for entry in vis_entries:
        pred_dense = None
        if entry.get("prebuilt_3d"):
            data        = np.load(entry["file"], allow_pickle=False)
            patch       = data["patch"].astype(np.float32)   # [patch_z, H, W]
            z_local     = int(data["z_local"])
            gt_label    = data["labels"][z_local].astype(np.int64)  # [H, W]  already remapped
            img         = patch[z_local]
            pred_patch  = predict_pullback(model, patch, cfg, device)
            pred        = pred_patch[z_local]
            tag         = os.path.splitext(os.path.basename(entry["file"]))[0]

            if cfg.use_dense_decoder and model.dense_decoder is not None:
                with torch.no_grad():
                    vol_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    if cfg.dense_decoder_skip_connections and cfg.encoder_depth == 5:
                        fv, _, l1, l2, l3, l4 = model.encoder.forward_deep_multiscale(vol_t)
                        d_logits = model.dense_decoder(fv, layer1=l1, layer2=l2, layer3=l3, layer4=l4)
                    elif cfg.dense_decoder_skip_connections:
                        fv, _, l1, l2 = model.encoder.forward_multiscale(vol_t)
                        d_logits = model.dense_decoder(fv, layer1=l1, layer2=l2)   # [1, C, pz', H', W']
                    else:
                        fv, _    = model.encoder(vol_t)
                        d_logits = model.dense_decoder(fv)           # [1, C, pz', H', W'] — may be smaller than patch (see DenseDecoder3D note)
                    if d_logits.shape[2:] != patch.shape:
                        d_logits = F.interpolate(
                            d_logits, size=patch.shape, mode="trilinear", align_corners=False,
                        )
                pred_dense = d_logits.argmax(dim=1).squeeze(0).cpu().numpy()[z_local]
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

        n_panels = 4 if pred_dense is not None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(gt_label, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1,
                       interpolation="nearest")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1,
                       interpolation="nearest")
        axes[2].set_title(f"INR (epoch {epoch})")
        axes[2].axis("off")
        if pred_dense is not None:
            axes[3].imshow(pred_dense, cmap=cmap, vmin=0, vmax=cfg.num_classes - 1,
                           interpolation="nearest")
            axes[3].set_title(f"Dense Decoder (epoch {epoch})")
            axes[3].axis("off")

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

def _sample_smoothness_3d_logits(
    model: ConditionalINR,
    feat_vol:    torch.Tensor,   # [B, feat_ch, D, H, W] — pre-computed, detached
    global_feat: torch.Tensor,   # [B, global_ch] — pre-computed, detached
    volume_shape,                # full volume shape (B, 1, D, H, W)
    cfg: Config,
    device: torch.device,
    n_spatial: int = 8,
    layer1: Optional[torch.Tensor] = None,   # pre-computed, detached — only if cfg.use_intermediate_features
    layer2: Optional[torch.Tensor] = None,
    layer3: Optional[torch.Tensor] = None,   # encoder_depth=5 only
    layer4: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporal smoothness (3D inter-frame): sample logits at all z-frames for random (x,y) positions.

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

    logits_flat = model.decode_3d(feat_vol, global_feat, coords_batch, (D, H, W),
                                   layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
    logits      = logits_flat.view(B, n_spatial, D, cfg.num_classes)
    return logits.mean(dim=1)                                       # [B, D, C]


def _sample_smoothness_2d_logits(
    model: ConditionalINR,
    feat_vol:    torch.Tensor,   # [B, feat_ch, D, H, W] — NOT detached (encoder gets gradient)
    global_feat: torch.Tensor,   # [B, global_ch] — NOT detached
    volume_shape: tuple,         # (B, C, D, H, W)
    ann_z: torch.Tensor,      # [B] annotated frame z-index per batch item
    cfg: Config,
    device: torch.device,
    layer1: Optional[torch.Tensor] = None,   # NOT detached — only if cfg.use_intermediate_features
    layer2: Optional[torch.Tensor] = None,
    layer3: Optional[torch.Tensor] = None,   # encoder_depth=5 only
    layer4: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D spatial smoothness (intra-frame): sample adjacent (x,y) coordinate pairs.

    Each pair (A, B) is 1 pixel apart in a random cardinal direction on the annotated frame.
    Unlike _sample_smoothness_3d_logits, encoder features are NOT detached so the spatial
    smoothness loss can also shape the encoder's feature map.
    Returns logits_A [B, K, C] and logits_B [B, K, C].
    """
    B, _, D, H, W = volume_shape
    K = cfg.smoothness_2d_n_pairs

    # Keep 1-pixel margin so the offset neighbor always stays in bounds
    xs = torch.randint(1, W - 1, (K,), device=device).float()   # [K]
    ys = torch.randint(1, H - 1, (K,), device=device).float()   # [K]

    # Randomly offset each pair in either x or y direction
    use_x = torch.rand(K, device=device) > 0.5   # [K]
    dx = use_x.float()        # 1.0 → x-offset,  0.0 → y-offset
    dy = (~use_x).float()     # 0.0 → x-offset,  1.0 → y-offset

    z_b  = ann_z.float().unsqueeze(1).expand(B, K)   # [B, K]
    xs_b = xs.unsqueeze(0).expand(B, K)               # [B, K]
    ys_b = ys.unsqueeze(0).expand(B, K)               # [B, K]
    dx_b = dx.unsqueeze(0).expand(B, K)               # [B, K]
    dy_b = dy.unsqueeze(0).expand(B, K)               # [B, K]

    coords_A = torch.stack([xs_b,        ys_b,        z_b], dim=-1)   # [B, K, 3]
    coords_B = torch.stack([xs_b + dx_b, ys_b + dy_b, z_b], dim=-1)   # [B, K, 3]

    logits_A = model.decode_3d(feat_vol, global_feat, coords_A, (D, H, W),
                                layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)   # [B, K, C]
    logits_B = model.decode_3d(feat_vol, global_feat, coords_B, (D, H, W),
                                layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)   # [B, K, C]
    return logits_A, logits_B


# Radial anatomical-ordering prior (2026-07-24). Class indices from
# dataset.py's _DEFAULT_LABEL_MAPPING (fixed taxonomy, not a per-experiment
# hyperparameter). Media is the outermost vessel-wall layer modeled — nothing
# except Background should be predicted at a larger radius than a confidently-
# identified Media point on the same ray. Two classes are deliberately excluded
# from "forbidden beyond Media": Guidewire (its shadow legitimately extends past
# the wall into what would otherwise be background — see the truncation loss
# below, which handles Guidewire on its own terms) and Sidebranch (a branch
# ostium can legitimately look like an opening beyond the wall).
_RADIAL_MEDIA_CLASS = 6
_RADIAL_FORBIDDEN_BEYOND_MEDIA_CLASSES = (1, 3, 4, 5, 8, 9, 10, 11)  # Lumen, Intima, Lipid, Calcium, Thrombus, Plaque rupture, Layered plaque, Neovascularization
_RADIAL_GUIDEWIRE_CLASS = 2


def _sample_radial_prior_logits(
    model: ConditionalINR,
    feat_vol:    torch.Tensor,   # [B, feat_ch, D, H, W] — NOT detached (encoder gets gradient)
    global_feat: torch.Tensor,   # [B, global_ch] — NOT detached
    volume_shape: tuple,         # (B, C, D, H, W)
    ann_z: torch.Tensor,         # [B] annotated frame z-index per batch item
    cfg: Config,
    device: torch.device,
    layer1: Optional[torch.Tensor] = None,
    layer2: Optional[torch.Tensor] = None,
    layer3: Optional[torch.Tensor] = None,
    layer4: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample points along rays from the frame center; return two raw (unweighted)
    penalty scalars — logged separately so a miscalibrated weight is easy to spot
    (smoothness_2d_weight stayed inert even at 400x its original weight, so this
    checks the raw magnitude explicitly rather than trusting the weighted term).

    media_boundary_loss:     penalises Media-close-to-center paired with any
                              non-excepted class (see _RADIAL_FORBIDDEN_BEYOND_
                              MEDIA_CLASSES) predicted farther out on the same
                              ray — Media is the outer boundary, only Background
                              (or Guidewire/Sidebranch, excepted) should appear
                              beyond it.
    guidewire_truncation_loss: penalises a Guidewire prediction with no Guidewire
                              probability anywhere FARTHER OUT toward the frame
                              edge on the same ray — a guidewire's shadow blocks
                              the signal behind it, so once guidewire starts it
                              should continue uninterrupted to the edge, not
                              revert to background partway (the "stops at half
                              radius" symptom).
    """
    B, _, D, H, W = volume_shape
    M, R = cfg.radial_prior_n_rays, cfg.radial_prior_n_radii
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    max_r  = min(cx, cy) - 1.0

    theta = torch.rand(M, device=device) * (2 * math.pi)          # [M]
    radii = torch.linspace(1.0, max_r, R, device=device)          # [R], near -> far

    xs = (cx + radii.view(1, R) * torch.cos(theta).view(M, 1)).reshape(-1)   # [M*R]
    ys = (cy + radii.view(1, R) * torch.sin(theta).view(M, 1)).reshape(-1)   # [M*R]

    z_b  = ann_z.float().unsqueeze(1).expand(B, M * R)             # [B, M*R]
    coords = torch.stack([xs.unsqueeze(0).expand(B, -1),
                           ys.unsqueeze(0).expand(B, -1),
                           z_b], dim=-1)                            # [B, M*R, 3]

    logits = model.decode_3d(feat_vol, global_feat, coords, (D, H, W),
                              layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
    probs  = F.softmax(logits.view(B, M, R, cfg.num_classes), dim=-1)

    p_media     = probs[..., _RADIAL_MEDIA_CLASS]                                            # [B, M, R]
    p_forbidden = probs[..., list(_RADIAL_FORBIDDEN_BEYOND_MEDIA_CLASSES)].sum(dim=-1)        # [B, M, R]
    p_gwire     = probs[..., _RADIAL_GUIDEWIRE_CLASS]                                         # [B, M, R]

    # Media boundary: media at r_i paired with a forbidden class at r_j, for every i < j on the ray.
    pair     = p_media.unsqueeze(-1) * p_forbidden.unsqueeze(-2)                       # [B, M, R(i), R(j)]
    tri_mask = torch.triu(torch.ones(R, R, device=device, dtype=torch.bool), diagonal=1)
    media_boundary_loss = pair[:, :, tri_mask].mean()

    # Guidewire truncation: penalise p_gwire[i] when no later point on the ray
    # (r_{i+1} .. r_{R-1}, i.e. everything farther toward the edge) has meaningful
    # guidewire probability — a confident guidewire prediction that isn't followed
    # by more guidewire all the way out is the truncation pattern. Computed via a
    # reverse-cummax: flip the ray so "edge" comes first, running-max from there
    # gives "best probability at-or-beyond this point", shift and flip back to get
    # "best probability strictly beyond this point" in original near->far order.
    p_rev            = torch.flip(p_gwire, dims=[-1])                                  # edge -> center
    running_max_rev  = torch.cummax(p_rev, dim=-1)[0]                                  # [B, M, R]
    future_max_rev   = F.pad(running_max_rev[..., :-1], (1, 0), value=0.0)             # shift toward center
    future_max       = torch.flip(future_max_rev, dims=[-1])                           # back to near -> far
    guidewire_truncation_loss = (p_gwire * (1.0 - future_max))[:, :, :-1].mean()       # exclude edge point itself

    return media_boundary_loss, guidewire_truncation_loss


def _save_latest_checkpoint(
    save_dir: str,
    epoch: int,
    model: "ConditionalINR",
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_dice: float,
    history: Dict,
) -> None:
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_dice": best_dice,
        "history":   history,
    }, os.path.join(save_dir, "latest.pt"))


def _load_latest_checkpoint(
    save_dir: str,
    model: "ConditionalINR",
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, float, Optional[Dict]]:
    """Load latest.pt if present. Returns (start_epoch, best_dice, history) or (1, 0.0, None)."""
    path = os.path.join(save_dir, "latest.pt")
    if not os.path.exists(path):
        return 1, 0.0, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    logger.info(
        f"Resumed from latest.pt — epoch {ckpt['epoch']}, "
        f"best val Dice so far {ckpt['best_dice']:.4f}"
    )
    return ckpt["epoch"] + 1, ckpt["best_dice"], ckpt["history"]


def train_one_epoch(
    model: ConditionalINR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Config,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    max_batches: int = 0,
) -> tuple:
    import time
    model.train()
    total_loss = 0.0
    log_accum  = {"ce": 0.0, "dice": 0.0, "mse_onehot": 0.0, "smooth_3d": 0.0, "smooth_2d": 0.0, "feat_sup": 0.0, "tv": 0.0, "dense_dec": 0.0, "radial_media_boundary": 0.0, "radial_guidewire_truncation": 0.0}
    n_batches  = 0

    for volume_or_shape, coords, labels, full_label in loader:
        t0         = time.time()
        coords     = coords.to(device)
        labels     = labels.to(device)
        full_label = full_label.to(device)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            if cfg.unconditional:
                volume_shape  = tuple(volume_or_shape[0].tolist())  # (D, H, W) from [B, 3]
                logits        = model(coords, volume_shape)
                smooth_logits = sp_logits_A = sp_logits_B = dp = None
                radial_media_boundary_loss = radial_guidewire_truncation_loss = None
            elif cfg.training_mode == "3D":
                volume = volume_or_shape.to(device)
                need_multiscale = cfg.use_intermediate_features or \
                    (cfg.use_dense_decoder and cfg.dense_decoder_skip_connections)
                ms_layer3 = ms_layer4 = None
                if need_multiscale and cfg.encoder_depth == 5:
                    # Reached both for use_intermediate_features (INR conditioning, depth5-supported
                    # since 2026-07-17) and dense_decoder_skip_connections.
                    feat_vol, global_feat, ms_layer1, ms_layer2, ms_layer3, ms_layer4 = \
                        model.encoder.forward_deep_multiscale(volume)
                elif need_multiscale:
                    feat_vol, global_feat, ms_layer1, ms_layer2 = model.encoder.forward_multiscale(volume)
                else:
                    feat_vol, global_feat = model.encoder(volume)
                    ms_layer1 = ms_layer2 = None
                if cfg.feature_supervision:
                    logits, aux_logits = model.decode_3d(feat_vol, global_feat, coords, volume.shape[2:],
                                                          return_feat_logits=True, layer1=ms_layer1, layer2=ms_layer2,
                                                          layer3=ms_layer3, layer4=ms_layer4)
                else:
                    logits     = model.decode_3d(feat_vol, global_feat, coords, volume.shape[2:],
                                                  layer1=ms_layer1, layer2=ms_layer2,
                                                  layer3=ms_layer3, layer4=ms_layer4)
                    aux_logits = None
                smooth_logits         = _sample_smoothness_3d_logits(
                    model, feat_vol.detach(), global_feat.detach(),
                    volume.shape, cfg, device,
                    layer1=ms_layer1.detach() if ms_layer1 is not None else None,
                    layer2=ms_layer2.detach() if ms_layer2 is not None else None,
                    layer3=ms_layer3.detach() if ms_layer3 is not None else None,
                    layer4=ms_layer4.detach() if ms_layer4 is not None else None,
                )
                if cfg.smoothness_2d_weight > 0:
                    ann_z = coords[:, 0, 2].long()
                    sp_logits_A, sp_logits_B = _sample_smoothness_2d_logits(
                        model, feat_vol, global_feat, volume.shape, ann_z, cfg, device,
                        layer1=ms_layer1, layer2=ms_layer2, layer3=ms_layer3, layer4=ms_layer4,
                    )
                else:
                    sp_logits_A = sp_logits_B = None
                if cfg.radial_media_prior_weight > 0 or cfg.radial_guidewire_prior_weight > 0:
                    ann_z = coords[:, 0, 2].long()
                    radial_media_boundary_loss, radial_guidewire_truncation_loss = _sample_radial_prior_logits(
                        model, feat_vol, global_feat, volume.shape, ann_z, cfg, device,
                        layer1=ms_layer1, layer2=ms_layer2, layer3=ms_layer3, layer4=ms_layer4,
                    )
                else:
                    radial_media_boundary_loss = radial_guidewire_truncation_loss = None
                dp = None
            else:
                volume = volume_or_shape.to(device)
                if cfg.feature_supervision:
                    logits, dense_pred = model(volume, coords, return_feat_logits=True)
                else:
                    logits = model(volume, coords)
                smooth_logits = None
                sp_logits_A = sp_logits_B = None
                dp = dense_pred if cfg.feature_supervision else None
                radial_media_boundary_loss = radial_guidewire_truncation_loss = None
            loss, log = compute_loss(logits, labels, cfg, smooth_logits, dp, sp_logits_A, sp_logits_B)

            if radial_media_boundary_loss is not None:
                loss = loss + cfg.radial_media_prior_weight * radial_media_boundary_loss \
                             + cfg.radial_guidewire_prior_weight * radial_guidewire_truncation_loss
                log["radial_media_boundary"] = radial_media_boundary_loss.item()
                log["radial_guidewire_truncation"] = radial_guidewire_truncation_loss.item()

            if not cfg.unconditional and cfg.feature_supervision:
                if cfg.training_mode == "3D" and aux_logits is not None:
                    # Point-wise CE on local features at sampled annotated coords
                    feat_loss = F.cross_entropy(
                        aux_logits.reshape(-1, cfg.num_classes),
                        labels.reshape(-1),
                        ignore_index=cfg.ignore_index,
                    )
                elif full_label.numel() > 0:
                    # 2D dense conv head on full GT frame
                    feat_loss = F.cross_entropy(dense_pred, full_label, ignore_index=cfg.ignore_index)
                else:
                    feat_loss = None
                if feat_loss is not None:
                    loss = loss + cfg.feature_supervision_weight * feat_loss
                    log["feat_sup"] = feat_loss.item()

            if cfg.use_dense_decoder and cfg.training_mode == "3D" and not cfg.unconditional:
                if cfg.dense_decoder_skip_connections and cfg.encoder_depth == 5:
                    dense_logits = model.dense_decoder(feat_vol, layer1=ms_layer1, layer2=ms_layer2,
                                                        layer3=ms_layer3, layer4=ms_layer4)  # [B, C, D', H', W']
                elif cfg.dense_decoder_skip_connections:
                    dense_logits = model.dense_decoder(feat_vol, layer1=ms_layer1, layer2=ms_layer2)  # [B, C, D', H', W']
                else:
                    dense_logits = model.dense_decoder(feat_vol)          # [B, C, D', H', W']
                dense_label  = full_label                             # [B, D, H, W], 255 for unannotated
                if dense_label.numel() > 0:
                    if dense_logits.shape[2:] != dense_label.shape[1:]:
                        # DenseDecoder3D (no-skip mode) only upsamples XY (H/8->H), never Z — if the
                        # encoder downsamples Z (encoder_z_strides) or uses a different XY dilation,
                        # dense_logits' spatial shape can end up smaller than dense_label's. Skip mode
                        # restores full Z by construction but this stays as a safety net either way.
                        dense_logits = F.interpolate(
                            dense_logits, size=dense_label.shape[1:],
                            mode="trilinear", align_corners=False,
                        )
                    B_, C_, D_, H_, W_ = dense_logits.shape
                    flat_dl = dense_logits.permute(0, 2, 3, 4, 1).reshape(-1, C_)  # [B*D*H*W, C]
                    flat_ll = dense_label.reshape(-1)                               # [B*D*H*W]
                    dec_ce   = F.cross_entropy(flat_dl, flat_ll, ignore_index=cfg.ignore_index)
                    dec_dice = dice_loss(flat_dl, flat_ll, C_, cfg.ignore_index, cfg.ignore_background)
                    dec_loss = dec_ce + cfg.dice_weight * dec_dice
                    loss     = loss + cfg.dense_decoder_weight * dec_loss
                    log["dense_dec"] = dec_loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in log.items():
            log_accum[k] = log_accum.get(k, 0.0) + v
        n_batches += 1

        if n_batches == 1 and logger is not None:
            batch_time  = time.time() - t0
            n_effective = max_batches if max_batches > 0 else len(loader)
            est_epoch   = batch_time * n_effective
            logger.info(
                f"First batch done in {batch_time:.1f}s — "
                f"est. {est_epoch / 60:.1f} min/epoch "
                f"({n_effective} batches)"
            )

        if max_batches > 0 and n_batches >= max_batches:
            break

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

    for volume_or_shape, coords, labels, _ in loader:
        coords = coords.to(device)
        labels = labels.to(device)

        if cfg.unconditional:
            volume_shape = tuple(volume_or_shape[0].tolist())
            logits = model(coords, volume_shape)
        else:
            volume = volume_or_shape.to(device)
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

    # Unconditional: everything is in RAM — workers only add IPC overhead
    train_workers = 0 if cfg.unconditional else 8
    val_workers   = 0 if cfg.unconditional else 4
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=train_workers, pin_memory=(not cfg.unconditional),
        persistent_workers=(train_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=val_workers, pin_memory=(not cfg.unconditional),
        persistent_workers=(val_workers > 0),
    )

    model  = (UnconditionalINR(cfg) if cfg.unconditional else ConditionalINR(cfg)).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    def _count_params(module):
        return sum(p.numel() for p in module.parameters())

    total_params = _count_params(model)
    inr_params   = _count_params(model.inr)
    if cfg.unconditional:
        logger.info(
            f"Model parameters (unconditional) — "
            f"total: {total_params:,}  |  INR: {inr_params:,}"
        )
    else:
        encoder_params = _count_params(model.encoder)
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

    # Fixed entries used for periodic visual inspection.
    # Unconditional: all entries are identical — use just one.
    vis_entries       = val_ds.entries[:1]   if cfg.unconditional else val_ds.entries[:10]
    train_vis_entries = train_ds.entries[:1] if cfg.unconditional else train_ds.entries[:10]

    history: Dict = {
        "epoch": [], "train_loss": [], "ce_loss": [], "dice_loss": [], "mse_loss": [],
        "smooth_loss": [], "spatial_smooth_loss": [],
        "feat_sup_loss": [], "tv_loss": [], "dense_dec_loss": [],
        "radial_media_loss": [], "radial_gwire_loss": [],
        "val_epoch": [], "val_loss": [], "val_dice_mean": [], "val_dice_per_class": [],
    }
    best_dice = 0.0
    start_epoch, best_dice, loaded_history = _load_latest_checkpoint(
        cfg.checkpoint_dir, model, optimizer, scheduler, scaler, device, logger,
    )
    if loaded_history is not None:
        history = loaded_history
        # Backfill any history keys added after this checkpoint was last saved (e.g. a
        # resumed run whose checkpoint predates the radial-prior loss) so every array
        # stays the same length — a missing key would KeyError on the next epoch's
        # append, and an empty-but-present key would silently misalign the x-axis.
        n_done = len(history.get("epoch", []))
        for key in ("radial_media_loss", "radial_gwire_loss"):
            if key not in history:
                history[key] = [0.0] * n_done

    if start_epoch > cfg.num_epochs:
        logger.info("Run already complete — nothing to do.")
        return model

    max_batches = max(1, int(cfg.max_batches_frac * len(train_ds))) if cfg.max_batches_frac > 0 else 0
    eff_val_interval = max(1, int(cfg.val_interval_pct * cfg.num_epochs)) if cfg.val_interval_pct > 0 else cfg.val_interval
    if max_batches:
        logger.info(f"Max batches/epoch: {max_batches} ({cfg.max_batches_frac * 100:.0f}% of {len(train_ds)})")
    if cfg.val_interval_pct > 0:
        logger.info(f"Val interval: every {eff_val_interval} epochs ({cfg.val_interval_pct * 100:.0f}% of {cfg.num_epochs})")

    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_loss, train_log = train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, logger, max_batches=max_batches)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["ce_loss"].append(train_log.get("ce", 0.0))
        history["dice_loss"].append(train_log.get("dice", 0.0))
        history["mse_loss"].append(train_log.get("mse_onehot", 0.0))
        history["smooth_loss"].append(train_log.get("smooth_3d", 0.0))
        history["spatial_smooth_loss"].append(train_log.get("smooth_2d", 0.0))
        history["feat_sup_loss"].append(train_log.get("feat_sup", 0.0))

        history["tv_loss"].append(train_log.get("tv", 0.0))
        history["dense_dec_loss"].append(train_log.get("dense_dec", 0.0))
        history["radial_media_loss"].append(train_log.get("radial_media_boundary", 0.0))
        history["radial_gwire_loss"].append(train_log.get("radial_guidewire_truncation", 0.0))

        if cfg.loss_type == "mse_onehot":
            log_str = (
                f"Epoch {epoch:03d}/{cfg.num_epochs} | "
                f"Train loss: {train_loss:.4f} "
                f"(MSE={train_log.get('mse_onehot', 0):.4f} "
                f"Smooth3D={train_log.get('smooth_3d', 0):.3f}"
            )
        else:
            log_str = (
                f"Epoch {epoch:03d}/{cfg.num_epochs} | "
                f"Train loss: {train_loss:.4f} "
                f"(CE={train_log.get('ce', 0):.3f} "
                f"Dice={train_log.get('dice', 0):.3f} "
                f"Smooth3D={train_log.get('smooth_3d', 0):.3f}"
            )
        if cfg.smoothness_2d_weight > 0:
            log_str += f" Smooth2D={train_log.get('smooth_2d', 0):.3f}(×{cfg.smoothness_2d_weight})"
        if cfg.feature_supervision:
            log_str += f" FeatSup={train_log.get('feat_sup', 0):.3f}(×{cfg.feature_supervision_weight})"
        if cfg.tv_weight > 0:
            log_str += f" TV={train_log.get('tv', 0):.3f}(×{cfg.tv_weight})"
        if cfg.use_dense_decoder:
            log_str += f" DenseDec={train_log.get('dense_dec', 0):.3f}(×{cfg.dense_decoder_weight})"
        if cfg.radial_media_prior_weight > 0 or cfg.radial_guidewire_prior_weight > 0:
            log_str += (
                f" RadialMedia={train_log.get('radial_media_boundary', 0):.4f}"
                f"(×{cfg.radial_media_prior_weight}) "
                f"RadialGwire={train_log.get('radial_guidewire_truncation', 0):.4f}"
                f"(×{cfg.radial_guidewire_prior_weight})"
            )
        log_str += ")"

        if epoch % eff_val_interval == 0 or epoch == cfg.num_epochs or epoch == 1:
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
                                smoothness_2d_weight=cfg.smoothness_2d_weight,
                                radial_media_prior_weight=cfg.radial_media_prior_weight,
                                radial_guidewire_prior_weight=cfg.radial_guidewire_prior_weight,
                                class_names=_ITKSNAP_LABELS, class_colors=_ITKSNAP_COLORS)

        # Save latest.pt every epoch (not just at val checkpoints) so a preempted/interrupted
        # job never loses more than one epoch of progress on resume — previously this was
        # only saved inside the val_interval block above, so jobs on preemptible queues could
        # lose up to (eff_val_interval - 1) epochs of compute per interruption.
        _save_latest_checkpoint(
            cfg.checkpoint_dir, epoch, model, optimizer, scheduler, scaler,
            best_dice, history,
        )

        if epoch % max(1, cfg.num_epochs // 10) == 0 or epoch == 5:
            _save_vis_predictions(
                vis_entries, model, cfg, device, epoch,
                cfg.checkpoint_dir, label_mapping=label_mapping, split="val",
            )
            _save_vis_predictions(
                train_vis_entries, model, cfg, device, epoch,
                cfg.checkpoint_dir, label_mapping=label_mapping, split="train",
            )
            logger.info(
                f"Saved prediction visualisations → "
                f"vis_val_epoch_{epoch:04d}/  +  vis_train_epoch_{epoch:04d}/"
            )
            model.train()   # restore train mode after eval inside _save_vis_predictions

        logger.info(log_str)

    logger.info(f"Training complete. Best val Dice: {best_dice:.4f}")

    if cfg.unconditional:
        import SimpleITK as sitk
        logger.info("Generating full pullback prediction from best model...")
        best_ckpt = torch.load(
            os.path.join(cfg.checkpoint_dir, cfg.best_ckpt_name),
            map_location=device, weights_only=False,
        )
        model.load_state_dict(best_ckpt["model"])
        volume_shape = val_ds._unc_volume_shape
        pred_vol     = predict_pullback_unconditional(model, volume_shape, cfg, device)
        # pred_vol is (D, H, W) = (z, y, x) — the order SimpleITK expects, no transpose needed
        img       = sitk.GetImageFromArray(pred_vol.astype(np.int16))
        img.SetSpacing([cfg.xy_spacing, cfg.xy_spacing, cfg.z_spacing])  # (x, y, z)
        save_path = os.path.join(cfg.checkpoint_dir, "prediction_full_pullback.nii.gz")
        sitk.WriteImage(img, save_path)
        logger.info(f"Saved → {save_path}")

    return model
