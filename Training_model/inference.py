import numpy as np
import torch

from config import Config
from model import (
    ConditionalINR,
    normalize_coords_for_grid_sample_2d,
    sample_features_2d,
    scale_to_mm_2d,
)


@torch.no_grad()
def predict_pullback(
    model:      ConditionalINR,
    volume:     np.ndarray,        # [D, H, W] float32, z-score normalised
    cfg:        Config,
    device:     torch.device,
    chunk_size: int = 16384,       # voxels processed per INR forward pass (tune for VRAM)
) -> np.ndarray:                   # [D, H, W] int64 class predictions
    """Dense segmentation of a full pullback via non-overlapping z-patches.

    The encoder processes one patch_z-frame patch at a time (matching training),
    then the INR classifies every voxel in that patch.  Patches tile the full
    pullback without overlap; the last patch is edge-padded if needed.
    """
    model.eval()
    D, H, W = volume.shape
    pz       = cfg.patch_z
    predictions = np.zeros((D, H, W), dtype=np.int64)

    # Pre-build the (x, y, z_local) coordinate grid once — same for every patch
    zz, yy, xx = np.meshgrid(
        np.arange(pz), np.arange(H), np.arange(W), indexing="ij"
    )
    patch_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    z_start = 0
    while z_start < D:
        z_end      = min(z_start + pz, D)
        actual_len = z_end - z_start

        # Build padded patch [pz, H, W]
        patch = volume[z_start:z_end].copy()
        if actual_len < pz:
            pad   = np.broadcast_to(patch[-1:], (pz - actual_len, H, W))
            patch = np.concatenate([patch, pad], axis=0)

        vol_t           = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        feat_vol, skips = model.encoder(vol_t)   # [1, feat_ch, pz, H/8, W/8], list

        patch_pred = np.zeros(pz * H * W, dtype=np.int64)
        for s in range(0, len(patch_coords), chunk_size):
            e      = min(s + chunk_size, len(patch_coords))
            coords = torch.from_numpy(patch_coords[s:e]).unsqueeze(0).to(device)
            logits = model.decode_3d(feat_vol, skips, coords, (pz, H, W))
            patch_pred[s:e] = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        predictions[z_start:z_end] = patch_pred.reshape(pz, H, W)[:actual_len]
        z_start += pz

    return predictions


@torch.no_grad()
def predict_frame_2d(
    model:      ConditionalINR,
    frame:      np.ndarray,        # [C, H, W] or [H, W] float32, normalised to [0, 1]
    cfg:        Config,
    device:     torch.device,
    chunk_size: int = 16384,
) -> np.ndarray:                   # [H, W] int64 class predictions
    """Dense 2D segmentation of a single frame, processed in chunks to avoid OOM."""
    model.eval()
    if frame.ndim == 2:
        frame = frame[np.newaxis]   # [1, H, W]
    C, H, W = frame.shape

    frame_tensor = (
        torch.from_numpy(frame).unsqueeze(0).float().to(device)
    )  # [1, C, H, W]
    feat_map, global_feat, skip_maps = model.encoder(frame_tensor)   # encode once, reuse for all chunks
    # global_feat: [1, global_ch] — same for every chunk, broadcast per coordinate batch

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    all_coords = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    # Raw pixel indices — centering is applied per-chunk only for the PE,
    # matching how model.forward() separates feature sampling from PE coords.
    predictions = np.zeros(H * W, dtype=np.int64)

    for start in range(0, len(all_coords), chunk_size):
        end    = min(start + chunk_size, len(all_coords))
        coords = torch.from_numpy(all_coords[start:end]).unsqueeze(0).to(device)

        center      = coords.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0])
        coords_mm   = scale_to_mm_2d(coords - center, cfg)
        pe          = model.pe(coords_mm)
        coords_norm = normalize_coords_for_grid_sample_2d(coords, (H, W))
        feat        = sample_features_2d(feat_map, coords_norm)
        N           = feat.shape[1]
        g_feat_exp  = global_feat.unsqueeze(1).expand(-1, N, -1)
        feat        = torch.cat([feat, g_feat_exp], dim=-1)
        for skip_map in skip_maps:
            feat = torch.cat([feat, sample_features_2d(skip_map, coords_norm)], dim=-1)

        logits = model.inr(torch.cat([pe, feat], dim=-1))   # [1, chunk, C]
        predictions[start:end] = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    return predictions.reshape(H, W)
