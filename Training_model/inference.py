import numpy as np
import torch

from config import Config
from model import ConditionalINR, UnconditionalINR


@torch.no_grad()
def predict_frame_unconditional(
    model:        UnconditionalINR,
    volume_shape: tuple,            # (D, H, W) of the full volume
    z_frame:      int,              # which frame to predict
    cfg:          Config,
    device:       torch.device,
    chunk_size:   int = 16384,
) -> np.ndarray:                    # [H, W] int64 class predictions
    """Dense prediction of one frame using the unconditional INR."""
    model.eval()
    D, H, W = volume_shape

    yy, xx      = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    all_coords  = np.stack(
        [xx, yy, np.full((H, W), z_frame, dtype=np.float32)], axis=-1
    ).reshape(-1, 3).astype(np.float32)

    predictions = np.zeros(H * W, dtype=np.int64)
    for start in range(0, len(all_coords), chunk_size):
        end    = min(start + chunk_size, len(all_coords))
        coords = torch.from_numpy(all_coords[start:end]).unsqueeze(0).to(device)
        logits = model(coords, volume_shape)
        predictions[start:end] = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    return predictions.reshape(H, W)


@torch.no_grad()
def predict_pullback_unconditional(
    model:        UnconditionalINR,
    volume_shape: tuple,            # (D, H, W)
    cfg:          Config,
    device:       torch.device,
    chunk_size:   int = 16384,
) -> np.ndarray:                    # [D, H, W] int64 class predictions
    """Dense prediction of the full pullback by iterating over all z-frames."""
    D, H, W = volume_shape
    predictions = np.zeros((D, H, W), dtype=np.int64)
    for z in range(D):
        predictions[z] = predict_frame_unconditional(
            model, volume_shape, z, cfg, device, chunk_size
        )
    return predictions


@torch.no_grad()
def predict_pullback(
    model:        ConditionalINR,
    volume:       np.ndarray,        # [D, H, W] float32, z-score normalised
    cfg:          Config,
    device:       torch.device,
    chunk_size:   int   = 16384,     # voxels processed per INR forward pass (tune for VRAM)
    overlap_frac: float = 0.0,       # fraction of patch_z that overlaps between patches
    use_amp:      bool  = False,     # autocast (bfloat16) forward passes — inference only, no
                                      # backward pass, so the SIREN-gradient-overflow risk that
                                      # rules out AMP for training doesn't apply. bfloat16 (not
                                      # float16) specifically: SIREN's omega_0=30 amplifies the
                                      # pre-sine value, and float16's narrow exponent range distorts
                                      # that (visibly at class boundaries, where predictions are
                                      # most sensitive to small numeric nudges) — bfloat16 keeps
                                      # fp32's exponent range so this doesn't happen, at the cost of
                                      # a bit of mantissa precision. Default False: preserves
                                      # existing behaviour for all callers unless explicitly opted in.
) -> np.ndarray:                     # [D, H, W] int64 class predictions
    """Dense segmentation of a full pullback via sliding z-patches.

    overlap_frac controls the stride between consecutive patches:
        0.0  → non-overlapping (stride = pz, original behaviour)
        0.5  → 50 % overlap    (stride = pz // 2)
        1.0  → maximum overlap (stride = 1, one step per frame)

    Logits from overlapping patches are summed before argmax.  Summing is
    equivalent to averaging for argmax purposes, so no division is needed.

    GPU memory is unchanged — one patch at a time.
    CPU memory: D × H × W × num_classes × 4 bytes for logit accumulation
                (~540 MB for a 180-frame 512×512 pullback with 12 classes).
    """
    model.eval()
    D, H, W = volume.shape
    pz       = cfg.patch_z
    C        = cfg.num_classes

    stride = max(1, round(pz * (1.0 - overlap_frac)))

    # Accumulate logit sums; argmax of sum == argmax of average
    logit_sum = np.zeros((D, H, W, C), dtype=np.float32)

    # Pre-build local (x, y, z_local) coordinate grid once — same for every patch
    zz, yy, xx = np.meshgrid(np.arange(pz), np.arange(H), np.arange(W), indexing="ij")
    patch_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

    z_start = 0
    while z_start < D:
        z_end      = min(z_start + pz, D)
        actual_len = z_end - z_start

        patch = volume[z_start:z_end].copy()
        if actual_len < pz:
            pad   = np.broadcast_to(patch[-1:], (pz - actual_len, H, W))
            patch = np.concatenate([patch, pad], axis=0)

        vol_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

        # Autocast covers ONLY the encoder (a standard conv net — safe in reduced precision).
        # The FiLM-SIREN decode loop below runs in full fp32: measured locally that autocast
        # here (even bfloat16, which has fp32's exponent range) drops boundary-pixel agreement
        # to ~93% vs a ~100% interior — omega_0=30 amplifies rounding error right before sin(),
        # and it's a precision (mantissa-bit) problem, not a range/overflow one, so bfloat16's
        # wider exponent range doesn't help. The decode loop is also where nearly all the FLOPs
        # are (millions of point evaluations vs one encoder pass), so excluding it from autocast
        # gives up most of the raw speedup in exchange for not corrupting predictions at class
        # boundaries — correctness wins here.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=use_amp and device.type == "cuda"):
            l1 = l2 = l3 = l4 = None
            if getattr(model, "use_multiscale", False):
                if cfg.encoder_depth == 5:
                    feat_vol, g_feat, l1, l2, l3, l4 = model.encoder.forward_deep_multiscale(vol_t)
                else:
                    feat_vol, g_feat, l1, l2 = model.encoder.forward_multiscale(vol_t)
            else:
                feat_vol, g_feat = model.encoder(vol_t)   # [1, feat_ch, pz, H/8, W/8], [1, global_ch]

        feat_vol = feat_vol.float()
        g_feat   = g_feat.float()
        l1 = l1.float() if l1 is not None else None
        l2 = l2.float() if l2 is not None else None
        l3 = l3.float() if l3 is not None else None
        l4 = l4.float() if l4 is not None else None

        # Accumulate on GPU and transfer once per patch (not once per chunk) — a .cpu()
        # call forces a device sync, so doing it inside this loop (chunk_size default
        # 16384 → ~256 chunks per patch) serializes GPU work behind ~256 stalls per
        # patch instead of letting the chunk loop's kernels queue back-to-back.
        patch_logits_gpu = torch.empty((pz * H * W, C), dtype=torch.float32, device=device)
        for s in range(0, len(patch_coords), chunk_size):
            e      = min(s + chunk_size, len(patch_coords))
            coords = torch.from_numpy(patch_coords[s:e]).unsqueeze(0).to(device)
            logits = model.decode_3d(feat_vol, g_feat, coords, (pz, H, W),
                                      layer1=l1, layer2=l2, layer3=l3, layer4=l4)
            patch_logits_gpu[s:e] = logits.squeeze(0).float()
        patch_logits = patch_logits_gpu.cpu().numpy()   # single sync point per patch

        logit_sum[z_start:z_end] += patch_logits.reshape(pz, H, W, C)[:actual_len]
        z_start += stride

    return logit_sum.argmax(axis=-1).astype(np.int64)

