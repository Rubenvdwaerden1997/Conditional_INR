import numpy as np
import torch

from config import Config
from model import ConditionalINR, ResidualUNet2p5D, ResidualUNet3D, UnconditionalINR


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
    query_hw:     "int | None" = None,   # if set, query the INR on a (query_hw, query_hw) coordinate
                                          # grid instead of the encoder input's own (H, W) — lets the
                                          # encoder see a downsampled volume while predictions come out
                                          # at a finer resolution (e.g. encoder fed 256, queried at
                                          # native 704). None (default) = old behaviour, grid == (H, W).
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
    qH, qW   = (query_hw, query_hw) if query_hw else (H, W)

    stride = max(1, round(pz * (1.0 - overlap_frac)))

    # Accumulate logit sums; argmax of sum == argmax of average
    logit_sum = np.zeros((D, qH, qW, C), dtype=np.float32)

    # Pre-build local (x, y, z_local) coordinate grid once — same for every patch.
    # Indexed on (qH, qW): the query grid, which may be finer than the encoder
    # input's own (H, W) — decode_3d's grid_sample handles that resolution
    # mismatch transparently (see model.py normalize_coords_for_grid_sample).
    zz, yy, xx = np.meshgrid(np.arange(pz), np.arange(qH), np.arange(qW), indexing="ij")
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
        patch_logits_gpu = torch.empty((pz * qH * qW, C), dtype=torch.float32, device=device)
        for s in range(0, len(patch_coords), chunk_size):
            e      = min(s + chunk_size, len(patch_coords))
            coords = torch.from_numpy(patch_coords[s:e]).unsqueeze(0).to(device)
            logits = model.decode_3d(feat_vol, g_feat, coords, (pz, qH, qW),
                                      layer1=l1, layer2=l2, layer3=l3, layer4=l4)
            patch_logits_gpu[s:e] = logits.squeeze(0).float()
        patch_logits = patch_logits_gpu.cpu().numpy()   # single sync point per patch

        logit_sum[z_start:z_end] += patch_logits.reshape(pz, qH, qW, C)[:actual_len]
        z_start += stride

    return logit_sum.argmax(axis=-1).astype(np.int64)


@torch.no_grad()
def predict_pullback_unet(
    model:  ResidualUNet2p5D,
    volume: np.ndarray,        # [D, H, W] float32, z-score normalised
    cfg:    Config,
    device: torch.device,
) -> np.ndarray:                # [D, H, W] int64 class predictions
    """Dense segmentation of a full pullback, one frame at a time.

    Unlike predict_pullback (ConditionalINR), ResidualUNet2p5D.forward() has no
    coordinate query — one forward pass on a patch_z-frame window always returns
    exactly one [C, H, W] map, trained to match whichever frame in the window is
    annotated (dataset.py's z-jitter puts that frame at a random local index
    during training; at validation/inference time it's always centred — see
    dataset.py's `_load_3d_patch`/`__getitem__` eval-mode z_start formula). So,
    to predict every frame z in the pullback: build a patch_z window centred on
    z with the SAME clipping convention dataset.py uses at eval time
    (z_start = clip(z - pz // 2, 0, max(0, D - pz))), run one forward pass, and
    take that single output as frame z's prediction directly — there is no
    local index to extract, the model's one answer IS the centred frame.

    This means D independent forward passes per pullback (no overlap/stride
    choice the way predict_pullback has, since there's nothing to accumulate —
    each frame gets its own window and exactly one answer). Each pass is a
    single cheap 2D conv forward (no point-sampling decode loop), so this is
    not necessarily slower in wall-clock terms than the INR path despite the
    larger pass count.
    """
    model.eval()
    D, H, W = volume.shape
    pz = cfg.patch_z

    predictions = np.zeros((D, H, W), dtype=np.int64)
    for z in range(D):
        z_start = int(np.clip(z - pz // 2, 0, max(0, D - pz)))
        z_end   = min(z_start + pz, D)
        actual_len = z_end - z_start

        patch = volume[z_start:z_end].copy()
        if actual_len < pz:
            pad   = np.broadcast_to(patch[-1:], (pz - actual_len, H, W))
            patch = np.concatenate([patch, pad], axis=0)

        vol_t  = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(vol_t)   # [1, C, H, W]
        predictions[z] = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return predictions


@torch.no_grad()
def predict_pullback_unet3d(
    model:        ResidualUNet3D,
    volume:       np.ndarray,        # [D, H, W] float32, z-score normalised
    cfg:          Config,
    device:       torch.device,
    overlap_frac: float = 0.0,       # fraction of patch_z that overlaps between patches — same
                                      # convention as predict_pullback (ConditionalINR)
) -> np.ndarray:                     # [D, H, W] int64 class predictions
    """Dense segmentation of a full pullback via sliding z-patches.

    Unlike ResidualUNet2p5D (one frame at a time, no overlap concept —
    see predict_pullback_unet above), ResidualUNet3D predicts a full
    [C, patch_z, H, W] volume per forward pass, so overlapping patches can be
    accumulated exactly like predict_pullback (ConditionalINR) does for its
    coordinate-sampled logits: sum overlapping logits, argmax at the end
    (equivalent to averaging for argmax purposes).

    overlap_frac controls the stride between consecutive patches:
        0.0  → non-overlapping (stride = pz, cheapest)
        0.5  → 50 % overlap    (stride = pz // 2)
        1.0  → maximum overlap (stride = 1, one step per frame)
    """
    model.eval()
    D, H, W = volume.shape
    pz = cfg.patch_z
    C  = cfg.num_classes
    stride = max(1, round(pz * (1.0 - overlap_frac)))

    logit_sum = np.zeros((D, H, W, C), dtype=np.float32)

    z_start = 0
    while z_start < D:
        z_end      = min(z_start + pz, D)
        actual_len = z_end - z_start

        patch = volume[z_start:z_end].copy()
        if actual_len < pz:
            pad   = np.broadcast_to(patch[-1:], (pz - actual_len, H, W))
            patch = np.concatenate([patch, pad], axis=0)

        vol_t  = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        logits = model(vol_t)   # [1, C, pz, H, W]
        logits = logits.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()   # [pz, H, W, C]

        logit_sum[z_start:z_end] += logits[:actual_len]
        z_start += stride

    return logit_sum.argmax(axis=-1).astype(np.int64)

