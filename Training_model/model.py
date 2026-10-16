"""
Model components for the Conditional INR.

Coordinate convention used throughout:
  coords[..., 0] = x  (width,  W dimension)
  coords[..., 1] = y  (height, H dimension)
  coords[..., 2] = z  (depth,  pullback axis / frame index)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import List, Optional, Tuple

from config import Config


# ---------------------------------------------------------------------------
# Resizing utilities
# ---------------------------------------------------------------------------

def resize_volume(x: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinearly resize the (H, W) dims of a [..., H, W] float tensor.

    Works on a bare [H, W] frame or a [D, H, W] volume/patch — each leading
    slice is resized independently, D is left untouched.
    """
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(0)
    out = T.Resize((size, size), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(x)
    return out.squeeze(0) if squeeze else out


def resize_labels(x: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize the (H, W) dims of a [..., H, W] integer label tensor.

    Nearest-neighbour avoids inventing intermediate class indices at boundaries.
    """
    squeeze = x.dim() == 2
    if squeeze:
        x = x.unsqueeze(0)
    out = T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST)(x)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def normalize_coords_3d(
    coords: torch.Tensor,        # [..., 3]  (x, y, z) voxel indices
    volume_shape: tuple,         # (D, H, W)  — H, W of the grid coords are indexed in;
                                  # need not match the encoder input's own resolution
                                  # (see query_shape in ConditionalINR.forward).
    cfg: Config,
) -> torch.Tensor:               # [..., 3]  physically-proportional normalised coords
    """Normalise voxel coords to physically-proportional space for the PE.

    Centers at the volume midpoint, converts to mm, then divides by the xy
    physical half-extent so that x, y ≈ [-1, 1] and z ≈ [-z_ratio, z_ratio]:

        z_ratio = (D × z_spacing) / (W × xy_spacing)

    Example — 512×512×32 with xy_spacing=0.01 mm, z_spacing=0.10 mm:
        z_ratio = (32 × 0.10) / (512 × 0.01) = 3.2 / 5.12 ≈ 0.625

    cfg.xy_spacing is mm/px at cfg.resize_to resolution (or native_xy_size if
    resize_to=0 — see Config.__post_init__). volume_shape's W need not equal
    that reference resolution (e.g. coords queried at native 704 while the
    encoder was fed a 256 input): xy_spacing is rescaled to W here so the
    physical half-extent (and therefore z_ratio) stays identical regardless
    of which grid W the caller's coords happen to be indexed in. When
    W == ref_w (the only case that existed before this), this is a no-op —
    existing single-resolution experiments are unaffected.
    """
    D, H, W        = volume_shape
    ref_w          = cfg.resize_to if cfg.resize_to else cfg.native_xy_size
    norm_scale     = (ref_w - 1) * cfg.xy_spacing / 2.0   # physical half-extent (mm), resolution-invariant
    xy_spacing_at_w = cfg.xy_spacing * ref_w / W          # mm/px at this call's W
    center     = coords.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0])
    spacing    = coords.new_tensor([xy_spacing_at_w, xy_spacing_at_w, cfg.z_spacing])
    return (coords - center) * spacing / norm_scale


def normalize_coords_for_grid_sample(
    coords: torch.Tensor,
    volume_shape: Tuple[int, int, int],   # (D, H, W)
) -> torch.Tensor:
    """Normalise voxel coords to [-1, 1] per axis for F.grid_sample.

    grid_sample expects its grid in (x, y, z) = (W, H, D) order — this
    function does that flip explicitly to avoid silent sampling errors.

    coords : [B, N, 3]  (x, y, z) voxel order
    returns: [B, N, 3]  grid_sample-compatible (W-norm, H-norm, D-norm)
    """
    D, H, W = volume_shape
    x = 2.0 * coords[..., 0] / (W - 1) - 1.0
    y = 2.0 * coords[..., 1] / (H - 1) - 1.0
    # D == 1 (single-frame 2D input, patch_z=1): (D - 1) would be a divide-by-zero. The feature
    # volume has exactly one z plane, so every query samples it -- pin z to its centre (0).
    z = 2.0 * coords[..., 2] / (D - 1) - 1.0 if D > 1 else torch.zeros_like(coords[..., 2])
    return torch.stack([x, y, z], dim=-1)


def sample_features(
    feat_vol: torch.Tensor,      # [B, C, D, H, W]
    coords_norm: torch.Tensor,   # [B, N, 3]  grid_sample convention
) -> torch.Tensor:               # [B, N, C]
    """Trilinearly sample a feature volume at normalised coordinates."""
    B, N, _ = coords_norm.shape
    grid     = coords_norm.view(B, N, 1, 1, 3)
    sampled  = F.grid_sample(
        feat_vol, grid,
        mode="bilinear",
        padding_mode="border",   # border avoids zero-padding edge artifacts
        align_corners=True,
    )  # [B, C, N, 1, 1]
    return sampled.view(B, feat_vol.shape[1], N).permute(0, 2, 1)  # [B, N, C]


def sample_features_liif(
    feat_vol: torch.Tensor,      # [B, C, Dz, Hy, Wx]
    coords_grid: torch.Tensor,   # [B, N, 3]  grid_sample convention (x→Wx, y→Hy, z→Dz)
) -> Tuple[torch.Tensor, torch.Tensor]:  # ([B, N, C], [B, N, 3])
    """LIIF nearest-cell sampling: constant feature per encoder cell + normalised delta.

    All query points that fall inside the same encoder cell share an identical
    feature vector → identical FiLM (γ, β) → the SIREN is a smooth function of
    coordinates within each cell.  The delta encodes sub-cell position so the
    SIREN can still resolve detail inside a cell.

    delta is normalised to [-0.5, 0.5] per axis (0 = cell centre, ±0.5 = edge).
    """
    B, N, _ = coords_grid.shape
    C, Dz, Hy, Wx = feat_vol.shape[1], feat_vol.shape[2], feat_vol.shape[3], feat_vol.shape[4]

    # Nearest-cell feature (constant within each cell)
    grid    = coords_grid.view(B, N, 1, 1, 3)
    sampled = F.grid_sample(feat_vol, grid, mode="nearest", padding_mode="border", align_corners=True)
    local_feat = sampled.view(B, C, N).permute(0, 2, 1)   # [B, N, C]

    # Compute delta: distance to nearest cell centre, normalised to [-0.5, 0.5]
    gx, gy, gz = coords_grid[..., 0], coords_grid[..., 1], coords_grid[..., 2]

    def cell_delta(g: torch.Tensor, N_feat: int) -> torch.Tensor:
        if N_feat <= 1:
            return torch.zeros_like(g)
        # Cell centre in grid coords for align_corners=True: 2*i/(N-1) - 1
        idx    = ((g + 1.0) * (N_feat - 1) * 0.5).round().clamp(0, N_feat - 1)
        center = 2.0 * idx / (N_feat - 1) - 1.0
        # Normalise by half cell width so delta ∈ [-0.5, 0.5]
        return (g - center) * (N_feat - 1) * 0.5

    delta = torch.stack([cell_delta(gx, Wx), cell_delta(gy, Hy), cell_delta(gz, Dz)], dim=-1)
    return local_feat, delta


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

class RawCoordEncoding(nn.Module):
    """Pass-through encoding for SIREN: returns the 3 normalised coordinates as-is.

    SIREN's sinusoidal activations act as an implicit frequency decomposition,
    so pre-encoding with Fourier features would cause nested sinusoids and
    break the ω₀-based weight initialisation.
    """
    out_dim = 3

    def forward(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """coords_norm: [..., 3]  →  [..., 3]"""
        return coords_norm


class AnisotropicPositionalEncoding(nn.Module):
    """Fourier positional encoding with separate frequency schedules per axis.

    Receives physically-normalised coordinates from normalize_coords_3d:
    x, y ≈ [-1, 1]  and  z ≈ [-z_ratio, z_ratio]  (z_ratio < 1 for IV-OCT).
    xy gets more frequencies (fine in-plane detail) and z fewer (coarse pullback axis).
    """

    def __init__(self, num_freqs_xy: int = 6, num_freqs_z: int = 4):
        super().__init__()
        freqs_xy = 2.0 ** torch.arange(num_freqs_xy).float()
        freqs_z  = 2.0 ** torch.arange(num_freqs_z).float()
        self.register_buffer("freqs_xy", freqs_xy)
        self.register_buffer("freqs_z",  freqs_z)

        self.out_dim = (
            3
            + 2 * num_freqs_xy   # sin + cos for x
            + 2 * num_freqs_xy   # sin + cos for y
            + 2 * num_freqs_z    # sin + cos for z
        )

    def forward(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """coords_norm: [..., 3] physically-normalised  →  [..., out_dim]"""
        x = coords_norm[..., 0:1]
        y = coords_norm[..., 1:2]
        z = coords_norm[..., 2:3]

        def fourier(val, freqs):
            angles = val * freqs
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return torch.cat([
            coords_norm,
            fourier(x, self.freqs_xy),
            fourier(y, self.freqs_xy),
            fourier(z, self.freqs_z),
        ], dim=-1)


# ---------------------------------------------------------------------------
# 3D ResNet building block
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """Anisotropic 3D residual block: (1,3,3) kernels operate only in XY.

    stride_xy=2 halves XY resolution (standard ResNet behaviour).
    dilation_xy>1 widens the receptive field without downsampling — used in
    the last stage when you want dilated convolutions instead of stride.
    The two are mutually exclusive: if dilation_xy>1, stride_xy is forced to 1.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride_xy: int = 1, dilation_xy: int = 1, stride_z: int = 1):
        super().__init__()
        # Dilation and stride are mutually exclusive; dilation overrides stride_z too
        actual_stride_z = 1 if dilation_xy > 1 else stride_z
        pad = dilation_xy                               # same-padding for any dilation rate

        # When striding in Z, widen the kernel to (3,3,3) so Z neighbours are
        # aggregated before downsampling.  Without this, every other Z frame
        # is silently discarded (stride without seeing neighbours = information loss).
        kz = 3 if actual_stride_z > 1 else 1
        pz = 1 if actual_stride_z > 1 else 0

        self.conv1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=(kz, 3, 3),
            stride=(actual_stride_z, stride_xy, stride_xy),
            padding=(pz, pad, pad),
            dilation=(1, dilation_xy, dilation_xy),
            bias=False,
        )
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=(1, 3, 3),
            padding=(0, pad, pad),
            dilation=(1, dilation_xy, dilation_xy),
            bias=False,
        )
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.ReLU(inplace=True)

        if stride_xy != 1 or actual_stride_z != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                          stride=(actual_stride_z, stride_xy, stride_xy), bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out      = self.act(self.norm1(self.conv1(x)))
        out      = self.norm2(self.conv2(out))
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# 3D ResNet encoder
# ---------------------------------------------------------------------------

class ResNetEncoder3D(nn.Module):
    """Anisotropic 3D ResNet encoder for IV-OCT volumes.

    XY is downsampled 8× via stem (2×) + two strided ResBlocks (2× each).
    Z is never pooled — already coarse at 0.10 mm/frame.

    dilation_xy controls the last ResBlock:
      1  → standard stride-2 conv  (bottleneck: H/8, W/8)
      >1 → dilated conv, no stride (bottleneck: H/4, W/4, wider receptive field)

    Returns:
        feat:        [B, feat_ch, Z, H/8, W/8]  — local feature volume
        global_feat: [B, global_ch]              — global context vector
    """

    def __init__(self, in_ch: int = 1, feat_ch: int = 64,
                 global_ch: int = 64, dilation_xy: int = 1,
                 z_strides: Optional[List[int]] = None,
                 depth: int = 3):
        super().__init__()
        if z_strides is None:
            z_strides = [1] * depth
        assert depth in (3, 5, 7), f"ResNetEncoder3D depth must be 3, 5, or 7, got {depth}"
        self.depth = depth
        b = feat_ch // 4                                # base channels (16 for feat_ch=64)

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, b, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.ReLU(inplace=True),
        )                                               # [B, b,   Z, H/2, W/2]

        self.layer1 = ResBlock3D(b,     b * 2, stride_xy=1,                          stride_z=z_strides[0])
        self.layer2 = ResBlock3D(b * 2, b * 4, stride_xy=2,                          stride_z=z_strides[1])
        self.layer3 = ResBlock3D(
            b * 4, b * 4,
            stride_xy=1 if dilation_xy > 1 else 2,     # dilated → no stride
            dilation_xy=dilation_xy,
            stride_z=z_strides[2],
        )
        if depth >= 4:
            # layer4/layer5 always use a standard stride-2 conv (dilation_xy only
            # applies to layer3, matching the existing depth=3 behavior).
            self.layer4 = ResBlock3D(b * 4, b * 4, stride_xy=2, stride_z=z_strides[3])
        if depth >= 5:
            self.layer5 = ResBlock3D(b * 4, b * 4, stride_xy=2, stride_z=z_strides[4])
        if depth >= 6:
            self.layer6 = ResBlock3D(b * 4, b * 4, stride_xy=2, stride_z=z_strides[5])
        if depth >= 7:
            self.layer7 = ResBlock3D(b * 4, b * 4, stride_xy=2, stride_z=z_strides[6])

        self.z_conv = nn.Sequential(
            nn.Conv3d(b * 4, feat_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(min(8, feat_ch), feat_ch),
            nn.ReLU(inplace=True),
        )                                               # [B, feat_ch, Z, H/8, W/8]  (H/32 if depth=5, H/128 if depth=7)

        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, global_ch),
            nn.ReLU(inplace=True),
        )

        self.out_ch    = feat_ch
        self.global_ch = global_ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, 1, Z, H, W]  →  (feat [B, feat_ch, Z, H/8, W/8],  global_feat [B, global_ch])

        H/8 bottleneck for depth=3 (unchanged from the original architecture);
        H/32 for depth=5; H/128 for depth=7 — the depth>=4/5/6/7 branches below
        are no-ops for shallower depths, so depth=3 (and depth=5) behavior/output
        is byte-identical to before this was added.
        """
        x    = self.stem(x)      # [B, b,      Z, H/2, W/2]
        x    = self.layer1(x)    # [B, 2b,     Z, H/2, W/2]
        x    = self.layer2(x)    # [B, 4b,     Z, H/4, W/4]
        x    = self.layer3(x)    # [B, 4b,     Z, H/8, W/8]  (or H/4 if dilated)
        if self.depth >= 4:
            x = self.layer4(x)   # [B, 4b,     Z, H/16, W/16]
        if self.depth >= 5:
            x = self.layer5(x)   # [B, 4b,     Z, H/32, W/32]
        if self.depth >= 6:
            x = self.layer6(x)   # [B, 4b,     Z, H/64, W/64]
        if self.depth >= 7:
            x = self.layer7(x)   # [B, 4b,     Z, H/128, W/128]
        feat = self.z_conv(x)    # [B, feat_ch, Z, H/8 (or H/32, H/128), W/8 (or H/32, H/128)]
        return feat, self.global_proj(feat)

    def forward_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward(), but also returns layer1/layer2 intermediate feature
        maps for multi-scale conditioning (cfg.use_intermediate_features).

        [B, 1, Z, H, W]  →  (feat [B, feat_ch, Z, H/8, W/8],
                              global_feat [B, global_ch],
                              layer1 [B, feat_ch/2, Z, H/2, W/2],
                              layer2 [B, feat_ch,   Z', H/4, W/4])
        """
        x    = self.stem(x)      # [B, b,      Z, H/2, W/2]
        l1   = self.layer1(x)    # [B, 2b,     Z, H/2, W/2]
        l2   = self.layer2(l1)   # [B, 4b,     Z, H/4, W/4]
        x    = self.layer3(l2)   # [B, 4b,     Z, H/8, W/8]  (or H/4 if dilated)
        feat = self.z_conv(x)    # [B, feat_ch, Z, H/8, W/8]
        return feat, self.global_proj(feat), l1, l2

    def forward_deep_multiscale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward_multiscale(), but for depth=5: also returns layer3/layer4.

        Only valid when self.depth == 5 (layer4/layer5 must exist).

        [B, 1, Z, H, W]  →  (feat [B, feat_ch, Z, H/32, W/32],
                              global_feat [B, global_ch],
                              layer1 [B, feat_ch/2, Z,  H/2,  W/2],
                              layer2 [B, feat_ch,   Z,  H/4,  W/4],
                              layer3 [B, feat_ch,   Z', H/8,  W/8],
                              layer4 [B, feat_ch,   Z', H/16, W/16])
        """
        assert self.depth == 5, f"forward_deep_multiscale requires depth=5, got {self.depth}"
        x    = self.stem(x)      # [B, b,      Z,  H/2,  W/2]
        l1   = self.layer1(x)    # [B, 2b,     Z,  H/2,  W/2]
        l2   = self.layer2(l1)   # [B, 4b,     Z,  H/4,  W/4]
        l3   = self.layer3(l2)   # [B, 4b,     Z', H/8,  W/8]
        l4   = self.layer4(l3)   # [B, 4b,     Z', H/16, W/16]
        x    = self.layer5(l4)   # [B, 4b,     Z', H/32, W/32]
        feat = self.z_conv(x)    # [B, feat_ch, Z', H/32, W/32]
        return feat, self.global_proj(feat), l1, l2, l3, l4


# ---------------------------------------------------------------------------
# 2.5D ("pseudo-3D") ResNet encoder — context frames as input channels
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Standard 2D residual block — ResBlock3D (above) with the Z axis removed entirely."""

    def __init__(self, in_ch: int, out_ch: int, stride_xy: int = 1, dilation_xy: int = 1):
        super().__init__()
        pad = dilation_xy

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride_xy,
                               padding=pad, dilation=dilation_xy, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               padding=pad, dilation=dilation_xy, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.ReLU(inplace=True)

        if stride_xy != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride_xy, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out      = self.act(self.norm1(self.conv1(x)))
        out      = self.norm2(self.conv2(out))
        return self.act(out + identity)


class ResNetEncoder2p5D(nn.Module):
    """2.5D ("pseudo-3D") encoder: the whole patch_z window is stacked as input
    channels to a 2D ResNet, instead of running true 3D convolutions over it
    (ResNetEncoder3D, above). Classic 2.5D segmentation — context frames feed
    the central/annotated frame's prediction as extra channels, not as a
    separate spatial axis.

    Because the Z axis is collapsed into channels before any convolution runs,
    the returned feature volume has no z resolution (D'=1): every z-coordinate
    queried downstream (decode_3d's grid_sample) samples the same single
    feature plane. Z-awareness for the INR then comes entirely from the
    existing z positional encoding (num_freqs_z) and the pooled global
    feature, not from a per-z-slice local feature — a deliberate design choice
    (confirmed with the user), not an oversight.

    Returns:
        feat:        [B, feat_ch, 1, H/8, W/8]  — local feature "volume" (D'=1)
        global_feat: [B, global_ch]              — global context vector
    """

    def __init__(self, in_ch: int, feat_ch: int = 64,
                global_ch: int = 64, dilation_xy: int = 1,
                depth: int = 3):
        super().__init__()
        assert depth in (3, 5, 7), f"ResNetEncoder2p5D depth must be 3, 5, or 7, got {depth}"
        self.depth = depth
        b = feat_ch // 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, b, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.ReLU(inplace=True),
        )                                               # [B, b, H/2, W/2]

        self.layer1 = ResBlock2D(b,     b * 2, stride_xy=1)
        self.layer2 = ResBlock2D(b * 2, b * 4, stride_xy=2)
        self.layer3 = ResBlock2D(
            b * 4, b * 4,
            stride_xy=1 if dilation_xy > 1 else 2,
            dilation_xy=dilation_xy,
        )
        if depth >= 4:
            self.layer4 = ResBlock2D(b * 4, b * 4, stride_xy=2)
        if depth >= 5:
            self.layer5 = ResBlock2D(b * 4, b * 4, stride_xy=2)
        if depth >= 6:
            self.layer6 = ResBlock2D(b * 4, b * 4, stride_xy=2)
        if depth >= 7:
            self.layer7 = ResBlock2D(b * 4, b * 4, stride_xy=2)

        self.out_conv = nn.Sequential(
            nn.Conv2d(b * 4, feat_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, feat_ch), feat_ch),
            nn.ReLU(inplace=True),
        )                                               # [B, feat_ch, H/8, W/8]

        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, global_ch),
            nn.ReLU(inplace=True),
        )

        self.out_ch    = feat_ch
        self.global_ch = global_ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, 1, D, H, W]  →  (feat [B, feat_ch, 1, H/8, W/8],  global_feat [B, global_ch])

        D is squeezed out of the channel-1 axis and treated as input channels
        (in_ch must equal D, i.e. cfg.patch_z) before any convolution runs.
        """
        x = x.squeeze(1)         # [B, 1, D, H, W] -> [B, D, H, W]
        x = self.stem(x)         # [B, b,  H/2, W/2]
        x = self.layer1(x)       # [B, 2b, H/2, W/2]
        x = self.layer2(x)       # [B, 4b, H/4, W/4]
        x = self.layer3(x)       # [B, 4b, H/8, W/8]  (or H/4 if dilated)
        if self.depth >= 4:
            x = self.layer4(x)   # [B, 4b, H/16, W/16]
        if self.depth >= 5:
            x = self.layer5(x)   # [B, 4b, H/32, W/32]
        if self.depth >= 6:
            x = self.layer6(x)   # [B, 4b, H/64, W/64]
        if self.depth >= 7:
            x = self.layer7(x)   # [B, 4b, H/128, W/128]
        feat2d = self.out_conv(x)        # [B, feat_ch, H/8 (or H/32, H/128), W/8 (or H/32, H/128)]
        feat   = feat2d.unsqueeze(2)     # [B, feat_ch, 1, H', W'] — D'=1, grid_sample-compatible
        return feat, self.global_proj(feat2d)

    def forward_deep_multiscale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward(), but also returns layer1..layer4 for skip connections
        (Dense2DDecoder(use_skip_connections=True)). Only valid for depth=5 —
        mirrors ResNetEncoder3D.forward_deep_multiscale, 2D.

        [B, 1, D, H, W]  →  (feat [B, feat_ch, 1, H/32, W/32],
                              global_feat [B, global_ch],
                              layer1 [B, 2b, H/2,  W/2],
                              layer2 [B, 4b, H/4,  W/4],
                              layer3 [B, 4b, H/8,  W/8],
                              layer4 [B, 4b, H/16, W/16])
        """
        assert self.depth == 5, f"forward_deep_multiscale requires depth=5, got {self.depth}"
        x    = x.squeeze(1)      # [B, D, H, W]
        x    = self.stem(x)      # [B, b,  H/2,  W/2]
        l1   = self.layer1(x)    # [B, 2b, H/2,  W/2]
        l2   = self.layer2(l1)   # [B, 4b, H/4,  W/4]
        l3   = self.layer3(l2)   # [B, 4b, H/8,  W/8]
        l4   = self.layer4(l3)   # [B, 4b, H/16, W/16]
        x    = self.layer5(l4)   # [B, 4b, H/32, W/32]
        feat2d = self.out_conv(x)
        feat   = feat2d.unsqueeze(2)
        return feat, self.global_proj(feat2d), l1, l2, l3, l4


# ---------------------------------------------------------------------------
# Symmetric dense decoder (diagnostic — no skip connections)
# ---------------------------------------------------------------------------

class DecBlock3D(nn.Module):
    """Upsample then apply a conv — symmetric to a stride-2 ResBlock3D.

    scale_factor defaults to (1,2,2) — XY only, matching the original
    diagnostic decoder. Pass (2,2,2) to also upsample Z (used where a skip
    connection's Z resolution is finer than the current tensor's).
    """

    def __init__(self, in_ch: int, out_ch: int, scale_factor: Tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        kernel  = (3, 3, 3) if scale_factor[0] > 1 else (1, 3, 3)
        padding = (1, 1, 1) if scale_factor[0] > 1 else (0, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False)
        return self.conv(x)


class DenseDecoder3D(nn.Module):
    """Symmetric upsampling decoder for ResNetEncoder3D.

    use_skip_connections=False (default, original diagnostic design): NO skip
    connections — mirrors the encoder channel progression in reverse and
    recovers full (H, W) via three XY bilinear upsamplings (H/8 → H/4 → H/2
    → H). forward(feat_vol) only. Z is never upsampled here (callers must
    handle any Z mismatch, e.g. via F.interpolate to the label shape). Only
    encoder_depth=3 is implemented for this no-skip path.

    use_skip_connections=True, encoder_depth=3 (default): real UNet-style skip
    connections — concatenates the matching-resolution encoder feature
    (layer2 at H/4, layer1 at H/2) at each upsampling stage before a fusion
    conv. Z is upsampled together with XY at the stage where layer1's full-Z
    resolution becomes available. forward(feat_vol, layer1, layer2).

    use_skip_connections=True, encoder_depth=5: two extra outer stages fuse
    layer4 (H/16) and layer3 (H/8) first — both at the bottleneck's Z
    resolution under the depth=5 centered z_strides convention ([1,1,2,1,1]),
    so no Z upsample needed yet — then the H/8→H/4 stage upsamples Z together
    with XY (Z catches up to layer2's full resolution here, one stage earlier
    than in the depth=3 case, because Z downsamples at layer3 not layer2).
    forward(feat_vol, layer1, layer2, layer3, layer4) — all four required.
    """

    def __init__(self, feat_ch: int, num_classes: int, use_skip_connections: bool = False,
                 encoder_depth: int = 3):
        super().__init__()
        assert encoder_depth in (3, 5), f"encoder_depth must be 3 or 5, got {encoder_depth}"
        assert encoder_depth == 3 or use_skip_connections, \
            "DenseDecoder3D(encoder_depth=5) requires use_skip_connections=True (no-skip depth=5 not implemented)"
        b = feat_ch // 4
        self.use_skip      = use_skip_connections
        self.encoder_depth = encoder_depth

        # Mirror z_conv: feat_ch → 4b, z-axis kernel only
        self.z_deconv = nn.Sequential(
            nn.Conv3d(feat_ch, 4 * b, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(min(8, 4 * b), 4 * b),
            nn.ReLU(inplace=True),
        )

        if encoder_depth == 5:
            # Two extra outer stages consuming layer4 (H/16) and layer3 (H/8).
            # Both share the bottleneck's Z resolution (Z downsamples once, at
            # layer3, under the depth=5 centered z_strides convention) — no Z
            # upsample needed yet, channels stay at 4b through both.
            self.up_l4 = DecBlock3D(4 * b, 4 * b)   # H/32 → H/16
            self.fuse_l4 = nn.Sequential(
                nn.Conv3d(4 * b + 4 * b, 4 * b, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.GroupNorm(min(8, 4 * b), 4 * b), nn.ReLU(inplace=True),
            )
            self.up_l3 = DecBlock3D(4 * b, 4 * b)   # H/16 → H/8
            self.fuse_l3 = nn.Sequential(
                nn.Conv3d(4 * b + 4 * b, 4 * b, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.GroupNorm(min(8, 4 * b), 4 * b), nn.ReLU(inplace=True),
            )

        # up3 mirrors "layer3→layer2" transition. depth=5: Z gets restored here
        # (layer2 has full Z under the centered z_strides). depth=3 (either
        # skip or no-skip): Z stays unchanged here, matching original behavior.
        up3_scale = (2, 2, 2) if (use_skip_connections and encoder_depth == 5) else (1, 2, 2)
        self.up3 = DecBlock3D(4 * b, 4 * b, scale_factor=up3_scale)   # H/8 → H/4   mirrors layer3

        if use_skip_connections:
            # layer2 is [4b, H/4] — resolution now matches up3's output (Z included), concat directly.
            self.fuse1 = nn.Sequential(
                nn.Conv3d(4 * b + 4 * b, 2 * b, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.GroupNorm(min(8, 2 * b), 2 * b),
                nn.ReLU(inplace=True),
            )
            # layer1 is [2b, full Z, H/2]. depth=3: Z hasn't been upsampled yet — do it here (as before).
            # depth=5: Z was already restored at the up3 stage above — XY only here.
            up2_scale = (1, 2, 2) if encoder_depth == 5 else (2, 2, 2)
            self.up2 = DecBlock3D(2 * b, 2 * b, scale_factor=up2_scale)   # H/4→H/2  mirrors layer2
            self.fuse2 = nn.Sequential(
                nn.Conv3d(2 * b + 2 * b, b, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.GroupNorm(min(8, b), b),
                nn.ReLU(inplace=True),
            )
        else:
            self.up2 = DecBlock3D(4 * b, 2 * b)   # H/4 → H/2   mirrors layer2
            self.mid = nn.Sequential(              # no upsample   mirrors layer1 (b→2b reversed)
                nn.Conv3d(2 * b, b, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.GroupNorm(min(8, b), b),
                nn.ReLU(inplace=True),
            )

        self.up1  = DecBlock3D(b, b)           # H/2  → H     mirrors stem
        self.head = nn.Conv3d(b, num_classes, kernel_size=1)

    def forward(
        self,
        feat_vol: torch.Tensor,
        layer1: Optional[torch.Tensor] = None,   # [B, feat_ch/2, Z, H/2, W/2] — required if use_skip_connections
        layer2: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z or Z/2, H/4, W/4]
        layer3: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z/2, H/8, W/8]  — required if encoder_depth=5
        layer4: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z/2, H/16, W/16] — required if encoder_depth=5
    ) -> torch.Tensor:
        """[B, feat_ch, Z', H/8 (or H/32), W/8 (or W/32)]  →  [B, num_classes, D, H, W]

        D=Z (full) when use_skip_connections=True, D=Z/2 otherwise — callers
        must handle any remaining shape mismatch against the label (see
        trainer.py / predict_singlepullback_conditional.py call sites).
        """
        x = self.z_deconv(feat_vol)
        if self.encoder_depth == 5:
            assert layer3 is not None and layer4 is not None, \
                "DenseDecoder3D(encoder_depth=5) requires layer3 and layer4"
            x = self.up_l4(x)
            x = self.fuse_l4(torch.cat([x, layer4], dim=1))
            x = self.up_l3(x)
            x = self.fuse_l3(torch.cat([x, layer3], dim=1))
        x = self.up3(x)
        if self.use_skip:
            assert layer1 is not None and layer2 is not None, \
                "DenseDecoder3D(use_skip_connections=True) requires layer1 and layer2"
            x = self.fuse1(torch.cat([x, layer2], dim=1))
            x = self.up2(x)
            x = self.fuse2(torch.cat([x, layer1], dim=1))
        else:
            x = self.up2(x)
            x = self.mid(x)
        x = self.up1(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# 2D dense decoder for ResNetEncoder2p5D (2.5D Residual U-Net baseline)
# ---------------------------------------------------------------------------

class DecBlock2D(nn.Module):
    """DecBlock3D with the Z axis removed — upsample (bilinear) then conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class Dense2DDecoder(nn.Module):
    """Symmetric upsampling decoder for ResNetEncoder2p5D.

    use_skip_connections=False (default): no skip connections — mirrors
    DenseDecoder3D's no-skip path, but (unlike DenseDecoder3D, which only
    implements that path for depth=3) generalizes across depth in {3, 5, 7}
    with `depth - 3` extra stages, exactly matching ResNetEncoder2p5D's own
    `if depth >= 4/5/6/7` pattern.

    use_skip_connections=True: real U-Net-style skip connections, depth=5
    only (mirrors DenseDecoder3D's depth=5 skip path, 2D) — concatenates
    layer4 (H/16), layer3 (H/8), layer2 (H/4), layer1 (H/2) from
    ResNetEncoder2p5D.forward_deep_multiscale at each matching-resolution
    upsampling stage before a fusion conv. forward(feat2d, layer1, layer2,
    layer3, layer4) — all four required.
    """

    def __init__(self, feat_ch: int, num_classes: int, depth: int = 3,
                 use_skip_connections: bool = False):
        super().__init__()
        assert depth in (3, 5, 7), f"Dense2DDecoder depth must be 3, 5, or 7, got {depth}"
        assert not use_skip_connections or depth == 5, \
            "Dense2DDecoder(use_skip_connections=True) requires depth=5 (no other depth implemented)"
        b = feat_ch // 4
        self.use_skip = use_skip_connections
        self.depth    = depth

        if use_skip_connections:
            # Each stage upsamples at a CONSTANT channel width, concatenates the
            # same-resolution skip, then the fuse conv reduces channels for the
            # next stage — mirrors DenseDecoder3D's actual skip pattern (channel
            # reduction happens in the fuse conv, not in the upsample itself).
            # layer1: 2b, layer2/3/4: 4b (same widths ResNetEncoder2p5D produces).
            self.up_l4 = DecBlock2D(4 * b, 4 * b)   # H/32 -> H/16, stays 4b
            self.fuse_l4 = nn.Sequential(
                nn.Conv2d(4 * b + 4 * b, 4 * b, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, 4 * b), 4 * b), nn.ReLU(inplace=True),
            )
            self.up3 = DecBlock2D(4 * b, 4 * b)     # H/16 -> H/8, stays 4b
            self.fuse_l3 = nn.Sequential(
                nn.Conv2d(4 * b + 4 * b, 4 * b, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, 4 * b), 4 * b), nn.ReLU(inplace=True),
            )
            self.up2 = DecBlock2D(4 * b, 4 * b)     # H/8 -> H/4, stays 4b
            self.fuse_l2 = nn.Sequential(
                nn.Conv2d(4 * b + 4 * b, 2 * b, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, 2 * b), 2 * b), nn.ReLU(inplace=True),
            )
            self.up1 = DecBlock2D(2 * b, 2 * b)     # H/4 -> H/2, stays 2b
            self.fuse_l1 = nn.Sequential(
                nn.Conv2d(2 * b + 2 * b, b, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, b), b), nn.ReLU(inplace=True),
            )
            self.up_stem = DecBlock2D(b, b)         # H/2 -> H, stays b
            self.head = nn.Conv2d(b, num_classes, kernel_size=1)
        else:
            # Extra stages mirroring the encoder's layer4..layer7 (each 4b -> 4b, no skip).
            self.extra = nn.ModuleList([DecBlock2D(4 * b, 4 * b) for _ in range(depth - 3)])

            self.up3 = DecBlock2D(4 * b, 4 * b)   # mirrors layer3
            self.up2 = DecBlock2D(4 * b, 2 * b)   # mirrors layer2
            self.mid = nn.Sequential(              # no upsample, mirrors layer1 (b->2b reversed)
                nn.Conv2d(2 * b, b, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(min(8, b), b),
                nn.ReLU(inplace=True),
            )
            self.up1  = DecBlock2D(b, b)           # mirrors the stem
            self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(
        self,
        feat2d: torch.Tensor,
        layer1: Optional[torch.Tensor] = None,   # [B, 2b, H/2,  W/2]  — required if use_skip_connections
        layer2: Optional[torch.Tensor] = None,   # [B, 4b, H/4,  W/4]
        layer3: Optional[torch.Tensor] = None,   # [B, 4b, H/8,  W/8]
        layer4: Optional[torch.Tensor] = None,   # [B, 4b, H/16, W/16]
    ) -> torch.Tensor:
        """[B, feat_ch, H/8 (or H/32, H/128), W/...]  ->  [B, num_classes, H, W]"""
        if self.use_skip:
            assert layer1 is not None and layer2 is not None and layer3 is not None and layer4 is not None, \
                "Dense2DDecoder(use_skip_connections=True) requires layer1..layer4"
            x = self.up_l4(feat2d)
            x = self.fuse_l4(torch.cat([x, layer4], dim=1))
            x = self.up3(x)
            x = self.fuse_l3(torch.cat([x, layer3], dim=1))
            x = self.up2(x)
            x = self.fuse_l2(torch.cat([x, layer2], dim=1))
            x = self.up1(x)
            x = self.fuse_l1(torch.cat([x, layer1], dim=1))
            x = self.up_stem(x)
            return self.head(x)

        x = feat2d
        for stage in self.extra:
            x = stage(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.mid(x)
        x = self.up1(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# INR MLP
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """MLP with a NeRF-style skip connection at the midpoint for stability.

    activation="relu"  — standard ReLU (default)
    activation="siren" — sinusoidal activation (Sitzmann et al. 2020) with
                         omega_0-scaled weight init; better at representing
                         smooth continuous functions like segmentation boundaries.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 4,
                 activation: str = "relu", omega_0: float = 30.0, extra_dim: int = 0):
        super().__init__()
        assert depth >= 2, "Need at least 2 layers for the skip connection"
        assert activation in ("relu", "siren"), f"Unknown activation: {activation}"

        mid = depth // 2
        self.activation       = activation
        self.omega_0          = omega_0
        self.extra_dim        = extra_dim   # features concatenated ONLY at the midpoint (skip) layer
        self.input_layer      = nn.Linear(in_dim, hidden)
        self.layers_pre_skip  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(mid - 1)])
        self.skip_layer       = nn.Linear(in_dim + extra_dim + hidden, hidden)
        self.layers_post_skip = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - mid - 1)])
        self.output_layer     = nn.Linear(hidden, out_dim)

        if activation == "siren":
            self._init_siren(omega_0)
        else:
            self._relu = nn.ReLU(inplace=True)

    def _init_siren(self, omega_0: float) -> None:
        # First layer: standard uniform — omega_0 in forward handles the spread
        nn.init.uniform_(self.input_layer.weight,
                         -1.0 / self.input_layer.in_features,
                          1.0 / self.input_layer.in_features)
        # Hidden layers: scaled so sin argument stays in the expressive [-π, π] range
        for layer in [*self.layers_pre_skip, self.skip_layer, *self.layers_post_skip]:
            nn.init.uniform_(layer.weight,
                             -math.sqrt(6.0 / layer.in_features) / omega_0,
                              math.sqrt(6.0 / layer.in_features) / omega_0)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "siren":
            return torch.sin(self.omega_0 * x)
        return self._relu(x)

    def forward(self, x: torch.Tensor, extra: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[..., in_dim] (+ optional [..., extra_dim] fed only to the midpoint layer)  →  [..., out_dim]"""
        h = self._act(self.input_layer(x))
        for layer in self.layers_pre_skip:
            h = self._act(layer(h))
        if self.extra_dim > 0:
            assert extra is not None, "INR built with extra_dim>0 needs `extra` at forward"
            h = self._act(self.skip_layer(torch.cat([x, extra, h], dim=-1)))
        else:
            h = self._act(self.skip_layer(torch.cat([x, h], dim=-1)))
        for layer in self.layers_post_skip:
            h = self._act(layer(h))
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# FiLM-conditioned SIREN MLP
# ---------------------------------------------------------------------------

class FiLMINR(nn.Module):
    """SIREN MLP conditioned on encoder features via FiLM modulation.

    FiLM is applied to the PRE-ACTIVATION (before sine), not after:
        a  = W · h_prev + b
        a' = (1 + γ(cond)) · a + β(cond)    ← FiLM modulates pre-activation
        h  = sin(ω₀ · a')

    This is correct because SIREN is frequency-based: γ controls the amplitude
    of the pre-activation (changing the frequency response) and β shifts the
    phase (shifting the spatial pattern). Applying FiLM after the sine would
    only scale/shift the output and lose these frequency-domain properties.

    Zero-init of FiLM weights → γ=0, β=0 at start → (1+0)·a+0 = a → pure
    SIREN at initialisation, conditioning learned on top.
    """

    def __init__(self, coord_dim: int, cond_dim: int, hidden: int, out_dim: int,
                 depth: int = 4, omega_0: float = 30.0):
        super().__init__()
        assert depth >= 2
        self.omega_0 = omega_0
        mid = depth // 2

        # Coordinate-only sinusoidal layers
        self.input_layer      = nn.Linear(coord_dim, hidden)
        self.layers_pre_skip  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(mid - 1)])
        self.skip_layer       = nn.Linear(coord_dim + hidden, hidden)   # skip re-injects raw coords
        self.layers_post_skip = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - mid - 1)])
        self.output_layer     = nn.Linear(hidden, out_dim)

        # FiLM layers: one (γ, β) pair per sinusoidal layer (all except output)
        n_hidden = depth
        self.film = nn.ModuleList([nn.Linear(cond_dim, 2 * hidden) for _ in range(n_hidden)])

        self._init_weights(omega_0)

    def _init_weights(self, omega_0: float) -> None:
        nn.init.uniform_(self.input_layer.weight,
                         -1.0 / self.input_layer.in_features,
                          1.0 / self.input_layer.in_features)
        for layer in [*self.layers_pre_skip, self.skip_layer, *self.layers_post_skip]:
            nn.init.uniform_(layer.weight,
                             -math.sqrt(6.0 / layer.in_features) / omega_0,
                              math.sqrt(6.0 / layer.in_features) / omega_0)
        # Small-nonzero init for FiLM weights so the encoder gets gradient from step 1.
        # Zero-init would make ∂loss/∂cond = 0 at init (weight=0 → no gradient path to encoder).
        # std=0.01 keeps γ,β small enough that the SIREN starts near-pure, but the encoder
        # gradient is non-zero and the conditioning can be learned immediately.
        for film_layer in self.film:
            nn.init.normal_(film_layer.weight, std=0.01)
            nn.init.zeros_(film_layer.bias)

    def _film(self, a: torch.Tensor, cond: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply FiLM to pre-activation a: returns (1 + γ) · a + β."""
        params      = self.film[idx](cond)           # [B, N, 2*hidden]
        gamma, beta = params.chunk(2, dim=-1)
        return a * (1.0 + gamma) + beta

    def forward(self, coords: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """coords: [B, N, coord_dim],  cond: [B, N, cond_dim]  →  [B, N, out_dim]"""
        film_idx = 0

        a = self._film(self.input_layer(coords), cond, film_idx);  film_idx += 1
        h = torch.sin(self.omega_0 * a)

        for layer in self.layers_pre_skip:
            a = self._film(layer(h), cond, film_idx);  film_idx += 1
            h = torch.sin(self.omega_0 * a)

        a = self._film(self.skip_layer(torch.cat([coords, h], dim=-1)), cond, film_idx);  film_idx += 1
        h = torch.sin(self.omega_0 * a)

        for layer in self.layers_post_skip:
            a = self._film(layer(h), cond, film_idx);  film_idx += 1
            h = torch.sin(self.omega_0 * a)

        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class UnconditionalINR(nn.Module):
    """PE(x, y, z) → class logits — no encoder, no image features.

    Overfits a single 3D volume. The MLP learns a continuous function from
    physical coordinate space to class probabilities for that one volume.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pe  = (
            RawCoordEncoding()
            if cfg.inr_activation == "siren"
            else AnisotropicPositionalEncoding(
                num_freqs_xy=cfg.num_freqs_xy,
                num_freqs_z=cfg.num_freqs_z,
            )
        )
        self.inr = INR(
            in_dim=self.pe.out_dim,
            hidden=cfg.inr_hidden,
            out_dim=cfg.num_classes,
            depth=cfg.inr_depth,
            activation=cfg.inr_activation,
            omega_0=cfg.siren_omega_0,
        )

    def forward(
        self,
        coords: torch.Tensor,   # [B, N, 3]  voxel space (x, y, z)
        volume_shape: tuple,    # (D, H, W)
    ) -> torch.Tensor:          # [B, N, num_classes]
        coords_norm = normalize_coords_3d(coords, volume_shape, self.cfg)
        return self.inr(self.pe(coords_norm))


class ConditionalINR(nn.Module):
    """3D anisotropic encoder + positional encoding + INR MLP.

    volume: [B, 1, D, H, W]  (z-score normalised patch)
    coords: [B, N, 3]        (x, y, z) voxel indices within the patch
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg      = cfg
        self.use_film = cfg.inr_activation == "siren"
        self.use_multiscale = cfg.use_intermediate_features
        if self.use_multiscale:
            assert cfg.use_local_features, "use_intermediate_features requires use_local_features=True"
        cond_dim      = (cfg.encoder_feat if cfg.use_local_features else 0) + cfg.global_feat_ch
        if cfg.use_local_features and cfg.feature_sampling == "liif":
            cond_dim += 3   # normalised delta offset (x, y, z) to nearest cell centre
        ms_dim = 0
        # Levels actually used: cfg.multiscale_levels clamped to what the depth provides
        # (1-2 at depth 3, 1-4 at depth 5).
        self.ms_levels = sorted(l for l in cfg.multiscale_levels
                                if l <= (4 if cfg.encoder_depth == 5 else 2))
        if self.use_multiscale:
            assert self.ms_levels, \
                f"multiscale_levels {cfg.multiscale_levels} leaves nothing at encoder_depth={cfg.encoder_depth}"
            # layer1 (feat_ch/2 channels, H/2); layer2 (H/4), layer3 (H/8), layer4 (H/16) all feat_ch
            # (layer3-5 preserve channel count, unlike layer1->layer2), each trilinearly sampled at
            # the same query coords as the bottleneck.
            ms_dim = sum(cfg.encoder_feat // 2 if l == 1 else cfg.encoder_feat for l in self.ms_levels)
        # "mid": the multiscale features skip the first MLP layer and enter only at the midpoint layer.
        self.ms_at_mid = self.use_multiscale and cfg.multiscale_inject == "mid"
        if not self.ms_at_mid:
            cond_dim += ms_dim

        if cfg.encoder_type == "2.5d":
            self.encoder = ResNetEncoder2p5D(
                in_ch=cfg.patch_z, feat_ch=cfg.encoder_feat,
                global_ch=cfg.global_feat_ch,
                dilation_xy=cfg.dilation_xy,
                depth=cfg.encoder_depth,
            )
        else:
            self.encoder = ResNetEncoder3D(
                in_ch=1, feat_ch=cfg.encoder_feat,
                global_ch=cfg.global_feat_ch,
                dilation_xy=cfg.dilation_xy,
                z_strides=cfg.encoder_z_strides,
                depth=cfg.encoder_depth,
            )

        if self.use_film:
            # SIREN: raw coords → FiLM-conditioned sinusoidal MLP
            # encoder features modulate each layer, never enter the sinusoidal path
            self.pe  = None
            self.inr = FiLMINR(
                coord_dim=3,
                cond_dim=cond_dim,
                hidden=cfg.inr_hidden,
                out_dim=cfg.num_classes,
                depth=cfg.inr_depth,
                omega_0=cfg.siren_omega_0,
            )
        else:
            # ReLU: Fourier PE + concatenated features → standard MLP
            self.pe  = AnisotropicPositionalEncoding(
                num_freqs_xy=cfg.num_freqs_xy,
                num_freqs_z=cfg.num_freqs_z,
            )
            self.inr = INR(
                in_dim=self.pe.out_dim + cond_dim,
                hidden=cfg.inr_hidden,
                out_dim=cfg.num_classes,
                depth=cfg.inr_depth,
                activation="relu",
                omega_0=cfg.siren_omega_0,
                extra_dim=ms_dim if self.ms_at_mid else 0,
            )

        if cfg.feature_supervision:
            self.feature_head: Optional[nn.Module] = nn.Linear(cfg.encoder_feat, cfg.num_classes)
        else:
            self.feature_head = None

        self.dense_decoder: Optional[nn.Module] = (
            DenseDecoder3D(cfg.encoder_feat, cfg.num_classes,
                            use_skip_connections=cfg.dense_decoder_skip_connections,
                            encoder_depth=cfg.encoder_depth)
            if cfg.use_dense_decoder else None
        )

    def decode_3d(
        self,
        feat_vol:    torch.Tensor,   # [B, feat_ch, D, H, W]
        global_feat: torch.Tensor,   # [B, global_ch]
        coords:      torch.Tensor,   # [B, N, 3]  voxel space (x, y, z)
        volume_shape: tuple,         # (D, H, W)
        return_feat_logits: bool = False,
        layer1: Optional[torch.Tensor] = None,   # [B, feat_ch/2, Z, H/2, W/2] — only used if cfg.use_intermediate_features
        layer2: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z', H/4, W/4]
        layer3: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z', H/8, W/8]  — encoder_depth=5 only
        layer4: Optional[torch.Tensor] = None,   # [B, feat_ch,   Z', H/16, W/16] — encoder_depth=5 only
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """Local feature sampling + global context + PE/FiLM + INR MLP.

        Exposed so the training loop can run the encoder once and reuse the
        (detached) features for the smoothness passes without a double forward.
        When return_feat_logits=True, also returns auxiliary logits from a
        linear head applied directly to local features (feature supervision).

        layer1-4 are optional multi-scale skip features (see
        ResNetEncoder3D.forward_multiscale / forward_deep_multiscale); when
        provided and cfg.use_intermediate_features is set, they are
        trilinearly sampled at the same query coords and concatenated into
        cond alongside the bottleneck feature, giving every INR/FiLM layer
        direct access to all scales at once (not a per-layer UNet-style
        staged assignment). layer3/layer4 only exist for encoder_depth=5.
        """
        D, H, W    = volume_shape
        N          = coords.shape[1]
        g_feat_exp = global_feat.unsqueeze(1).expand(-1, N, -1)       # [B, N, global_ch]

        if self.cfg.use_local_features:
            coords_grid = normalize_coords_for_grid_sample(coords, (D, H, W))
            if self.cfg.feature_sampling == "liif":
                local_feat, delta = sample_features_liif(feat_vol, coords_grid)
                cond_parts = [local_feat, g_feat_exp, delta]
            else:
                local_feat = sample_features(feat_vol, coords_grid)      # [B, N, feat_ch]
                cond_parts = [local_feat, g_feat_exp]
            ms_parts = []
            if self.use_multiscale and layer1 is not None and layer2 is not None:
                levels = {1: layer1, 2: layer2, 3: layer3, 4: layer4}    # 1: feat/2 ch, 2-4: feat ch
                for lvl in self.ms_levels:
                    ms_parts.append(sample_features(levels[lvl], coords_grid))   # [B, N, C_lvl]
            ms_feat = torch.cat(ms_parts, dim=-1) if (self.ms_at_mid and ms_parts) else None
            if not self.ms_at_mid:
                cond_parts.extend(ms_parts)
            cond = torch.cat(cond_parts, dim=-1)                         # [B, N, cond_dim]
        else:
            local_feat = None
            ms_feat    = None
            cond       = g_feat_exp                                    # [B, N, global_ch]
        coords_norm = normalize_coords_3d(coords, volume_shape, self.cfg)

        if self.use_film:
            logits = self.inr(coords_norm, cond)                       # FiLM: coords + cond separate
        else:
            pe     = self.pe(coords_norm)                              # [B, N, pe_dim]
            logits = self.inr(torch.cat([pe, cond], dim=-1), extra=ms_feat)   # ReLU: all concatenated (ms_feat only at mid layer)

        if return_feat_logits:
            assert local_feat is not None, "return_feat_logits requires use_local_features=True"
            return logits, self.feature_head(local_feat)               # [B, N, C], [B, N, C]
        return logits

    def forward(
        self,
        volume: torch.Tensor,            # [B, 1, D, H, W]
        coords: torch.Tensor,            # [B, N, 3]  voxel space (x, y, z)
        return_feat_logits: bool = False,
        query_shape: Optional[Tuple[int, int, int]] = None,   # (D, H, W) coords are indexed in;
                                                                # defaults to volume.shape (old behaviour).
                                                                # Pass a larger (H, W) to query the INR at a
                                                                # finer grid than the encoder was fed (e.g.
                                                                # encoder sees 256, coords queried at native 704).
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        D, H, W = query_shape if query_shape is not None else volume.shape[2:]
        if self.use_multiscale:
            if self.cfg.encoder_depth == 5:
                feat_vol, global_feat, layer1, layer2, layer3, layer4 = self.encoder.forward_deep_multiscale(volume)
                return self.decode_3d(feat_vol, global_feat, coords, (D, H, W), return_feat_logits,
                                       layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
            feat_vol, global_feat, layer1, layer2 = self.encoder.forward_multiscale(volume)
            return self.decode_3d(feat_vol, global_feat, coords, (D, H, W), return_feat_logits,
                                   layer1=layer1, layer2=layer2)
        feat_vol, global_feat = self.encoder(volume)
        return self.decode_3d(feat_vol, global_feat, coords, (D, H, W), return_feat_logits)


# ---------------------------------------------------------------------------
# 2.5D Residual U-Net baseline (dense prediction, no INR/coordinate decoding)
# ---------------------------------------------------------------------------

class ResidualUNet2p5D(nn.Module):
    """2.5D encoder (patch_z frames as input channels) + dense 2D decoder.

    Predicts a single [num_classes, H, W] map for the central frame of the
    patch window only — the only frame with real supervision in most windows
    anyway (see dataset.py's window-centering: the annotated frame lands at
    local index patch_z // 2 in the common, non-boundary-clipped case).

    Baseline to compare against a ConditionalINR run with encoder_type="2.5d":
    same ResNetEncoder2p5D construction (same cfg fields), only the decoder
    differs (this dense conv decoder vs the INR's coordinate-sampled MLP).

    cfg.dense_decoder_skip_connections (same flag the 3D diagnostic dense
    decoder uses) toggles real U-Net-style skip connections on this decoder
    — off by default, since the no-skip version is what keeps this a
    decoder-only-difference comparison against the INR (which also never
    sees multi-scale features, only the final feature map). Skip
    connections require encoder_depth=5 (forward_deep_multiscale).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_skip = cfg.dense_decoder_skip_connections
        if self.use_skip:
            assert cfg.encoder_depth == 5, \
                "ResidualUNet2p5D skip connections require encoder_depth=5"
        self.encoder = ResNetEncoder2p5D(
            in_ch=cfg.patch_z, feat_ch=cfg.encoder_feat,
            global_ch=cfg.global_feat_ch,
            dilation_xy=cfg.dilation_xy,
            depth=cfg.encoder_depth,
        )
        self.dense_decoder = Dense2DDecoder(
            cfg.encoder_feat, cfg.num_classes, depth=cfg.encoder_depth,
            use_skip_connections=self.use_skip,
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """[B, 1, patch_z, H, W]  ->  [B, num_classes, H, W]"""
        if self.use_skip:
            feat, _, l1, l2, l3, l4 = self.encoder.forward_deep_multiscale(volume)
            feat2d = feat.squeeze(2)
            return self.dense_decoder(feat2d, layer1=l1, layer2=l2, layer3=l3, layer4=l4)
        feat, _ = self.encoder(volume)     # feat: [B, feat_ch, 1, H/8 (or H/32, H/128), W/...]
        feat2d = feat.squeeze(2)           # drop the D'=1 axis
        return self.dense_decoder(feat2d)


# ---------------------------------------------------------------------------
# 3D Residual U-Net baseline (dense prediction, no INR/coordinate decoding)
# ---------------------------------------------------------------------------

class ResidualUNet3D(nn.Module):
    """3D encoder (ResNetEncoder3D) + dense 3D decoder (DenseDecoder3D), no INR.

    Predicts a full [num_classes, D, H, W] volume in one forward pass — the
    direct 3D counterpart to ResidualUNet2p5D (which predicts only the central
    2D frame). Requires encoder_depth=5 WITH skip connections (asserted
    below): DenseDecoder3D's no-skip path only upsamples XY, never Z, so
    without skip connections the output D would be coarser than the input
    patch — skip connections are what restores full D resolution (see
    DenseDecoder3D's docstring).

    Baseline to compare against a ConditionalINR run with encoder_type="3d":
    same ResNetEncoder3D construction (same cfg fields: encoder_feat,
    global_feat_ch, dilation_xy, encoder_z_strides, encoder_depth), only the
    decoder differs (this dense conv decoder vs the INR's coordinate-sampled
    MLP) — mirrors ResidualUNet2p5D's own role as a decoder-only-difference
    comparison against the 2.5D INR.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.encoder_depth == 5, "ResidualUNet3D requires encoder_depth=5"
        assert cfg.dense_decoder_skip_connections, \
            "ResidualUNet3D requires dense_decoder_skip_connections=True (no-skip depth=5 not implemented)"
        self.cfg = cfg
        self.encoder = ResNetEncoder3D(
            in_ch=1, feat_ch=cfg.encoder_feat,
            global_ch=cfg.global_feat_ch,
            dilation_xy=cfg.dilation_xy,
            z_strides=cfg.encoder_z_strides,
            depth=cfg.encoder_depth,
        )
        self.dense_decoder = DenseDecoder3D(
            cfg.encoder_feat, cfg.num_classes,
            use_skip_connections=True, encoder_depth=5,
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """[B, 1, D, H, W]  ->  [B, num_classes, D, H, W]"""
        feat_vol, _, l1, l2, l3, l4 = self.encoder.forward_deep_multiscale(volume)
        return self.dense_decoder(feat_vol, layer1=l1, layer2=l2, layer3=l3, layer4=l4)
