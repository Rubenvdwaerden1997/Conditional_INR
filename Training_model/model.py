"""
Model components for the Conditional INR.

Coordinate convention used throughout:
  coords[..., 0] = x  (width,  W dimension)
  coords[..., 1] = y  (height, H dimension)
  coords[..., 2] = z  (depth,  pullback axis / frame index)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import Config


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def scale_to_mm(coords: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Convert voxel-index coordinates to physical millimetre space.

    coords : [..., 3]  (x, y, z) in voxel indices
    returns: [..., 3]  in mm
    """
    scale = torch.tensor(
        [cfg.xy_spacing, cfg.xy_spacing, cfg.z_spacing],
        device=coords.device, dtype=coords.dtype,
    )
    return coords * scale


def scale_to_mm_2d(coords: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Convert 2-D voxel-index coordinates to physical millimetre space.

    coords : [..., 2]  (x, y) in voxel indices
    returns: [..., 2]  in mm
    """
    scale = torch.tensor(
        [cfg.xy_spacing, cfg.xy_spacing],
        device=coords.device, dtype=coords.dtype,
    )
    return coords * scale


def normalize_coords_for_grid_sample(
    coords: torch.Tensor,
    volume_shape: Tuple[int, int, int],   # (D, H, W)
) -> torch.Tensor:
    """Normalise voxel coords to [-1, 1] for F.grid_sample.

    grid_sample expects its grid in (x, y, z) = (W, H, D) order — this
    function does that flip explicitly to avoid silent sampling errors.

    coords : [B, N, 3]  (x, y, z) voxel order
    returns: [B, N, 3]  grid_sample-compatible (W-norm, H-norm, D-norm)
    """
    D, H, W = volume_shape
    x = 2.0 * coords[..., 0] / (W - 1) - 1.0
    y = 2.0 * coords[..., 1] / (H - 1) - 1.0
    z = 2.0 * coords[..., 2] / (D - 1) - 1.0
    return torch.stack([x, y, z], dim=-1)


def normalize_coords_for_grid_sample_2d(
    coords: torch.Tensor,
    image_shape: Tuple[int, int],   # (H, W)
) -> torch.Tensor:
    """Normalise 2-D voxel coords to [-1, 1] for F.grid_sample.

    coords : [B, N, 2]  (x, y) voxel order
    returns: [B, N, 2]  grid_sample-compatible (W-norm, H-norm)
    """
    H, W = image_shape
    x = 2.0 * coords[..., 0] / (W - 1) - 1.0
    y = 2.0 * coords[..., 1] / (H - 1) - 1.0
    return torch.stack([x, y], dim=-1)


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


def sample_features_2d(
    feat_map: torch.Tensor,      # [B, C, H, W]
    coords_norm: torch.Tensor,   # [B, N, 2]  grid_sample convention
) -> torch.Tensor:               # [B, N, C]
    """Bilinearly sample a feature map at normalised 2-D coordinates."""
    B, N, _ = coords_norm.shape
    grid     = coords_norm.view(B, N, 1, 2)
    sampled  = F.grid_sample(
        feat_map, grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [B, C, N, 1]
    return sampled.view(B, feat_map.shape[1], N).permute(0, 2, 1)  # [B, N, C]


# ---------------------------------------------------------------------------
# Anisotropic positional encoding
# ---------------------------------------------------------------------------

class AnisotropicPositionalEncoding(nn.Module):
    """Fourier positional encoding with separate frequency schedules per axis.

    IV-OCT has a 10:1 anisotropy (0.01 mm xy vs 0.10 mm z), so xy gets
    more frequencies (fine detail) and z gets fewer (coarse pullback axis).
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

    def forward(self, coords_mm: torch.Tensor) -> torch.Tensor:
        """coords_mm: [..., 3] in physical mm  →  [..., out_dim]"""
        x = coords_mm[..., 0:1]
        y = coords_mm[..., 1:2]
        z = coords_mm[..., 2:3]

        def fourier(val, freqs):
            angles = val * freqs
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return torch.cat([
            coords_mm,
            fourier(x, self.freqs_xy),
            fourier(y, self.freqs_xy),
            fourier(z, self.freqs_z),
        ], dim=-1)


# ---------------------------------------------------------------------------
# 2D positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """Fourier positional encoding for 2-D (x, y) coordinates only."""

    def __init__(self, num_freqs_xy: int = 6):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_freqs_xy).float()
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 + 2 * num_freqs_xy + 2 * num_freqs_xy   # raw + sin/cos x + sin/cos y

    def forward(self, coords_mm: torch.Tensor) -> torch.Tensor:
        """coords_mm: [..., 2] in physical mm  →  [..., out_dim]"""
        x = coords_mm[..., 0:1]
        y = coords_mm[..., 1:2]

        def fourier(val):
            angles = val * self.freqs
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return torch.cat([coords_mm, fourier(x), fourier(y)], dim=-1)


# ---------------------------------------------------------------------------
# 3D encoder
# ---------------------------------------------------------------------------

class Encoder3D(nn.Module):
    """Anisotropic 3D encoder for patch-based training.

    Downsamples aggressively in XY (8× total via strided stem + 2 MaxPool stages)
    while keeping Z at full patch resolution — matching the 10:1 physical
    anisotropy of IV-OCT (0.01 mm in-plane vs 0.10 mm along pullback).

    Memory estimate for patch_z=32, H=W=704, feat_ch=64 (batch 1):
        stem out : [16, 32, 352, 352] ≈ 256 MB
        enc2 out : [32, 32, 352, 352] ≈ 512 MB
        enc3 out : [64, 32, 176, 176] ≈ 256 MB
        enc4 out : [64, 32,  88,  88] ≈  64 MB
        output   : [64, 32,  88,  88] ≈  64 MB   (feat_ch × patch_z × H/8 × W/8)

    Input:  [B, 1,       patch_z, H,    W   ]
    Output: [B, feat_ch, patch_z, H//8, W//8]
    """

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        groups = min(8, out_ch)
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def __init__(self, in_ch: int = 1, feat_ch: int = 64,
                 use_intermediate_features: bool = False):
        super().__init__()
        b = feat_ch // 4   # base channels (16 for feat_ch=64)
        self.b = b
        self.use_intermediate_features = use_intermediate_features

        # Strided stem: immediate 2× xy reduction — avoids storing a full-res [Z,H,W] activation
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, b, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=False),
            nn.GroupNorm(min(8, b), b),
            nn.ReLU(inplace=True),
        )                                           # → [B, b,   Z, H/2, W/2]

        self.enc2 = self._block(b,     b * 2)      # [B, 2b,  Z, H/2, W/2]
        self.enc3 = self._block(b * 2, b * 4)      # [B, 4b,  Z, H/4, W/4]  (after pool)
        self.enc4 = self._block(b * 4, b * 4)      # [B, 4b,  Z, H/8, W/8]  (after pool)

        # Z-context aggregation at bottleneck (lightweight: one (3,1,1) conv)
        self.z_conv = nn.Sequential(
            nn.Conv3d(b * 4, feat_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(min(8, feat_ch), feat_ch),
            nn.ReLU(inplace=True),
        )                                           # → [B, feat_ch, Z, H/8, W/8]

        self.pool   = nn.MaxPool3d(kernel_size=(1, 2, 2))   # xy-only pooling
        self.out_ch = feat_ch

    def forward(self, x: torch.Tensor):
        """[B, 1, Z, H, W]  →  (feat [B, feat_ch, Z, H//8, W//8], skip_maps)

        skip_maps is [s_stem, s2, s3] with shapes
          [B, b, Z, H/2, W/2], [B, 2b, Z, H/2, W/2], [B, 4b, Z, H/4, W/4]
        or an empty list when use_intermediate_features is False.
        """
        s_stem = self.stem(x)                 # [B, b,   Z, H/2, W/2]
        s2     = self.enc2(s_stem)            # [B, 2b,  Z, H/2, W/2]
        s3     = self.enc3(self.pool(s2))     # [B, 4b,  Z, H/4, W/4]
        s4     = self.enc4(self.pool(s3))     # [B, 4b,  Z, H/8, W/8]
        feat   = self.z_conv(s4)              # [B, feat_ch, Z, H/8, W/8]

        skip_maps = [s_stem, s2, s3] if self.use_intermediate_features else []
        return feat, skip_maps


# ---------------------------------------------------------------------------
# 2D encoder
# ---------------------------------------------------------------------------

class Encoder2D(nn.Module):
    """Multi-scale 2-D encoder with 3 downsampling stages and skip connections.

    Three 2× downsampling stages increase the receptive field from 7 px (old
    flat encoder) to ~60 px, giving the model context over tissue layers that
    span hundreds of pixels in OCT.  Skip connections preserve fine-grained
    detail for accurate boundary localisation.

    Receptive field per stage (approx):
        s1 (full res)  : 7 px
        s2 (×½ res)    : 16 px in original space
        s3 (×¼ res)    : 36 px in original space
        s4 (×⅛ res)    : ~60 px in original space
    """

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        groups = min(8, out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def __init__(self, in_ch: int = 1, feat_ch: int = 64, global_ch: int = 32,
                 encoder_only: bool = False, use_intermediate_features: bool = False):
        super().__init__()
        b = feat_ch // 4   # base channel count  (16 for feat_ch=64)
        self.encoder_only             = encoder_only
        self.use_intermediate_features = use_intermediate_features

        self.enc1 = self._block(in_ch, b)          # full res  → [B, b,   H,   W]
        self.enc2 = self._block(b,     b * 2)      # ×½ res    → [B, 2b,  H/2, W/2]
        self.enc3 = self._block(b * 2, b * 4)      # ×¼ res    → [B, 4b,  H/4, W/4]
        self.enc4 = self._block(b * 4, b * 4)      # ×⅛ res    → [B, 4b,  H/8, W/8]

        if not encoder_only:
            self.dec3 = self._block(b * 4 + b * 4, b * 4)  # after cat skip3
            self.dec2 = self._block(b * 4 + b * 2, b * 2)  # after cat skip2
            self.dec1 = self._block(b * 2 + b,     b)      # after cat skip1
            self.out_conv = nn.Conv2d(b, feat_ch, kernel_size=1)
        else:
            # No decoder: bottleneck (b*4 channels) is interpolated directly to input res.
            self.out_conv = nn.Conv2d(b * 4, feat_ch, kernel_size=1)

        self.out_ch   = feat_ch
        self.pool     = nn.MaxPool2d(2)

        # Global context: pool the bottleneck to a single vector per image.
        # Each query coordinate concatenates this with its local feature so
        # the INR MLP can see the whole-image tissue structure.
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b * 4, global_ch),
            nn.ReLU(inplace=True),
        )
        self.global_ch = global_ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """[B, in_ch, H, W]  →  ([B, feat_ch, H, W], [B, global_ch], skip_maps)

        skip_maps is a list of raw encoder feature maps [s1, s2, s3] with
        shapes [B, b, H, W], [B, 2b, H/2, W/2], [B, 4b, H/4, W/4], or an
        empty list when use_intermediate_features is False.
        """
        H, W = x.shape[2], x.shape[3]
        s1 = self.enc1(x)                                          # [B, b,   H,   W]
        s2 = self.enc2(self.pool(s1))                              # [B, 2b,  H/2, W/2]
        s3 = self.enc3(self.pool(s2))                              # [B, 4b,  H/4, W/4]
        s4 = self.enc4(self.pool(s3))                              # [B, 4b,  H/8, W/8]

        global_feat = self.global_proj(s4)                         # [B, global_ch]

        skip_maps: list = [s1, s2, s3] if self.use_intermediate_features else []

        if self.encoder_only:
            feat = F.interpolate(s4, size=(H, W), mode="bilinear", align_corners=False)
            return self.out_conv(feat), global_feat, skip_maps

        d3 = self.dec3(torch.cat([F.interpolate(s4, size=s3.shape[2:], mode="bilinear",
                                                align_corners=False), s3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=s2.shape[2:], mode="bilinear",
                                                align_corners=False), s2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=s1.shape[2:], mode="bilinear",
                                                align_corners=False), s1], dim=1))

        return self.out_conv(d1), global_feat, skip_maps


# ---------------------------------------------------------------------------
# INR MLP
# ---------------------------------------------------------------------------

class INR(nn.Module):
    """MLP with a NeRF-style skip connection at the midpoint for stability."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 4):
        super().__init__()
        assert depth >= 2, "Need at least 2 layers for the skip connection"

        mid = depth // 2
        self.input_layer      = nn.Linear(in_dim, hidden)
        self.layers_pre_skip  = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(mid - 1)])
        self.skip_layer       = nn.Linear(in_dim + hidden, hidden)
        self.layers_post_skip = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - mid - 1)])
        self.output_layer     = nn.Linear(hidden, out_dim)
        self.act              = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., in_dim]  →  [..., out_dim]"""
        h = self.act(self.input_layer(x))
        for layer in self.layers_pre_skip:
            h = self.act(layer(h))
        h = self.act(self.skip_layer(torch.cat([x, h], dim=-1)))
        for layer in self.layers_post_skip:
            h = self.act(layer(h))
        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ConditionalINR(nn.Module):
    """Encoder + positional encoding + INR MLP.

    Supports two modes controlled by cfg.training_mode:
      "2D" — single-frame encoder (Encoder2D) with 2-D positional encoding.
             volume: [B, 1, H, W],  coords: [B, N, 2]
      "3D" — anisotropic 3-D encoder (Encoder3D) with separate xy/z frequencies.
             volume: [B, 1, D, H, W],  coords: [B, N, 3]
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.mode_2d = cfg.training_mode == "2D"

        if self.mode_2d:
            in_ch        = 2 * cfg.context_frames + 1
            self.encoder = Encoder2D(in_ch=in_ch, feat_ch=cfg.encoder_feat,
                                     global_ch=cfg.global_feat_ch,
                                     encoder_only=cfg.encoder_only,
                                     use_intermediate_features=cfg.use_intermediate_features)
            self.pe      = PositionalEncoding2D(num_freqs_xy=cfg.num_freqs_xy)
            # s1 + s2 + s3 raw channel counts: b + 2b + 4b = 7b  (b = encoder_feat // 4)
            b = cfg.encoder_feat // 4
            skip_dims = (b + 2 * b + 4 * b) if cfg.use_intermediate_features else 0
            in_dim = self.pe.out_dim + cfg.encoder_feat + cfg.global_feat_ch + skip_dims
        else:
            self.encoder = Encoder3D(in_ch=1, feat_ch=cfg.encoder_feat,
                                     use_intermediate_features=cfg.use_intermediate_features)
            self.pe      = AnisotropicPositionalEncoding(
                num_freqs_xy=cfg.num_freqs_xy,
                num_freqs_z=cfg.num_freqs_z,
            )
            # s_stem + s2 + s3 raw channel counts: b + 2b + 4b = 7b
            b_3d = cfg.encoder_feat // 4
            skip_dims_3d = (b_3d + 2 * b_3d + 4 * b_3d) if cfg.use_intermediate_features else 0
            in_dim = self.pe.out_dim + cfg.encoder_feat + skip_dims_3d

        self.inr = INR(in_dim=in_dim, hidden=cfg.inr_hidden,
                       out_dim=cfg.num_classes, depth=cfg.inr_depth)

        # Optional deep-supervision head: small conv stack on encoder feature map.
        # Produces a dense class prediction [B, num_classes, H, W] that is
        # supervised with CE against the full GT frame. Training only; not used
        # at inference time.
        if cfg.feature_supervision:
            conv = nn.Conv2d if self.mode_2d else nn.Conv3d
            mid  = cfg.encoder_feat // 2
            self.feature_head: Optional[nn.Module] = nn.Sequential(
                conv(cfg.encoder_feat, mid, kernel_size=1),
                nn.ReLU(inplace=True),
                conv(mid, cfg.num_classes, kernel_size=1),
            )
        else:
            self.feature_head = None

    def decode_3d(
        self,
        feat_vol: torch.Tensor,   # [B, feat_ch, D, H, W]
        skips: list,              # intermediate encoder maps
        coords: torch.Tensor,     # [B, N, 3]  voxel space (x, y, z)
        volume_shape: tuple,      # (D, H, W)
    ) -> torch.Tensor:            # [B, N, num_classes]
        """Feature sampling + PE + INR for 3D queries.

        Exposed so the training loop can run the encoder once and pass cached
        (detached) features to the smoothness pass, avoiding a double forward.
        """
        D, H, W     = volume_shape
        coords_norm = normalize_coords_for_grid_sample(coords, (D, H, W))
        feat        = sample_features(feat_vol, coords_norm)
        for skip_map in skips:
            feat = torch.cat([feat, sample_features(skip_map, coords_norm)], dim=-1)
        # Centre xy at catheter and z at patch midpoint so the PE reflects
        # distance from the structural centre, not from the patch edge.
        center    = coords.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0])
        coords_mm = scale_to_mm(coords - center, self.cfg)
        pe        = self.pe(coords_mm)
        return self.inr(torch.cat([pe, feat], dim=-1))

    def forward(
        self,
        volume: torch.Tensor,            # [B, 1, H, W] (2D) or [B, 1, D, H, W] (3D)
        coords: torch.Tensor,            # [B, N, 2] (2D) or [B, N, 3] (3D), voxel space
        return_feat_logits: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        # [B, N, num_classes], or (logits, feat_logits) both [B, N, num_classes]
        if self.mode_2d:
            H, W                     = volume.shape[2], volume.shape[3]
            feat_vol, g_feat, skips  = self.encoder(volume)                        # [B, C, H, W], [B, G], list
            coords_norm              = normalize_coords_for_grid_sample_2d(coords, (H, W))
            feat                     = sample_features_2d(feat_vol, coords_norm)   # [B, N, C]
            N                        = feat.shape[1]
            g_feat_exp               = g_feat.unsqueeze(1).expand(-1, N, -1)       # [B, N, global_ch]
            feat                     = torch.cat([feat, g_feat_exp], dim=-1)       # [B, N, C+global_ch]
            # Sample each intermediate skip map at the same query coordinates.
            # align_corners=True ensures the same [-1, 1] range maps to the same
            # physical location regardless of each feature map's spatial resolution.
            for skip_map in skips:
                feat = torch.cat([feat, sample_features_2d(skip_map, coords_norm)], dim=-1)
            # Shift origin to image center (catheter location) before PE so the
            # encoding reflects distance from catheter, not distance from top-left.
            center                   = coords.new_tensor([(W - 1) / 2.0, (H - 1) / 2.0])
            coords_mm                = scale_to_mm_2d(coords - center, self.cfg)
        else:
            D, H, W         = volume.shape[2], volume.shape[3], volume.shape[4]
            feat_vol, skips = self.encoder(volume)
            logits          = self.decode_3d(feat_vol, skips, coords, (D, H, W))
            if return_feat_logits and self.feature_head is not None:
                return logits, self.feature_head(feat_vol)
            return logits

        # --- 2D only below this point ---
        pe     = self.pe(coords_mm)                                        # [B, N, pe_dim]
        logits = self.inr(torch.cat([pe, feat], dim=-1))                   # [B, N, num_classes]

        if return_feat_logits and self.feature_head is not None:
            return logits, self.feature_head(feat_vol)

        return logits
