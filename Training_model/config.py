from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Training dimensionality: "2D" trains on individual annotated frames only;
    # "3D" trains on full pullback volumes with loss only on annotated frames.
    training_mode: str = "3D"

    # Physical spacing (mm) — IV-OCT: 0.01 mm in-plane, 0.10 mm along pullback
    # at the native resolution (native_xy_size). If resize_to is set, xy_spacing
    # is automatically rescaled in __post_init__ so PE math stays correct.
    xy_spacing: float = 0.01
    z_spacing:  float = 0.10

    # Native in-plane resolution of the source data (used to rescale xy_spacing
    # automatically when resize_to is set).
    native_xy_size: int = 704

    # If > 0, resize each frame's (H, W) to (resize_to, resize_to) at load time —
    # bilinear for the image, nearest-neighbour for labels. Reduces encoder
    # memory/compute and low-pass filters OCT speckle. 0 disables resizing.
    resize_to: int = 0

    # Model
    num_classes:  int   = 12
    encoder_feat: int   = 64      # output channels of 3D encoder
    use_local_features:   bool  = True   # if False, local encoder features are dropped; only global conditioning remains
    use_dense_decoder:    bool  = False  # add symmetric upsampling decoder as diagnostic head alongside the INR
    dense_decoder_weight: float = 1.0   # CE loss weight for the dense decoder
    # Give the dense decoder real UNet-style skip connections from encoder layer1(H/2)/layer2(H/4)
    # instead of no-skip upsampling. Only used when use_dense_decoder=True.
    dense_decoder_skip_connections: bool = False
    num_freqs_xy: int   = 6       # Fourier frequencies for x, y (fine in-plane detail)
    num_freqs_z:  int   = 4       # fewer frequencies for coarser z axis
    inr_hidden:   int   = 512
    inr_depth:    int   = 4

    # Training
    batch_size:   int   = 2
    num_epochs:   int   = 100
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    n_points:     int   = 8192    # query coordinates sampled per volume per step

    # Loss weights
    dice_weight:          float = 1.0

    # Main loss function: "ce_dice" (cross-entropy + dice) or "mse_onehot" (MSE on one-hot targets).
    loss_type: str = "mse_onehot"
    # 3D temporal smoothness: penalises class-prob changes between consecutive z-frames.
    # Improves pullback consistency. Enable AFTER 2D per-frame quality is satisfactory.
    smoothness_3d_weight: float = 0.0
    # 2D spatial smoothness: penalises adjacent (x,y) pixel disagreement on the annotated frame.
    # Directly targets pixelated predictions. Enable first before smoothness_3d_weight.
    smoothness_2d_weight: float = 0.0
    smoothness_2d_n_pairs: int  = 64   # adjacent coordinate pairs sampled per step

    # Classes to apply both smoothness losses to. Empty list = all classes.
    # Device classes (stent, shadow, …) are typically excluded as they can appear/disappear fast.
    smoothness_class_indices: List[int] = field(default_factory=list)

    # Radial anatomical prior (2026-07-24): frames are centered on the catheter, so sampling
    # along rays from image center lets us penalise two curated, always-true geometric errors —
    # Lumen predicted farther from center than tissue, and Guidewire predicted with a gap back
    # to center (the phantom-blob pattern). Off by default (0.0) — zero effect on existing runs.
    # Split into two independent weights (2026-07-31): radial_media_boundary_loss was confirmed
    # inert (stuck at exactly 0.0000 for a full 300-epoch run, diluted by averaging over ~66
    # near/far ray-pairs) while radial_guidewire_truncation_loss was demonstrably active — a
    # shared weight meant scaling one to compensate also scaled the other, which made no sense
    # once their behavior diverged this much. Previously a single `radial_prior_weight` applied
    # to both; any config still using that key is simply ignored now (defaults both new fields
    # to 0.0), not silently misread.
    radial_media_prior_weight:     float = 0.0
    radial_guidewire_prior_weight: float = 0.0
    radial_prior_n_rays:   int   = 8    # rays sampled per annotated frame per step
    radial_prior_n_radii:  int   = 12   # radius samples per ray, near -> far

    # Inference: ignore predictions beyond this many frames from nearest labeled frame
    max_propagation_frames: int = 45   # ≈ 4.5 mm at 0.1 mm/frame

    # Validation
    val_interval:   int = 5    # run validation every N epochs

    # LR warmup: linearly ramp from lr*warmup_start_factor to lr over this many epochs
    warmup_epochs:       int   = 20
    warmup_start_factor: float = 0.1

    # Context frames (2D mode only): stack this many frames before/after the
    # annotated frame as extra input channels → in_channels = 2*context_frames+1
    context_frames: int = 0

    # Pre-split per-frame .npz folder produced by preprocess_frames.py --mode 2d.
    # Leave empty to fall back to loading full pullback .npz files.
    frames_folder: str = ""

    # Pre-split 3D patch .npz folder produced by preprocess_frames.py --mode 3d.
    # Leave empty to fall back to on-the-fly patch extraction from full volumes.
    patches_folder_3d: str = ""

    # Output
    checkpoint_dir: str = "checkpoints"
    best_ckpt_name: str = "best_model.pt"

    # Sentinel value for truly unlabeled pixels (non-annotated frames in 3D mode).
    # Must NOT equal any real class label (0–11). All 12 classes including
    # background (class 0) participate in the loss and mIoU.
    ignore_index:       int  = 255
    ignore_background:  bool = False   # exclude class 0 from CE and Dice

    # Deep supervision: auxiliary CE+Dice loss applied directly on encoder
    # feature logits (before the INR MLP). Forces the encoder to produce
    # linearly-separable features and shortens the gradient path to the encoder.
    feature_supervision:        bool  = False
    feature_supervision_weight: float = 0.2

    # Global context pooling (2D mode only): pool the encoder bottleneck to a
    # compact vector and concatenate it with every query coordinate's local
    # feature before the INR MLP.  Gives each point awareness of the full-image
    # tissue structure, reducing wrong-class predictions in globally implausible
    # locations.
    global_feat_ch: int = 32

    # Encoder-only mode (2D): skip the UNet decoder; instead bilinearly
    # interpolate the bottleneck feature map back to input resolution.
    # Faster and uses less memory; loses the skip-connection detail.
    encoder_only: bool = False

    # Multi-scale skip features (3D): in addition to the bottleneck feat_vol,
    # also trilinearly sample the encoder's intermediate layers at each query
    # coordinate and concatenate all of them into cond. Every INR/FiLM layer
    # sees all scales at once (flat concatenation, not a per-layer staged
    # UNet-style assignment). Requires use_local_features=True.
    # encoder_depth=3: layer1 (H/2, feat_ch/2 ch) + layer2 (H/4, feat_ch ch).
    # encoder_depth=5: layer1 (H/2, feat_ch/2 ch) + layer2 (H/4) + layer3 (H/8)
    #                  + layer4 (H/16), the latter two at feat_ch channels each.
    # encoder_depth=7 not supported (no forward_deep_multiscale equivalent).
    use_intermediate_features: bool = False

    # Spatial TV regularisation weight applied to the dense feature-head output.
    # Penalises isolated wrong-class predictions (e.g. background inside lumen).
    # Only active when feature_supervision=True (dense head must be enabled).
    tv_weight: float = 0.00

    # Sub-pixel coordinate jitter (2026-07-31): at train time, offset each sampled query
    # point's (x, y) by a fresh uniform draw from [-coord_jitter_max, +coord_jitter_max]
    # pixels (z untouched -- each z-frame is its own independent annotation, no continuous
    # ground truth between frames to jitter into). The pixel's own label is kept unchanged
    # regardless of the offset -- no soft/blended labels, just denser hard-label supervision
    # right up to class boundaries, constraining the INR's behavior in the gap between
    # adjacent pixels that today only ever gets two point-supervised endpoints. 0.0 disables
    # (exact pixel centers, byte-identical to previous behavior). 0.5 = full pixel footprint
    # (every point in [x-0.5, x+0.5) x [y-0.5, y+0.5) is equally likely). Validation always
    # uses exact pixel centers regardless of this setting, so Val Dice stays comparable
    # across every prior run.
    coord_jitter_max: float = 0.0

    # 3D patch-based training: number of z-frames loaded per sample.
    # Encoder sees this many frames; only the annotated frame within the patch
    # contributes to the supervised loss.  Patch is centered on the annotated frame.
    patch_z: int = 32

    # Coordinate sampling strategy during training.
    # "random"     — uniform random draw from labeled pixels (70 %) + random (30 %)
    # "stratified" — labeled budget split equally across present classes, so rare
    #                classes (calcium, thrombus, …) get the same representation as
    #                dominant ones (lumen, intima, background).
    sampling_strategy: str = "stratified"

    # Pre-load patch .npz files into RAM at dataset init to avoid disk I/O
    # during training. Requires enough system RAM (request ~1.5× dataset size).
    # preload_frac controls what fraction is loaded upfront (0.0–1.0); the
    # remainder is cached lazily on first access during training.
    preload_data: bool = False
    preload_frac: float = 0.5

    # Limit training to this fraction of available patches per epoch.
    # 0.0 = use all patches. 0.25 = 25% per epoch → ~4× faster epochs, same epoch count.
    # Useful for frequent validation feedback without reducing total training time.
    max_batches_frac: float = 0.0

    # Validation interval as fraction of total epochs. 0.0 = use val_interval (absolute).
    # 0.05 = every 5% of num_epochs → ~20 checkpoints regardless of total epoch count.
    # Always validates at epoch 1 and epoch num_epochs.
    val_interval_pct: float = 0.0

    # Minimum fraction of total query points (n_pts) guaranteed to be background
    # (class 0) when sampling_strategy="stratified". Without this floor, background
    # drops from ~60% (random) to 1/n_classes (~8%), causing scattered predictions
    # in background regions. 0.0 disables the floor (pure stratified).
    # Recommended: 0.2 (reserves 20% of n_pts for background before distributing
    # the remainder equally across all other present classes).
    background_floor_frac: float = 0.0

    # Mixed-precision training (AMP). Float16 overflows at ~65504; with ω₀=30
    # SIREN gradients can exceed this → GradScaler silently skips those steps.
    # Use bfloat16 (wider exponent range) or disable entirely for SIREN training.
    use_amp: bool = False

    # Gradient clipping: global L2 norm clipped to this value each step.
    # 0.0 disables clipping. SIREN init already bounds gradient magnitudes;
    # clipping at 1.0 is too aggressive (effective LR ≈ 1/actual_norm × lr).
    grad_clip_norm: float = 0.0

    # INR activation function: "relu" (default) or "siren" (Sitzmann et al. 2020).
    # SIREN uses sin(omega_0 * x) with a matching weight init that keeps the sin
    # argument in its most expressive range — better for smooth curved boundaries.
    inr_activation: str   = "relu"
    siren_omega_0:  float = 30.0

    # ResNet encoder: dilation rate for the last XY conv stage.
    # 1 = standard stride-2 conv (bottleneck H/8, W/8).
    # >1 = dilated conv, no stride (bottleneck H/4, W/4, wider receptive field).
    dilation_xy: int = 1

    # Number of downsampling stages in the 3D encoder. 3 = original (layer1-3,
    # bottleneck H/8). 5 = deeper variant with layer4/layer5 added (bottleneck
    # H/32), tests whether encoder capacity/depth (not just skip connections)
    # was the limiting factor vs a full UNet baseline. 7 = layer6/layer7 added
    # (bottleneck H/128) — INR path (use_local_features/LIIF/trilinear) only;
    # DenseDecoder3D does not support depth=7. Only 3, 5, 7 supported.
    encoder_depth: int = 3

    # Per-layer Z stride for the 3D encoder [layer1, layer2, ...] — length must
    # match encoder_depth.
    # 1 = no Z downsampling (default — Z is already coarse at 0.10 mm/frame).
    # 2 = stride-2 in Z for that layer; automatically switches the conv kernel
    #     from (1,3,3) to (3,3,3) so Z neighbours are aggregated before striding.
    # Example (depth=3): [1, 2, 1] halves Z at layer2 → feat_vol [B,C,D/2,H/8,W/8].
    # Example (depth=5): [1, 1, 2, 1, 1] halves Z at layer3 (centered) → feat_vol [B,C,D/2,H/32,W/32].
    encoder_z_strides: List[int] = field(default_factory=lambda: [1, 1, 1])

    # Feature sampling mode when use_local_features=True.
    # "trilinear" — standard trilinear interpolation (current default).
    # "liif"      — nearest-cell sampling (LIIF style): all pixels in the same
    #               encoder cell share one feature vector; a normalised delta
    #               offset (3 dims, range [-0.5, 0.5]) is appended to cond so
    #               the SIREN knows its sub-cell position. Reduces per-pixel
    #               FiLM specificity → smoother grouped predictions.
    feature_sampling: str = "trilinear"

    # Unconditional INR: overfit a single MLP on one 3D volume (no encoder).
    # Maps AnisotropicPE(x, y, z) → class logits directly.
    # Use to study INR representational capacity and smoothness before adding conditioning.
    unconditional:           bool  = False
    unconditional_pullback:  str   = ""    # pullback stem to overfit; empty = first in training set
    unconditional_n_repeats: int   = 100   # dataset length for train (= steps per epoch with batch_size=1)

    def __post_init__(self):
        if self.resize_to:
            self.xy_spacing *= self.native_xy_size / self.resize_to
        assert self.encoder_depth in (3, 5, 7), \
            f"encoder_depth must be 3, 5, or 7, got {self.encoder_depth}"
        assert len(self.encoder_z_strides) == self.encoder_depth, \
            f"encoder_z_strides must have exactly {self.encoder_depth} values, got {self.encoder_z_strides}"
        assert self.feature_sampling in ("trilinear", "liif"), \
            f"feature_sampling must be 'trilinear' or 'liif', got '{self.feature_sampling}'"
        assert not (self.use_intermediate_features and self.encoder_depth not in (3, 5)), \
            "use_intermediate_features (INR multiscale conditioning) is only implemented for " \
            "encoder_depth=3 or 5 so far (no forward_deep_multiscale equivalent for depth=7)."
