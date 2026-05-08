from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Training dimensionality: "2D" trains on individual annotated frames only;
    # "3D" trains on full pullback volumes with loss only on annotated frames.
    training_mode: str = "3D"

    # Physical spacing (mm) — IV-OCT: 0.01 mm in-plane, 0.10 mm along pullback
    xy_spacing: float = 0.01
    z_spacing:  float = 0.10

    # Model
    num_classes:  int   = 12
    encoder_feat: int   = 64      # output channels of 3D encoder
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
    dice_weight:       float = 1.0
    smoothness_weight: float = 0.1

    # Classes to apply smoothness to. Empty list = all classes.
    # Device classes (stent, shadow, …) are typically excluded as they can appear/disappear fast.
    smoothness_class_indices: List[int] = field(default_factory=list)

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
    ignore_index: int = 255

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

    # Intermediate encoder features (2D + 3D): sample skip feature maps at each
    # query coordinate and concatenate them into the INR input.
    # Gives the MLP direct access to multi-scale context at every resolution.
    # Adds b + 2b + 4b = 7b dims to the INR input (b = encoder_feat // 4).
    # 2D: samples s1/s2/s3 (full, ½, ¼ res).  3D: samples s_stem/s2/s3 (½, ½, ¼ res).
    use_intermediate_features: bool = False

    # Spatial TV regularisation weight applied to the dense feature-head output.
    # Penalises isolated wrong-class predictions (e.g. background inside lumen).
    # Only active when feature_supervision=True (dense head must be enabled).
    tv_weight: float = 0.00

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
