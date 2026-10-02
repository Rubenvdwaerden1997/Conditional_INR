"""Every option of the project, with its default, as plain dataclasses.

An experiment YAML under `experiments/` sets only the keys that matter for that experiment; every
other key keeps the default below. `--set a.b=c` on the command line overrides either. Values that
are tuples here may be written as YAML lists.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass

import yaml

TASKS = ("pretrain", "seg")


@dataclass
class StoreCfg:
    """`oct-fm build-store`: raw images + label volumes + split sheet -> the Cartesian frame store."""
    dst: str = ""                         # required: `--dst` (where the store is written)
    excel: str = ""                       # required: `--excel` (the split sheet)
    excel_sheet: str = "train_val_test_split"
    label_root: str = ""                  # required: `--label-root` (holds <id><label_suffix>.nii.gz)
    label_suffix: str = "_HPNV"
    image_roots: tuple[str, ...] = ()     # required: `--image-root` (repeatable); each root is searched
                                          # as <root>/<id>.<suf> then <root>/*/<id>.<suf>
    include_unlabelled: bool = True       # every image under the roots not in the sheet -> split "unlabelled"
    center: tuple[float, float] = (352.0, 352.0)   # (cx, cy) of the imaging disc, px
    disc_radius: float = 352.0            # disc used for the mean/std statistics, px
    # frames -> (pullback_mm, z_spacing_mm); OPTIS survey 75mm/375f, standard 54mm/540f, 54mm/270f
    z_spacing_table: dict = field(default_factory=lambda: {375: (75.0, 0.2), 540: (54.0, 0.1),
                                                           270: (54.0, 0.2)})
    chunk_frames: int = 1
    shard_frames: int = 64
    zstd_level: int = 5
    lut_path: str = ""                    # "" -> vendor table; else an existing lut.npz
    read_batch: int = 32                  # frames per streamed read batch
    qc_frames_per_pullback: int = 2       # first manual frame + middle frame
    orientation_min_contrast: float = 30.0
    disk_reserve_bytes: int = 60 * 10**9


@dataclass
class FramesCfg:
    """The dataloader shared by every experiment: polar windows of `window` frames from the store."""
    store: str = ""                       # required: `--store` (the frame store built by build-store)
    window: int = 8                       # frames per sample (ViT: a multiple of model.tubelet[0]; dynunet: odd)
    stride: int | tuple[int, ...] = 1     # frame step inside the window; a list -> one drawn per sample (train)
    z_reference_mm: float = 0.0           # 0 = off; else the stride is multiplied per pullback by
                                          # round(z_reference_mm / z_spacing_mm) (min 1; 1 when the spacing is
                                          # unknown), so a window spans the same length of vessel on 0.1 mm
                                          # and 0.2 mm pullbacks. Train and eval, every task.
    polar_r0: int = 32                    # first radius kept, px (32 crops the catheter dead zone)
    polar_r1: int = 352                   # last radius, px; rows = polar_r1 - polar_r0
    polar_n_theta: int = 704              # angular bins over a full turn
    label_source: str = "manual"          # which frames may be anchors: manual | pseudo | all
    class_map: str | dict = "preset12"    # raw17 | preset12 | {src_name_or_id: dst_id, "names": [...]}
    sampler: str = "uniform"              # uniform | class_balanced (rare classes drawn more often)
    sampler_alpha: float = 0.5            # class_balanced: weight ~ class_frequency ** -alpha
    sampler_cap: float = 50.0             # class_balanced: max weight of one frame
    samples_per_epoch: int = 0            # draws per epoch; 0 -> len(dataset)
    batch_size: int = 8
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    mp_context: str = "fork"              # fork | spawn
    normalize: bool = True                # (x - image_mean) / image_std from dataset_info.json


@dataclass
class ModelCfg:
    """The ViT-S encoder shared by pretraining and segmentation."""
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    tubelet: tuple[int, int, int] = (2, 16, 16)     # (t, r, theta) patch size
    drop_path: float = 0.0
    grad_checkpointing: bool = False


@dataclass
class MaskCfg:
    """V-JEPA target masks over the (t', r', theta') token grid. The five probabilities sum to 1."""
    p_multiblock: float = 0.7       # spatial tubes: union of random blocks (V-JEPA default)
    p_sector: float = 0.2           # spatial tubes: one angular sector, full radius
    p_band: float = 0.1             # spatial tubes: one radial band, full turn
    p_future: float = 0.0           # temporal: hide the last frames (extrapolate along z)
    p_slab: float = 0.0             # temporal: hide an interior slab (interpolate along z)
    n_short: int = 8
    short_scale: tuple[float, float] = (0.1, 0.2)
    n_long: int = 2
    long_scale: tuple[float, float] = (0.6, 0.8)
    aspect: tuple[float, float] = (0.75, 1.5)
    sector_deg: tuple[float, float] = (60.0, 120.0)
    band_frac: tuple[float, float] = (0.25, 0.5)
    future_frac: tuple[float, float] = (0.25, 0.5)   # fraction of the time axis hidden
    slab_frac: tuple[float, float] = (0.25, 0.5)
    min_context_frac: float = 0.15   # spatial strategies keep at least this many tokens visible


@dataclass
class PretrainCfg:
    """V-JEPA pretraining (`task: pretrain`): clips are `frames.window` frames long."""
    theta_crop: int = 384           # angular window fed to the model (random roll + crop of the 704 bins)
    r_jitter: int = 8               # radial shift augmentation, px
    flip_p: float = 0.5             # theta mirror probability
    time_reverse_p: float = 0.5
    brightness: float = 0.1
    contrast: float = 0.1
    train_split: str = "train,unlabelled"   # pretraining anchors; `test` is never read
    val_split: str = "val"                  # label-free feature bank for the health metrics
    pred_dim: int = 192             # predictor width
    pred_depth: int = 6
    pred_heads: int = 6
    mask: MaskCfg = field(default_factory=MaskCfg)
    # iBOT-style masked self-distillation at the token level, next to the L1 latent loss: the predictor's
    # output at every masked token is matched, as a distribution over `ibot_prototypes` visual words,
    # to the EMA target encoder's token at the same position (shared DINO head, EMA teacher copy,
    # EMA-centred and sharpened teacher). Gives the encoder a per-token target the L1 term does not.
    ibot_weight: float = 0.0        # loss = l1 + ibot_weight * ibot; 0 = off (no head is built)
    ibot_prototypes: int = 4096     # K output prototypes of the DINO head
    ibot_teacher_temp: float = 0.07 # softmax temperature of the centred teacher logits
    ibot_student_temp: float = 0.1  # softmax temperature of the student logits
    ibot_center_momentum: float = 0.9   # EMA momentum of the teacher centre
    # health metrics computed every train.eval_every steps on `val_split`
    bank_clips: int = 2000          # clips in the frozen-feature bank
    knn_k: int = 5
    collapse_every: int = 100       # steps between cheap batch-statistics checks
    feature_std_min: float = 1e-3   # warn below this target-feature std (collapse)


@dataclass
class SegCfg:
    """Supervised segmentation (`task: seg`): ViT-S encoder + UNETR-style 2d decoder on the centre frame."""
    arch: str = "vit"                     # vit | dynunet (nnU-Net-style 2d DynUNet on the window frames
                                          # stacked as channels; needs an odd frames.window)
    dynunet_filters: tuple[int, ...] = (32, 64, 128, 256, 320, 320)   # dynunet: channels per level,
                                          # stride 2 between levels (nnU-Net's 2d plan)
    context_ckpt: str = ""                # dynunet only: "" = off; else a pretraining checkpoint whose EMA
                                          # encoder runs *frozen* on the window as a context branch, its
                                          # token map projected to `context_channels` planes and concatenated
                                          # to the input frames (models/context.py). The fair test of H3.
    context_channels: int = 8             # planes the frozen context branch adds to the DynUNet input
    head: str = "flat"                    # flat | factorised. factorised: the head emits region logits
                                          # (non-wall classes + Wall, every pixel) and type logits (the
                                          # `wall_classes`, wall pixels only) and eval composes
                                          # p(class) = p(region) * p(type | wall) (models/factorised.py)
    wall_classes: tuple[str, ...] = ("Intima", "Lipid", "Calcium", "Media", "Thrombus", "Plaque_rupture",
                                     "Healed_plaque", "Neovascularization")   # factorised: the type group
    pretrained_ckpt: str = ""             # "" = from scratch; else a pretraining checkpoint path
    init_ckpt: str = ""                   # "" = off; else a *seg* checkpoint whose whole SegViT
                                          # (encoder + decoder) initialises this run. Its optimiser,
                                          # step and RNG are not restored: stage B of a two-stage
                                          # recipe starts a fresh schedule on the stage-A weights.
    freeze_encoder: bool = False          # linear-probe style: train the decoder only
    encoder_lr_scale: float = 1.0         # encoder lr multiplier (~0.1 when pretrained)
    llrd: float = 1.0                     # layer-wise lr decay over the encoder blocks; 1.0 = off
    taps: tuple[int, ...] = (3, 6, 9, 12)  # 1-based encoder blocks feeding the decoder, shallow -> deep
    feature_size: int = 16                # decoder base channels
    norm_name: str = "instance"
    temporal_pool: str = "dwconv"         # dwconv | mean   (collapse the tubelet time axis)
    augment: bool = True                  # false = no augmentation at all (overfit sanity runs)
    aug_r_jitter: int = 8                 # radial jitter in px (image + label)
    aug_r_scale: float = 0.1              # radial resample factor drawn in [1 - s, 1 + s] (image + label);
                                          # 0 = off. Catheter-to-wall distance varies between pullbacks
    aug_flip_p: float = 0.5               # theta mirror probability
    aug_gamma: float = 0.3                # image gamma exponent drawn in [1 - g, 1 + g] on the 0..1 scale
    aug_speckle: float = 0.1              # image multiplicative speckle noise std; 0 = off
    aug_brightness: float = 0.15
    aug_contrast: float = 0.15
    deep_supervision: bool = True         # aux heads at /2 and /4
    ds_weights: tuple[float, ...] = (1.0, 0.5, 0.25)   # main /1, aux /2, aux /4; normalised to sum 1
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    dice_batch: bool = True               # Dice over the pooled batch instead of one value per sample:
                                          # a rare class absent from a frame no longer scores that frame
                                          # a free 1.0, which on 616 frames is most of its Dice signal
    dice_include_background: bool = False  # drop channel 0 from the *Dice* term only (CE keeps every
                                          # class, weighted as before): background is ~85% of the pixels
                                          # and its Dice is ~1.0 from the first epoch, so including it
                                          # dilutes the gradient the foreground classes get
    class_weight_alpha: float = 0.25      # CE weight w_c ~ px_frac ** -alpha; 0 = uniform
    class_weight_clip: tuple[float, float] = (0.2, 10.0)
    radial_weight_power: float = 0.0      # per-pixel loss weight ~ r ** p (mean 1 over the rows), on the
                                          # CE and the Dice terms: the loss lives on the polar grid, where
                                          # every pixel counts once, but the published Dice is Cartesian,
                                          # where a polar cell's area grows with r (11x from r=32 to 352).
                                          # 1.0 = optimise the Cartesian quantity; 0 = off (polar weights)
    # Supervise every frame of the window instead of the centre one: the decoder runs once per frame
    # (shared weights, frame k off the token slab it was encoded in), and each frame's loss is weighted
    # 1.0 where the store says the annotation is manual and `pseudo_weight` where it was propagated.
    # Costs one decoder pass per frame, so `window` times the decoder compute of the centre-only head.
    supervise_all_frames: bool = False
    pseudo_weight: float = 0.3
    eval_cartesian: bool = True           # also report Dice/IoU mapped back to Cartesian (the number to publish)
    cartesian_from: str = "probs"         # probs | labels: how eval/predict make the Cartesian label map.
                                          # probs = bilinear resampling of the softmax, argmax in Cartesian;
                                          # labels = nearest lookup of the polar argmax (the pre-2026-09-12 path)
    tta_rolls: int = 2                    # eval/predict: average the softmax over this many equally
                                          # spaced theta rolls (2 = the 0 and half-turn pair); 1 = off
    tta_flip: bool = False                # eval/predict: also average over the theta mirror of every roll
    eval_figures: int = 0                 # GT|pred panels per eval into <out_dir>/figs; 0 = off
    # Label-free z-consistency regulariser: penalise |p(z) - p(z+gap)| between the predictions for two
    # nearby frames of an unlabelled pullback. 0 = off (no second loader is built).
    consistency_weight: float = 0.0
    consistency_gap: int = 1              # frames between the two windows' centre frames
    consistency_split: str = "train,unlabelled"   # label-free anchors; `test` is never read
    consistency_batch: int = 4            # samples per consistency step (two forwards each)
    consistency_warmup_frac: float = 0.1  # ramp the weight in linearly over this fraction of steps


@dataclass
class TrainCfg:
    """Optimisation, checkpointing and logging, for both tasks."""
    steps: int = 20_000
    optimizer: str = "adamw"        # adamw | sgd (nnU-Net: momentum 0.99, Nesterov; seg dynunet only)
    schedule: str = "cosine"        # lr after warmup: cosine to final_lr | poly, (1 - t) ** 0.9 (nnU-Net)
    warmup_frac: float = 0.05
    lr: float = 5e-3                # for reference batch 128: effective lr = lr * batch * accum / 128
    final_lr: float = 1e-6
    wd: tuple[float, float] = (0.05, 0.05)   # weight decay, cosine from first to second
    ema: tuple[float, float] = (0.996, 1.0)  # pretraining target-encoder momentum, cosine
    betas: tuple[float, float] = (0.9, 0.95)
    accum: int = 1                  # gradient accumulation steps
    grad_clip: float = 1.0          # 0 = off
    bf16: bool = True
    compile: bool = True
    seed: int = 0
    out_dir: str = "runs/default"
    eval_every: int = 1000          # steps between evals on the val split
    ckpt_every_steps: int = 1000
    ckpt_every_min: float = 20.0
    ckpt_keep: int = 3
    ckpt_permanent_every: int = 25_000
    log_every: int = 25


@dataclass
class Config:
    task: str = "seg"               # pretrain | seg
    store: StoreCfg = field(default_factory=StoreCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    pretrain: PretrainCfg = field(default_factory=PretrainCfg)
    seg: SegCfg = field(default_factory=SegCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_config(path: str | None = None, overrides: list[str] | None = None,
                values: dict | None = None) -> Config:
    """Build a Config from an optional experiment YAML, `a.b=c` command-line overrides, and
    `values`: already-typed dotted keys (the CLI's path flags), applied last."""
    cfg = Config()
    if path:
        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {}, "")
    for ov in overrides or []:
        key, sep, raw = ov.partition("=")
        if not sep:
            raise ValueError(f"override must look like a.b=c, got {ov!r}")
        _set(cfg, key, yaml.safe_load(raw))
    for key, value in (values or {}).items():
        _set(cfg, key, value)
    if cfg.task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {cfg.task!r}")
    return cfg


def _merge(obj, tree: dict, prefix: str) -> None:
    """Recursively copy a YAML tree onto the dataclass tree."""
    for key, value in tree.items():
        full = f"{prefix}{key}"
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {full!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, full + ".")
        else:
            _set(obj, key, value)


def _set(obj, key: str, value) -> None:
    """Set a dotted key, coercing lists to tuples where the default is a tuple."""
    head, _, rest = key.partition(".")
    if not hasattr(obj, head):
        raise KeyError(f"unknown config key: {head!r} (in {key!r})")
    if rest:
        _set(getattr(obj, head), rest, value)
        return
    current = getattr(obj, head)
    if is_dataclass(current):
        raise KeyError(f"{key!r} is a section, not an option")
    if isinstance(current, tuple) and not isinstance(value, tuple):
        value = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    setattr(obj, head, value)
