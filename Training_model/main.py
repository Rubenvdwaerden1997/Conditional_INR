"""
main.py — entry point for training the Conditional INR.

Usage:
    python main.py --config Config/config.yaml
    python main.py --config Config/config.yaml --sanity_check
    python main.py --config Config/config.yaml --data_check path/to/file.npz
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from config import Config
from dataset import OCTPullbackDataset
from losses import compute_loss
from metrics import compute_dice
from model import ConditionalINR
from trainer import train
from utils import setup_output_dir


def build_config(yml: dict) -> Config:
    """Map YAML keys to the Config dataclass."""
    seg = yml["model_seg_decoder"]
    tr  = yml["training_segmentation"]
    return Config(
        training_mode        = yml.get("training_mode", "3D"),
        num_classes          = seg["out_channels"],
        encoder_feat         = seg["feature_size"],
        use_local_features   = seg.get("use_local_features", True),
        use_dense_decoder    = seg.get("use_dense_decoder", False),
        dense_decoder_weight = tr.get("dense_decoder_weight", 1.0),
        dense_decoder_skip_connections = seg.get("dense_decoder_skip_connections", False),
        encoder_depth        = seg.get("encoder_depth", 3),
        num_epochs           = tr["epochs"],
        lr                   = tr["lr_settings"]["initial_lr"],
        weight_decay         = tr["lr_settings"]["weight_decay"],
        batch_size           = yml["train_batch_size"],
        val_interval         = tr.get("val_interval", 5),
        warmup_epochs        = tr.get("warmup_epochs", 20),
        warmup_start_factor  = tr.get("warmup_start_factor", 0.1),
        checkpoint_dir       = os.path.join(yml["save_root"].format(**yml), yml["experiment_name"]),
        context_frames       = yml.get("context_frames", 0),
        frames_folder         = yml["paths"][yml["env"]].get("frames_folder", ""),
        patches_folder_3d     = yml["paths"][yml["env"]].get("patches_folder_3d", ""),
        ignore_index                = tr.get("ignore_index", 255),
        ignore_background           = tr.get("ignore_background", False),
        feature_supervision         = tr.get("feature_supervision", False),
        feature_supervision_weight  = tr.get("feature_supervision_weight", 0.5),
        global_feat_ch              = seg.get("global_feat_ch", 64),
        dilation_xy                 = seg.get("dilation_xy", 1),
        encoder_only                = seg.get("encoder_only", False),
        use_intermediate_features   = seg.get("use_intermediate_features", False),
        inr_hidden                  = seg.get("inr_hidden", 512),
        inr_depth                   = seg.get("inr_depth", 4),
        tv_weight                   = tr.get("tv_loss_weight", 0.0),
        patch_z                     = tr.get("patch_z", 32),
        resize_to                   = tr.get("resize_to", 0),
        sampling_strategy           = tr.get("sampling_strategy", "random"),
        smoothness_3d_weight          = tr.get("smoothness_3d_weight", 0.0),
        smoothness_class_indices      = tr.get("smoothness_class_indices", []),
        smoothness_2d_weight          = tr.get("smoothness_2d_weight", 0.0),
        smoothness_2d_n_pairs         = tr.get("smoothness_2d_n_pairs", 64),
        radial_media_prior_weight     = tr.get("radial_media_prior_weight", 0.0),
        radial_guidewire_prior_weight = tr.get("radial_guidewire_prior_weight", 0.0),
        radial_prior_n_rays           = tr.get("radial_prior_n_rays", 8),
        radial_prior_n_radii          = tr.get("radial_prior_n_radii", 12),
        background_floor_frac       = tr.get("background_floor_frac", 0.0),
        coord_jitter_max            = tr.get("coord_jitter_max", 0.0),
        n_points                    = tr.get("n_points", 8192),
        num_freqs_xy                = seg.get("num_freqs_xy", 6),
        num_freqs_z                 = seg.get("num_freqs_z", 4),
        inr_activation              = seg.get("inr_activation", "relu"),
        siren_omega_0               = seg.get("siren_omega_0", 30.0),
        preload_data                = tr.get("preload_data", False),
        preload_frac                = tr.get("preload_frac", 0.5),
        use_amp                     = tr.get("use_amp", False),
        grad_clip_norm              = tr.get("grad_clip_norm", 0.0),
        unconditional               = yml.get("unconditional", False),
        unconditional_pullback      = yml.get("unconditional_pullback", ""),
        unconditional_n_repeats     = yml.get("unconditional_n_repeats", 100),
        loss_type                   = tr.get("loss_type", "mse_onehot"),
        tversky_alpha               = tr.get("tversky_alpha", 0.5),
        tversky_beta                = tr.get("tversky_beta", 0.5),
        tversky_class_alpha         = tr.get("tversky_class_alpha", {}),
        tversky_class_beta          = tr.get("tversky_class_beta", {}),
        encoder_z_strides           = seg.get("encoder_z_strides", [1, 1, 1]),
        feature_sampling            = seg.get("feature_sampling", "trilinear"),
        max_batches_frac            = tr.get("max_batches_frac", 0.0),
        val_interval_pct            = tr.get("val_interval_pct", 0.0),
    )


def sanity_check(cfg: Config) -> None:
    """Quick forward-pass check with random tensors — no real data needed."""
    from model import UnconditionalINR
    print(f"=== Sanity check (random tensors, mode={cfg.training_mode}, unconditional={cfg.unconditional}) ===")
    device = torch.device("cpu")

    B, H, W, N = 2, 128, 128, 2048
    labels = torch.randint(0, cfg.num_classes, (B, N))
    labels[0, :100] = cfg.ignore_index

    if cfg.unconditional:
        D      = 32
        model  = UnconditionalINR(cfg)
        coords = torch.rand(B, N, 3) * torch.tensor([W, H, D], dtype=torch.float32)
        logits = model(coords, (D, H, W))
    elif cfg.training_mode == "2D":
        model  = ConditionalINR(cfg)
        volume = torch.randn(B, 1, H, W)
        coords = torch.rand(B, N, 2) * torch.tensor([W, H], dtype=torch.float32)
        logits = model(volume, coords)
    else:
        D      = 32
        model  = ConditionalINR(cfg)
        volume = torch.randn(B, 1, D, H, W)
        coords = torch.rand(B, N, 3) * torch.tensor([W, H, D], dtype=torch.float32)
        logits = model(volume, coords)

    loss, log    = compute_loss(logits, labels, cfg)
    mean_dice, _ = compute_dice(logits, labels, cfg.num_classes, cfg.ignore_index)

    print(f"  Logits shape     : {logits.shape}")
    print(f"  Loss             : {loss.item():.4f}  {log}")
    print(f"  Mean Dice (random): {mean_dice:.4f}")
    print("Sanity check passed.")


def data_check(npz_path: str, cfg: Config) -> None:
    """Load one real .npz file, run a forward pass, and print shapes + stats."""
    import numpy as np
    from pathlib import Path

    print(f"\n=== Data check: {Path(npz_path).name} ===\n")

    # --- 1. Raw file inspection ---
    data = np.load(npz_path)
    print("Keys in file:")
    for k in data.keys():
        arr = data[k]
        print(f"  {k}: shape={arr.shape}  dtype={arr.dtype}  "
              f"min={float(arr.min()):.4f}  max={float(arr.max()):.4f}")

    # --- 2. Build fake dataset entry/entries ---
    vol_raw = data["Volume_input_image"]
    if vol_raw.ndim == 4:
        vol_raw = vol_raw[0]
    D, H, W = vol_raw.shape

    annotated = list(range(0, D, 50))   # treat every 50th frame as labeled
    print(f"\nVolume shape (D, H, W): ({D}, {H}, {W})")
    print(f"Annotated frames used for check: {annotated}")

    seg_raw = data["Segmentation_image"]
    if seg_raw.ndim == 4:
        seg_raw = seg_raw[0]
    unique_labels = sorted(set(seg_raw[annotated].flatten().tolist()))
    print(f"Unique label values in annotated frames: {unique_labels}")
    if max(unique_labels) >= cfg.num_classes:
        print(f"  WARNING: max label {max(unique_labels)} >= num_classes {cfg.num_classes}. "
              f"Enable label mapping or increase num_classes.")

    if cfg.training_mode == "2D":
        # One entry per annotated frame; use the first one for the check
        entries = [
            {"file": npz_path, "pullback": Path(npz_path).stem, "frame_idx": z}
            for z in annotated
        ]
    else:
        entries = [{
            "file":             npz_path,
            "pullback":         Path(npz_path).stem,
            "annotated_frames": annotated,
        }]

    # --- 3. Dataset __getitem__ ---
    ds = OCTPullbackDataset(
        entries=entries, cfg=cfg, n_points=cfg.n_points,
        mode="val", augment=False,
    )
    vol_t, coords_t, labels_t, _ = ds[0]
    print(f"\nDataset __getitem__ output (mode={cfg.training_mode}):")
    print(f"  volume tensor : {tuple(vol_t.shape)}  "
          f"mean={vol_t.mean():.3f}  std={vol_t.std():.3f}  "
          f"min={vol_t.min():.3f}  max={vol_t.max():.3f}")
    coord_info = f"range x=[{coords_t[:,0].min():.1f}, {coords_t[:,0].max():.1f}]"
    if cfg.training_mode == "3D":
        coord_info += f"  z=[{coords_t[:,2].min():.1f}, {coords_t[:,2].max():.1f}]"
    print(f"  coords tensor : {tuple(coords_t.shape)}  {coord_info}")
    print(f"  labels tensor : {tuple(labels_t.shape)}  "
          f"labeled={( labels_t != cfg.ignore_index).sum().item()}  "
          f"ignored={(labels_t == cfg.ignore_index).sum().item()}")

    # # --- 4. Forward pass (CPU, small crop to keep memory reasonable) ---
    # print("\nRunning forward pass on CPU (this may take ~30 s for 704x704) …")
    # device = torch.device("cpu")
    # model  = ConditionalINR(cfg)
    # model.eval()

    # with torch.no_grad():
    #     vol_b    = vol_t.unsqueeze(0)                   # [1, 1, D, H, W]
    #     coords_b = coords_t.unsqueeze(0)                # [1, N, 3]
    #     labels_b = labels_t.unsqueeze(0)                # [1, N]

    #     logits   = model(vol_b, coords_b)               # [1, N, C]
    #     loss, log = compute_loss(logits, labels_b, cfg)
    #     miou, per_class = compute_miou(
    #         logits, labels_b, cfg.num_classes, cfg.ignore_index
    #     )

    # print(f"  logits shape : {tuple(logits.shape)}")
    # print(f"  loss         : {loss.item():.4f}  {log}")
    # print(f"  mIoU         : {miou:.4f}")
    # present = [(i, f"{v:.3f}") for i, v in enumerate(per_class) if v == v]  # skip nan
    # print(f"  per-class IoU (present classes): {present}")
    # print("\nData check passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Train Conditional INR for IV-OCT 3D segmentation"
    )
    parser.add_argument(
        "--config", type=str, default="Config/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--sanity_check", action="store_true",
        help="Run a quick forward-pass check with random tensors and exit",
    )
    parser.add_argument(
        "--data_check", type=str, default=None, metavar="NPZ_FILE",
        help="Load a real .npz file, verify shapes, and run a forward pass",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yml = yaml.safe_load(f)

    cfg = build_config(yml)

    if args.sanity_check:
        sanity_check(cfg)
        return

    if args.data_check:
        data_check(args.data_check, cfg)
        return

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    seed = yml.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    env   = yml["env"]
    paths = yml["paths"][env]
    seg   = yml["model_seg_decoder"]

    df_split = pd.read_excel(paths["split_excel"])
    df_split.columns = [c.strip().lower() for c in df_split.columns]
    for col in ("pullback", "set", "frames"):
        assert col in df_split.columns, f"Excel must contain '{col}' column"

    mapping_activated = bool(seg.get("mapping_activated", True))
    label_mapping = (
        {int(k): int(v) for k, v in seg["mapping"].items()}
        if mapping_activated and seg.get("mapping")
        else None
    )

    dataset_kwargs = dict(
        n_points          = cfg.n_points,
        mapping_activated = mapping_activated,
        label_mapping     = label_mapping,
    )

    folders = paths["data_root"]   # list in the YAML

    if cfg.unconditional:
        # Find the target pullback and its annotated frames from the Excel
        df_train = df_split[df_split["set"].str.lower() == "training"]
        if cfg.unconditional_pullback:
            row = df_train[df_train["pullback"].astype(str) == cfg.unconditional_pullback]
            assert len(row) == 1, (
                f"unconditional_pullback '{cfg.unconditional_pullback}' not found in training set"
            )
            target_pid = cfg.unconditional_pullback
        else:
            row        = df_train.iloc[[0]]
            target_pid = str(row.iloc[0]["pullback"])

        ann_frames = [
            int(f.strip()) - 1
            for f in str(row.iloc[0]["frames"]).split(",")
            if f.strip().isdigit()
        ]

        from pathlib import Path as _Path
        npz_path = None
        for folder in folders:
            for npz in _Path(folder).glob("*.npz"):
                pid = npz.stem.replace("_circ_gray", "")
                if pid == target_pid:
                    npz_path = str(npz)
                    break
            if npz_path:
                break
        assert npz_path, f"Could not find .npz for pullback '{target_pid}' in {folders}"

        print(
            f"Unconditional mode — overfitting pullback '{target_pid}'  "
            f"({len(ann_frames)} annotated frames: {ann_frames})"
        )
        train_ds = OCTPullbackDataset.from_single_volume(
            npz_path, ann_frames, cfg, mode="train", **dataset_kwargs,
        )
        val_ds = OCTPullbackDataset.from_single_volume(
            npz_path, ann_frames, cfg, mode="val", **dataset_kwargs,
        )
    else:
        train_ds = OCTPullbackDataset.from_excel(
            folders, df_split, cfg, set_excel="training",
            mode="train", augment=True, **dataset_kwargs,
        )
        val_ds = OCTPullbackDataset.from_excel(
            folders, df_split, cfg, set_excel="validation",
            mode="val", augment=False, **dataset_kwargs,
        )

    # -----------------------------------------------------------------------
    # Output dir + logger
    # -----------------------------------------------------------------------
    logger = setup_output_dir(cfg, config_path=args.config)
    logger.info(f"Experiment    : {yml['experiment_name']}")
    logger.info(f"Training mode : {cfg.training_mode}")
    logger.info(f"Config        : {args.config}")
    logger.info(f"Split         : {paths['split_excel']}")
    resize_desc = f"{cfg.resize_to}x{cfg.resize_to}" if cfg.resize_to else f"native {cfg.native_xy_size}x{cfg.native_xy_size}"
    logger.info(f"Resolution    : {resize_desc}  (xy_spacing={cfg.xy_spacing:.5f} mm/px)")

    if cfg.unconditional:
        logger.info(f"Input volume shape (D,H,W): {train_ds._unc_volume_shape}")
        logger.info(
            f"Unconditional dataset — train steps: {len(train_ds)} | val: 1"
        )
    elif cfg.training_mode == "2D":
        logger.info(
            f"Samples (frames) — train: {len(train_ds)} | val: {len(val_ds)}"
        )
    else:
        if train_ds.entries and train_ds.entries[0].get("prebuilt_3d"):
            total_ann_train = len(train_ds.entries)
            total_ann_val   = len(val_ds.entries)
            logger.info(
                f"Annotated frames (prebuilt patches) — "
                f"train: {total_ann_train} | val: {total_ann_val}"
            )
        else:
            total_ann_train = sum(len(e["annotated_frames"]) for e in train_ds.entries)
            total_ann_val   = sum(len(e["annotated_frames"]) for e in val_ds.entries)
            logger.info(
                f"Annotated frames — "
                f"train: {total_ann_train} across {len(train_ds)} pullbacks | "
                f"val: {total_ann_val} across {len(val_ds)} pullbacks"
            )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    device = yml.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    train(train_ds, val_ds, cfg=cfg, device=device, logger=logger,
          config_path=args.config, label_mapping=label_mapping)


if __name__ == "__main__":
    main()