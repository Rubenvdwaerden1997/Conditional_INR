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
        num_epochs           = tr["epochs"],
        lr                   = tr["lr_settings"]["initial_lr"],
        weight_decay         = tr["lr_settings"]["weight_decay"],
        batch_size           = yml["train_batch_size"],
        val_interval         = tr.get("val_interval", 5),
        warmup_epochs        = tr.get("warmup_epochs", 20),
        warmup_start_factor  = tr.get("warmup_start_factor", 0.1),
        checkpoint_dir       = os.path.join(yml["save_root"], yml["experiment_name"]),
        context_frames       = yml.get("context_frames", 0),
        frames_folder         = yml["paths"][yml["env"]].get("frames_folder", ""),
        patches_folder_3d     = yml["paths"][yml["env"]].get("patches_folder_3d", ""),
        ignore_index                = tr.get("ignore_index", 255),
        feature_supervision         = tr.get("feature_supervision", False),
        feature_supervision_weight  = tr.get("feature_supervision_weight", 0.5),
        global_feat_ch              = seg.get("global_feat_ch", 32),
        encoder_only                = seg.get("encoder_only", False),
        tv_weight                   = tr.get("tv_loss_weight", 0.0),
        patch_z                     = tr.get("patch_z", 32),
    )


def sanity_check(cfg: Config) -> None:
    """Quick forward-pass check with random tensors — no real data needed."""
    print(f"=== Sanity check (random tensors, mode={cfg.training_mode}) ===")
    device = torch.device("cpu")
    model  = ConditionalINR(cfg)

    B, H, W, N = 2, 128, 128, 2048
    if cfg.training_mode == "2D":
        volume = torch.randn(B, 1, H, W)
        coords = torch.rand(B, N, 2) * torch.tensor([W, H], dtype=torch.float32)
    else:
        D = 32
        volume = torch.randn(B, 1, D, H, W)
        coords = torch.rand(B, N, 3) * torch.tensor([W, H, D], dtype=torch.float32)
    labels = torch.randint(0, cfg.num_classes, (B, N))
    labels[0, :100] = cfg.ignore_index

    logits     = model(volume, coords)
    loss, log  = compute_loss(logits, labels, cfg)
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

    folders  = paths["data_root"]   # list in the YAML
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

    if cfg.training_mode == "2D":
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