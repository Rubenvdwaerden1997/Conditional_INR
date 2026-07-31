"""
Run inference with a trained UnconditionalINR on a single .npz pullback file.

Usage:
    python predict_single.py \
        --model_dir /path/to/model_folder \
        --npz       /path/to/pullback.npz \
        [--checkpoint best_model.pt] \
        [--out       /path/to/output.nii.gz] \
        [--device    cuda]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import yaml

# Allow imports from the same Training_model directory
sys.path.insert(0, str(Path(__file__).parent / "Training_model"))

from config import Config
from model import UnconditionalINR
from inference import predict_pullback_unconditional


def build_config_from_yaml(yml: dict) -> Config:
    seg = yml["model_seg_decoder"]
    tr  = yml["training_segmentation"]
    return Config(
        training_mode             = yml.get("training_mode", "3D"),
        num_classes               = seg["out_channels"],
        encoder_feat              = seg["feature_size"],
        num_epochs                = tr["epochs"],
        lr                        = tr["lr_settings"]["initial_lr"],
        weight_decay              = tr["lr_settings"]["weight_decay"],
        batch_size                = yml.get("train_batch_size", 1),
        ignore_index              = tr.get("ignore_index", 255),
        inr_hidden                = seg.get("inr_hidden", 512),
        inr_depth                 = seg.get("inr_depth", 4),
        num_freqs_xy              = seg.get("num_freqs_xy", 6),
        use_intermediate_features = seg.get("use_intermediate_features", False),
        patch_z                   = tr.get("patch_z", 32),
        resize_to                 = tr.get("resize_to", 0),
        n_points                  = tr.get("n_points", 8192),
        unconditional             = yml.get("unconditional", True),
        unconditional_pullback    = yml.get("unconditional_pullback", ""),
        unconditional_n_repeats   = yml.get("unconditional_n_repeats", 100),
    )


def find_yaml(model_dir: Path) -> Path:
    yamls = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    if not yamls:
        raise FileNotFoundError(f"No yaml config found in {model_dir}")
    return yamls[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True,  help="Folder with checkpoint + yaml config")
    parser.add_argument("--npz",        required=True,  help="Input .npz pullback file")
    parser.add_argument("--checkpoint", required=True, choices=["best", "latest"], help="Which checkpoint to use: best = best_model.pt, latest = latest.pt")
    parser.add_argument("--out",        default=None,   help="Output .nii.gz path (default: model_dir/prediction_<stem>.nii.gz)")
    parser.add_argument("--device",     default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    npz_path  = Path(args.npz)
    device    = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Config ---
    yaml_path = find_yaml(model_dir)
    print(f"Config : {yaml_path.name}")
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    cfg = build_config_from_yaml(yml)

    # --- Checkpoint ---
    ckpt_path = model_dir / ("best_model.pt" if args.checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path.name}")

    # --- Model ---
    model = UnconditionalINR(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    # --- Volume shape from npz ---
    data = np.load(npz_path, allow_pickle=False)
    vol  = data["Volume_input_image"]
    if vol.ndim == 4:
        vol = vol[0]
    D, H, W = vol.shape
    if cfg.resize_to:
        H = W = cfg.resize_to   # pixel data is never used by the unconditional model — only shape matters
    volume_shape = (D, H, W)
    print(f"Volume shape: D={D}, H={H}, W={W}")

    # --- Predict ---
    print("Running inference...")
    pred_vol = predict_pullback_unconditional(model, volume_shape, cfg, device)

    # --- Save ---
    # pred_vol is (D, H, W) = (z, y, x) — the order SimpleITK expects, no transpose needed
    img      = sitk.GetImageFromArray(pred_vol.astype(np.int16))
    img.SetSpacing([cfg.xy_spacing, cfg.xy_spacing, cfg.z_spacing])  # (x, y, z)
    out_path = Path(args.out) if args.out else model_dir / f"prediction_{npz_path.stem}.nii.gz"
    sitk.WriteImage(img, str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
