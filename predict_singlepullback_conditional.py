"""
Run inference with a trained ConditionalINR on a single .npz pullback file.

Usage:
    python predict_singlepullback_conditional.py \
        --model_dir  /path/to/model_folder \
        --npz        /path/to/pullback.npz \
        --checkpoint best|latest \
        [--out       /path/to/output.nii.gz] \
        [--device    cuda] \
        [--overlap   0.5]

--overlap controls the sliding-window stride:
    0.0  (default) → non-overlapping patches, stride = patch_z
    0.5            → 50 % overlap, stride = patch_z // 2
    1.0            → stride = 1 frame (maximum overlap, slowest)

If the model has a dense decoder, a second file <out_stem>_decoder.nii.gz is
saved alongside the INR prediction for direct comparison.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent / "Training_model"))

from config import Config
from model import ConditionalINR, resize_volume
from inference import predict_pullback


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
        global_feat_ch            = seg.get("global_feat_ch", 32),
        encoder_only              = seg.get("encoder_only", False),
        use_local_features        = seg.get("use_local_features", True),
        use_dense_decoder         = seg.get("use_dense_decoder", False),
        dense_decoder_skip_connections = seg.get("dense_decoder_skip_connections", False),
        encoder_depth              = seg.get("encoder_depth", 3),
        inr_activation            = seg.get("inr_activation", "relu"),
        siren_omega_0             = seg.get("siren_omega_0", 30.0),
        dilation_xy               = seg.get("dilation_xy", 1),
        encoder_z_strides         = seg.get("encoder_z_strides", [1, 1, 1]),
        feature_sampling          = seg.get("feature_sampling", "trilinear"),
    )


def find_yaml(model_dir: Path) -> Path:
    yamls = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
    if not yamls:
        raise FileNotFoundError(f"No yaml config found in {model_dir}")
    return yamls[0]


@torch.no_grad()
def predict_pullback_decoder(
    model:  ConditionalINR,
    volume: np.ndarray,        # [D, H, W] float32
    cfg:    Config,
    device: torch.device,
) -> np.ndarray:               # [D, H, W] int64
    """Dense decoder predictions, patch by patch — mirrors predict_pullback."""
    model.eval()
    D, H, W = volume.shape
    pz       = cfg.patch_z
    predictions = np.zeros((D, H, W), dtype=np.int64)

    z_start = 0
    while z_start < D:
        z_end      = min(z_start + pz, D)
        actual_len = z_end - z_start

        patch = volume[z_start:z_end].copy()
        if actual_len < pz:
            pad   = np.broadcast_to(patch[-1:], (pz - actual_len, H, W))
            patch = np.concatenate([patch, pad], axis=0)

        vol_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        if cfg.dense_decoder_skip_connections and cfg.encoder_depth == 5:
            feat_vol, _, layer1, layer2, layer3, layer4 = model.encoder.forward_deep_multiscale(vol_t)
            dec_logits = model.dense_decoder(feat_vol, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        elif cfg.dense_decoder_skip_connections:
            feat_vol, _, layer1, layer2 = model.encoder.forward_multiscale(vol_t)
            dec_logits = model.dense_decoder(feat_vol, layer1=layer1, layer2=layer2)  # [1, C, pz', H', W']
        else:
            feat_vol, _ = model.encoder(vol_t)                      # [1, feat_ch, pz', H/8, W/8]
            dec_logits  = model.dense_decoder(feat_vol)              # [1, C, pz', H', W'] — may be smaller than patch (see DenseDecoder3D note)
        if dec_logits.shape[2:] != patch.shape:
            dec_logits = F.interpolate(
                dec_logits, size=patch.shape, mode="trilinear", align_corners=False,
            )
        pred_patch         = dec_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [pz, H, W]

        predictions[z_start:z_end] = pred_patch[:actual_len]
        z_start += pz

    return predictions


def save_nii(arr: np.ndarray, path: Path, cfg: Config) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.int16))
    img.SetSpacing([cfg.xy_spacing, cfg.xy_spacing, cfg.z_spacing])
    sitk.WriteImage(img, str(path))
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True, help="Folder with checkpoint + yaml config")
    parser.add_argument("--npz",        required=False, default=r'W:\rubenvdw\Dataset\3D_Grayscale_Circularmask_Zscorenorm_Dicoms_Segmentations_16012026\NL-RUMC-00013-1-1_circ_gray.npz', help="Input .npz pullback file")
    parser.add_argument("--checkpoint", required=True, choices=["best", "latest"], help="best = best_model.pt, latest = latest.pt")
    parser.add_argument("--out",        default=None,  help="Output .nii.gz path (default: model_dir/prediction_<stem>.nii.gz)")
    parser.add_argument("--device",     default="cuda", help="cuda or cpu")
    parser.add_argument("--overlap",    default=0.5, type=float,
                        help="Patch overlap fraction [0.0–1.0]. "
                             "0.0 = no overlap (default), 0.5 = 50%% overlap, "
                             "1.0 = stride of 1 frame (maximum overlap).")
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
    model = ConditionalINR(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    # --- Load volume ---
    data = np.load(npz_path, allow_pickle=False)
    vol  = data["Volume_input_image"]
    if vol.ndim == 4:
        vol = vol[0]
    volume = vol.astype(np.float32)
    if cfg.resize_to:
        volume = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
    D, H, W = volume.shape
    print(f"Volume shape: D={D}, H={H}, W={W}")

    # --- Base output path ---
    stem     = npz_path.stem
    out_base = Path(args.out) if args.out else model_dir / f"prediction_{stem}.nii.gz"

    # --- Save (resized) input volume ---
    input_path = out_base.parent / (out_base.stem.replace(".nii", "") + "_input.nii.gz")
    input_img  = sitk.GetImageFromArray(volume.astype(np.float32))
    input_img.SetSpacing([cfg.xy_spacing, cfg.xy_spacing, cfg.z_spacing])
    sitk.WriteImage(input_img, str(input_path))
    print(f"Saved → {input_path}")

    # --- INR prediction ---
    print(f"Running INR inference (overlap_frac={args.overlap})...")
    pred_inr = predict_pullback(model, volume, cfg, device, overlap_frac=args.overlap)
    save_nii(pred_inr, out_base, cfg)

    # --- Dense decoder prediction (if available) ---
    if cfg.use_dense_decoder and model.dense_decoder is not None:
        print("Running dense decoder inference...")
        pred_dec  = predict_pullback_decoder(model, volume, cfg, device)
        dec_path  = out_base.parent / (out_base.stem.replace(".nii", "") + "_decoder.nii.gz")
        save_nii(pred_dec, dec_path, cfg)
    else:
        print("No dense decoder in this model — skipping decoder output.")


if __name__ == "__main__":
    main()
