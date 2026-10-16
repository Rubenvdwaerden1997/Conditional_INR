"""
ResidualUNet3D model loading + inference + saving -- plays the same role for
the 3D U-Net baseline that residualunet2p5d_inference.py plays for
ResidualUNet2p5D (and conditional_inr_inference.py for ConditionalINR).

Not placed in the external CARA_Pipelines_Segmentationmodels repo -- the 3D
U-Net is an internal architecture-comparison experiment, not a deployed
model, so it stays inside Conditional_INR/Metrics.

Unlike ResidualUNet2p5D (one annotated frame per forward pass, no overlap
concept), ResidualUNet3D predicts a full [C, patch_z, H, W] volume per
forward pass, so inference uses the same sliding-window overlap/accumulate
scheme as ConditionalINR's predict_pullback (via inference.py's
predict_pullback_unet3d) rather than ResidualUNet2p5D's one-frame-at-a-time
loop. Like ResidualUNet2p5D, there's no coordinate query / native-resolution
decoupling -- predictions always come out at exactly the encoder's own input
resolution (cfg.resize_to, or native if resize_to=0).
"""
import sys
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "Training_model"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config                                # Training_model/config.py
from model import ResidualUNet3D                          # Training_model/model.py
from inference import predict_pullback_unet3d              # Training_model/inference.py
from predict_singlepullback_conditional import find_yaml   # Conditional_INR/ root

_PIPELINE_DIR_LOCAL   = "W:/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
_PIPELINE_DIR_CLUSTER = "/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
sys.path.insert(0, _PIPELINE_DIR_LOCAL if Path(_PIPELINE_DIR_LOCAL).is_dir() else _PIPELINE_DIR_CLUSTER)
from postprocessing_conditionalinr import pixel_postprocess


def build_unet3d_config_from_yaml(yml: dict) -> Config:
    """Mirrors residualunet2p5d_inference.build_unet_config_from_yaml, restricted
    to the fields train_unet3d.py / ResidualUNet3D actually read."""
    seg = yml["model_seg_decoder"]
    tr  = yml["training_segmentation"]
    return Config(
        training_mode             = yml.get("training_mode", "3D"),
        num_classes               = seg["out_channels"],
        encoder_feat               = seg["feature_size"],
        encoder_depth              = seg.get("encoder_depth", 5),
        global_feat_ch              = seg.get("global_feat_ch", 64),
        dilation_xy                = seg.get("dilation_xy", 1),
        encoder_z_strides          = seg.get("encoder_z_strides", [1, 1, 1]),
        encoder_type                = seg.get("encoder_type", "3d"),
        dense_decoder_skip_connections = seg.get("dense_decoder_skip_connections", True),
        patch_z                    = tr.get("patch_z", 16),
        resize_to                  = tr.get("resize_to", 0),
        n_points                   = tr.get("n_points", 8192),
        ignore_index               = tr.get("ignore_index", 255),
        batch_size                  = yml.get("train_batch_size", 1),
    )


def load_residualunet3d_model(
    model_dir:  str,
    checkpoint: str = "best",
    device_str: str = "cuda",
):
    """Load yaml -> Config, instantiate ResidualUNet3D, load best_model.pt/latest.pt.
    Mirrors residualunet2p5d_inference.load_residualunet2p5d_model exactly, just
    a different model class and config builder."""
    model_dir = Path(model_dir)
    yaml_path = find_yaml(model_dir)
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    cfg = build_unet3d_config_from_yaml(yml)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt_path = model_dir / ("best_model.pt" if checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ResidualUNet3D(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[ResidualUNet3D] Loaded {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}) "
          f"on {device}, config={yaml_path.name}")
    return model, cfg, device


@torch.no_grad()
def predict_residualunet3d(
    volume:            np.ndarray,   # [D, native_xy_size, native_xy_size] float32, z-score normalised
    model:              ResidualUNet3D,
    cfg:                Config,
    device:             torch.device,
    output_file_path:   str,
    overlap_frac:       float = 0.5,   # sliding-window overlap between patch_z windows (0 = none)
    postprocess_classes:     Optional[Mapping[int, Tuple[int, int]]] = None,
    postprocess_output_path: Optional[str] = None,
    postprocess_n_procs:     Optional[int] = 4,
) -> None:
    """Resize to cfg.resize_to (if set), run predict_pullback_unet3d, save as
    .nii.gz at the correct spacing for THAT grid -- same resolution contract
    as predict_residualunet2p5d (no native-resolution query to opt back into).
    Caller must downsample ground truth to the same grid before scoring."""
    from model import resize_volume   # Training_model/model.py

    volume = volume.astype(np.float32, copy=False)
    if cfg.resize_to:
        volume_infer = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
    else:
        volume_infer = volume

    print(f"[ResidualUNet3D] Inference: D={volume_infer.shape[0]} "
          f"H={volume_infer.shape[1]} W={volume_infer.shape[2]} patch_z={cfg.patch_z} "
          f"overlap_frac={overlap_frac}")
    pred = predict_pullback_unet3d(model, volume_infer, cfg, device, overlap_frac=overlap_frac)   # [D, H, W] int64

    out_xy_spacing = cfg.xy_spacing

    img = sitk.GetImageFromArray(pred.astype(np.int16))
    img.SetSpacing([out_xy_spacing, out_xy_spacing, cfg.z_spacing])
    sitk.WriteImage(img, str(output_file_path))
    print(f"Saved -> {output_file_path}")

    if postprocess_classes is not None:
        cleaned = pixel_postprocess(pred, postprocess_classes, num_classes=cfg.num_classes,
                                     n_procs=postprocess_n_procs)
        post_img = sitk.GetImageFromArray(cleaned.astype(np.int16))
        post_img.SetSpacing([out_xy_spacing, out_xy_spacing, cfg.z_spacing])
        sitk.WriteImage(post_img, str(postprocess_output_path))
        print(f"Saved -> {postprocess_output_path}")
