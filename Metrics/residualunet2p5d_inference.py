"""
ResidualUNet2p5D model loading + inference + saving -- plays the same role for
the U-Net baseline that conditional_inr_inference.py plays for ConditionalINR.

Not placed in the external CARA_Pipelines_Segmentationmodels repo like
conditional_inr_inference.py -- the U-Net is an internal architecture-comparison
experiment (see [[project_residualunet2p5d_baseline]]), not a deployed model, so
it stays inside Conditional_INR/Metrics alongside the script that uses it.

Unlike ConditionalINR, ResidualUNet2p5D has no coordinate query / native-resolution
decoupling -- predictions always come out at exactly the encoder's own input
resolution (cfg.resize_to, or native if resize_to=0). So there is no
eval_resolution="native" vs "encoder" distinction here: there is only one
resolution this model can ever predict at, which Pullback_prediction_unet.py's
ground truth must be downsampled to match (same as ConditionalINR's
eval_resolution="encoder" mode).
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
from model import ResidualUNet2p5D                        # Training_model/model.py
from inference import predict_pullback_unet                # Training_model/inference.py
from predict_singlepullback_conditional import find_yaml   # Conditional_INR/ root

# postprocessing_conditionalinr.py lives in the external pipeline repo (dual local/cluster
# path, same pattern as Pullback_prediction.py's own PIPELINE_DIR resolution) -- not in this
# Metrics/ folder, despite the module name's similarity to this file.
_PIPELINE_DIR_LOCAL   = "W:/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
_PIPELINE_DIR_CLUSTER = "/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
sys.path.insert(0, _PIPELINE_DIR_LOCAL if Path(_PIPELINE_DIR_LOCAL).is_dir() else _PIPELINE_DIR_CLUSTER)
from postprocessing_conditionalinr import pixel_postprocess


def build_unet_config_from_yaml(yml: dict) -> Config:
    """Mirrors main.py's build_config, restricted to the fields train_unet.py /
    ResidualUNet2p5D actually read (see UNet2p5D/Config/*.yaml's own comment on
    this) -- INR-only keys (inr_hidden/depth, feature_sampling, ...) are never
    read by this model, so they're simply absent here, not defaulted around."""
    seg = yml["model_seg_decoder"]
    tr  = yml["training_segmentation"]
    return Config(
        training_mode             = yml.get("training_mode", "3D"),
        num_classes               = seg["out_channels"],
        encoder_feat               = seg["feature_size"],
        encoder_depth              = seg.get("encoder_depth", 5),
        global_feat_ch              = seg.get("global_feat_ch", 64),
        dilation_xy                = seg.get("dilation_xy", 1),
        encoder_type                = seg.get("encoder_type", "2.5d"),
        dense_decoder_skip_connections = seg.get("dense_decoder_skip_connections", False),
        patch_z                    = tr.get("patch_z", 16),
        resize_to                  = tr.get("resize_to", 0),
        n_points                   = tr.get("n_points", 8192),
        ignore_index               = tr.get("ignore_index", 255),
        batch_size                  = yml.get("train_batch_size", 1),
    )


def load_residualunet2p5d_model(
    model_dir:  str,
    checkpoint: str = "best",
    device_str: str = "cuda",
):
    """Load yaml -> Config, instantiate ResidualUNet2p5D, load best_model.pt/latest.pt.
    Mirrors conditional_inr_inference.load_conditional_inr_model exactly, just a
    different model class and config builder."""
    model_dir = Path(model_dir)
    yaml_path = find_yaml(model_dir)
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    cfg = build_unet_config_from_yaml(yml)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt_path = model_dir / ("best_model.pt" if checkpoint == "best" else "latest.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = ResidualUNet2p5D(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[ResidualUNet2p5D] Loaded {ckpt_path.name} (epoch {ckpt.get('epoch', '?')}) "
          f"on {device}, config={yaml_path.name}")
    return model, cfg, device


@torch.no_grad()
def predict_residualunet2p5d(
    volume:            np.ndarray,   # [D, native_xy_size, native_xy_size] float32, z-score normalised
    model:              ResidualUNet2p5D,
    cfg:                Config,
    device:             torch.device,
    output_file_path:   str,
    postprocess_classes:     Optional[Mapping[int, Tuple[int, int]]] = None,
    postprocess_output_path: Optional[str] = None,
    postprocess_n_procs:     Optional[int] = 4,
) -> None:
    """Resize to cfg.resize_to (if set), run predict_pullback_unet, save as .nii.gz
    at the correct spacing for THAT grid (there's no native-resolution query to
    opt back into -- see this module's docstring). Caller must downsample ground
    truth to the same grid before scoring (model.resize_labels), exactly like
    ConditionalINR's eval_resolution="encoder" mode."""
    from model import resize_volume   # Training_model/model.py

    volume = volume.astype(np.float32, copy=False)
    if cfg.resize_to:
        volume_infer = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
    else:
        volume_infer = volume

    print(f"[ResidualUNet2p5D] Inference: D={volume_infer.shape[0]} "
          f"H={volume_infer.shape[1]} W={volume_infer.shape[2]} patch_z={cfg.patch_z}")
    pred = predict_pullback_unet(model, volume_infer, cfg, device)   # [D, H, W] int64

    # cfg.xy_spacing is already the correct spacing for volume_infer's own grid --
    # Config.__post_init__ rescales it from native_xy_size up whenever resize_to is
    # set, same as ConditionalINR's eval_resolution="encoder" branch.
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
