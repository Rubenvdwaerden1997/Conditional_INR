"""Standalone smoke check for OCTPullbackDataset.from_zarr() — no training, no pytest.

Usage:
    python check_zarr_dataset.py --config Config/Conditional_foregroundnorm/config_3D_conditional_cedice_encoder64_depth5_nodense_relu_zarrstore.yaml
"""

import argparse

import torch
import yaml

from config import Config
from dataset import OCTPullbackDataset
from main import build_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        yml = yaml.safe_load(f)
    cfg: Config = build_config(yml)
    assert cfg.zarr_store, f"{args.config} has no paths.<env>.zarr_store set"

    seg = yml["model_seg_decoder"]
    mapping_activated = bool(seg.get("mapping_activated", True))
    label_mapping = (
        {int(k): int(v) for k, v in seg["mapping"].items()}
        if mapping_activated and seg.get("mapping") else None
    )

    print(f"=== Zarr dataset check: {cfg.zarr_store} ===")
    ds = OCTPullbackDataset.from_zarr(
        cfg.zarr_store, cfg, set_excel="training", mode="train", augment=False,
        n_points=cfg.n_points, mapping_activated=mapping_activated, label_mapping=label_mapping,
    )
    print(f"{len(ds)} pullbacks in the training split.")
    assert len(ds) > 0, "from_zarr() found 0 pullbacks — check store path / split values"

    for i in range(min(args.n_samples, len(ds))):
        volume, coords, point_labels, labels = ds[i]

        assert volume.shape[0] == 1 and volume.shape[1] == cfg.patch_z, \
            f"unexpected volume shape {tuple(volume.shape)}"
        assert labels.shape[0] == cfg.patch_z, f"unexpected labels shape {tuple(labels.shape)}"

        labeled = labels[labels != cfg.ignore_index]
        if labeled.numel():
            assert labeled.min() >= 0 and labeled.max() < cfg.num_classes, \
                f"label value out of [0, {cfg.num_classes}) range: " \
                f"min={labeled.min().item()} max={labeled.max().item()}"

        print(f"[{i}] pullback={ds.entries[i]['pullback']:20s} "
              f"volume={tuple(volume.shape)} mean={volume.mean():.3f} std={volume.std():.3f} "
              f"labels={tuple(labels.shape)} "
              f"labeled_px={int((labels != cfg.ignore_index).sum())} "
              f"ignored_px={int((labels == cfg.ignore_index).sum())} "
              f"coords={tuple(coords.shape)}")

    print("Zarr dataset check passed.")


if __name__ == "__main__":
    main()
