"""
train_unet.py — standalone training entry point for the 2.5D Residual U-Net baseline.

Separate from main.py/trainer.py on purpose: ResidualUNet2p5D predicts a dense
[num_classes, H, W] map directly (no coordinate sampling), a different output
contract from ConditionalINR's point-based decode_3d path. Reuses the same
Config/OCTPullbackDataset/zarr-or-npz data loading as main.py unchanged, so a
config that only differs from an INR config in model_seg_decoder.encoder_type
(and drops the INR-only keys) gives a true model-only comparison.

Usage:
    python train_unet.py --config UNet2p5D/Config/config_3D_residualunet2p5d_encoder64_depth5_zarrstore.yaml
    python train_unet.py --config ... --sanity_check
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import Config
from dataset import OCTPullbackDataset
from losses import dice_loss
from main import build_config
from model import ResidualUNet2p5D
from trainer import _load_latest_checkpoint, _save_latest_checkpoint
from utils import setup_output_dir


def sanity_check(cfg: Config) -> None:
    """Quick forward-pass check with random tensors — no real data needed."""
    print(f"=== U-Net sanity check (random tensors, encoder_type={cfg.encoder_type}) ===")
    device = torch.device("cpu")
    model = ResidualUNet2p5D(cfg).to(device)

    B, H, W = 2, 128, 128
    volume = torch.randn(B, 1, cfg.patch_z, H, W)
    labels = torch.randint(0, cfg.num_classes, (B, H, W))
    labels[0, :10, :10] = cfg.ignore_index

    logits = model(volume)
    assert logits.shape == (B, cfg.num_classes, H, W), \
        f"expected logits shape {(B, cfg.num_classes, H, W)}, got {tuple(logits.shape)}"

    loss, log = _dense_loss(logits, labels, cfg)
    print(f"  Logits shape : {tuple(logits.shape)}")
    print(f"  Loss         : {loss.item():.4f}  {log}")
    print("U-Net sanity check passed.")


def _extract_annotated_label(labels: torch.Tensor, cfg: Config) -> torch.Tensor:
    """labels: [B, patch_z, H, W] -> [B, H, W], the one annotated frame's labels per sample.

    NOT simply labels[:, patch_z // 2] -- dataset.py's zarr-backed loader now randomly places
    the annotated frame anywhere within the window during training (z-position augmentation,
    matching the .npz pipeline's prebuilt_3d behaviour), so its index varies per sample, not
    fixed at the center. Finds the actual non-ignore_index frame per sample instead. If a
    window has no annotated frame (shouldn't happen given how entries/windows are built, but
    not asserted against), falls back to index 0 -- an all-ignore_index slice there, so the
    loss for that sample is simply zero (same self-handling behaviour as elsewhere), not a crash.
    """
    valid = (labels != cfg.ignore_index).any(dim=-1).any(dim=-1)   # [B, patch_z] bool
    z_idx = valid.long().argmax(dim=1)                              # [B] -- index of first True
    return labels[torch.arange(labels.shape[0], device=labels.device), z_idx]


def _dense_loss(logits2d: torch.Tensor, center_label: torch.Tensor, cfg: Config):
    """logits2d: [B, C, H, W]; center_label: [B, H, W]. Same CE+Dice formula as
    trainer.py's dense-decoder auxiliary loss (trainer.py:589-615), just 2D."""
    B_, C_, H_, W_ = logits2d.shape
    flat_logits = logits2d.permute(0, 2, 3, 1).reshape(-1, C_)   # [B*H*W, C]
    flat_labels = center_label.reshape(-1)                        # [B*H*W]
    ce   = F.cross_entropy(flat_logits, flat_labels, ignore_index=cfg.ignore_index)
    dice = dice_loss(flat_logits, flat_labels, C_, cfg.ignore_index, cfg.ignore_background)
    loss = ce + cfg.dice_weight * dice
    return loss, {"ce": ce.item(), "dice": dice.item()}


def build_datasets(yml: dict, cfg: Config):
    """Mirrors main.py's data section (main.py:271-356), zarr_store/from_excel branches only --
    the unconditional branch isn't supported here (doesn't make sense for a dense baseline)."""
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

    if cfg.zarr_store:
        import zarr_store_reader
        store_path = cfg.zarr_store
        if cfg.zarr_local_cache:
            store_path = zarr_store_reader.ensure_local_copy(cfg.zarr_store, cfg.zarr_local_cache)
        train_ds = OCTPullbackDataset.from_zarr(
            store_path, cfg, set_excel="training", mode="train", augment=True, **dataset_kwargs,
        )
        val_ds = OCTPullbackDataset.from_zarr(
            store_path, cfg, set_excel="validation", mode="val", augment=False, **dataset_kwargs,
        )
    else:
        folders = paths["data_root"]
        train_ds = OCTPullbackDataset.from_excel(
            folders, df_split, cfg, set_excel="training", mode="train", augment=True, **dataset_kwargs,
        )
        val_ds = OCTPullbackDataset.from_excel(
            folders, df_split, cfg, set_excel="validation", mode="val", augment=False, **dataset_kwargs,
        )
    return train_ds, val_ds


def train_one_epoch_unet(model, loader, optimizer, cfg, device, logger=None, max_batches: int = 0):
    import time
    model.train()
    total_loss = 0.0
    log_accum  = {"ce": 0.0, "dice": 0.0, "wait_pct": 0.0}
    n_batches  = 0
    # Dataloader-wait instrumentation: times next(it) explicitly, same methodology as
    # trainer.py's train_one_epoch and oct-fm's SegTrainer.fit() (seg_loop.py).
    it = iter(loader)
    while True:
        t_fetch   = time.time()
        batch     = next(it, None)
        wait_time = time.time() - t_fetch
        if batch is None:
            break
        volume, _coords, _point_labels, labels = batch
        t0 = time.time()
        volume = volume.to(device)
        labels = labels.to(device)
        center_label = _extract_annotated_label(labels, cfg)   # [B, H, W]

        logits = model(volume)
        loss, log = _dense_loss(logits, center_label, cfg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in log.items():
            log_accum[k] = log_accum.get(k, 0.0) + v
        n_batches += 1

        compute_time = time.time() - t0
        wait_pct     = 100.0 * wait_time / max(wait_time + compute_time, 1e-9)
        log_accum["wait_pct"] += wait_pct

        if n_batches == 1 and logger is not None:
            batch_time  = compute_time
            n_effective = max_batches if max_batches > 0 else len(loader)
            logger.info(f"First batch done in {batch_time:.1f}s (DataWait={wait_pct:.1f}%) — "
                        f"est. {batch_time * n_effective / 60:.1f} min/epoch ({n_effective} batches)")

        if max_batches > 0 and n_batches >= max_batches:
            break

    avg_loss = total_loss / max(n_batches, 1)
    avg_log  = {k: v / max(n_batches, 1) for k, v in log_accum.items()}
    return avg_loss, avg_log


@torch.no_grad()
def validate_unet(model, loader, cfg, device):
    """Pixel-weighted loss + global Dice (all val pixels pooled before averaging) --
    same methodology as trainer.py's validate() (trainer.py:671-733), so Dice numbers
    from the two runs are directly comparable."""
    model.eval()
    total_weighted_loss = 0.0
    total_valid_pixels  = 0
    all_preds, all_lbls = [], []

    for volume, _coords, _point_labels, labels in loader:
        volume = volume.to(device)
        labels = labels.to(device)
        center_label = _extract_annotated_label(labels, cfg)

        logits = model(volume)
        loss, _ = _dense_loss(logits, center_label, cfg)

        flat_labels = center_label.reshape(-1)
        valid   = flat_labels != cfg.ignore_index
        n_valid = int(valid.sum().item())
        total_weighted_loss += loss.item() * n_valid
        total_valid_pixels  += n_valid

        preds = logits.permute(0, 2, 3, 1).reshape(-1, cfg.num_classes).argmax(dim=-1)
        all_preds.append(preds[valid].cpu())
        all_lbls.append(flat_labels[valid].cpu())

    avg_loss = total_weighted_loss / max(total_valid_pixels, 1)

    all_preds_t = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    all_lbls_t  = torch.cat(all_lbls)  if all_lbls  else torch.empty(0, dtype=torch.long)

    dice_per_class = []
    for c in range(cfg.num_classes):
        if c == cfg.ignore_index or c == 0:   # 0 = background: in loss, not in metric (matches trainer.py)
            dice_per_class.append(float("nan"))
            continue
        pred_c = all_preds_t == c
        true_c = all_lbls_t == c
        tp     = (pred_c & true_c).sum().float()
        denom  = pred_c.sum().float() + true_c.sum().float()
        dice_per_class.append(float("nan") if denom == 0 else (2 * tp / denom).item())

    valid_dice = [v for v in dice_per_class if not math.isnan(v)]
    mean_dice  = float(np.mean(valid_dice)) if valid_dice else 0.0
    return avg_loss, mean_dice, dice_per_class


def train_unet(train_ds, val_ds, cfg: Config, device: str, logger, config_path=None):
    if logger is None:
        logger = setup_output_dir(cfg, config_path)

    device = torch.device(device)
    logger.info(f"Device: {device}")
    logger.info(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=(cfg.num_workers > 0),
                              **({"prefetch_factor": cfg.prefetch_factor} if cfg.num_workers > 0 else {}))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.val_num_workers, pin_memory=True,
                            persistent_workers=(cfg.val_num_workers > 0),
                            **({"prefetch_factor": cfg.prefetch_factor} if cfg.val_num_workers > 0 else {}))
    logger.info(f"DataLoader: num_workers={cfg.num_workers} (val={cfg.val_num_workers}), "
                f"prefetch_factor={cfg.prefetch_factor}")

    model = ResidualUNet2p5D(cfg).to(device)
    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Model parameters — total: {total_params:,}  |  encoder: {encoder_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = LinearLR(optimizer, start_factor=cfg.warmup_start_factor, end_factor=1.0,
                      total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, cfg.num_epochs - cfg.warmup_epochs),
                               eta_min=cfg.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])

    # _save_latest_checkpoint/_load_latest_checkpoint expect a scaler (trainer.py:431-474) --
    # kept disabled (no AMP for this baseline) so .state_dict()/.load_state_dict() still work.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    history = {"epoch": [], "train_loss": [], "val_epoch": [], "val_loss": [], "val_dice_mean": []}
    start_epoch, best_dice, loaded_history = _load_latest_checkpoint(
        cfg.checkpoint_dir, model, optimizer, scheduler, scaler, device, logger,
    )
    if loaded_history is not None:
        history = loaded_history

    if start_epoch > cfg.num_epochs:
        logger.info("Run already complete — nothing to do.")
        return model

    max_batches = max(1, int(cfg.max_batches_frac * len(train_ds))) if cfg.max_batches_frac > 0 else 0
    eff_val_interval = max(1, int(cfg.val_interval_pct * cfg.num_epochs)) if cfg.val_interval_pct > 0 else cfg.val_interval
    if max_batches:
        logger.info(f"Max batches/epoch: {max_batches} ({cfg.max_batches_frac * 100:.0f}% of {len(train_ds)})")

    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train_loss, train_log = train_one_epoch_unet(model, train_loader, optimizer, cfg, device,
                                                       logger, max_batches=max_batches)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        log_str = (f"Epoch {epoch:03d}/{cfg.num_epochs} | Train loss: {train_loss:.4f} "
                   f"(CE={train_log['ce']:.3f} Dice={train_log['dice']:.3f})"
                   f" | DataWait={train_log.get('wait_pct', 0):.1f}%")

        if epoch % eff_val_interval == 0 or epoch == cfg.num_epochs or epoch == 1:
            val_loss, val_dice, _ = validate_unet(model, val_loader, cfg, device)
            history["val_epoch"].append(epoch)
            history["val_loss"].append(val_loss)
            history["val_dice_mean"].append(val_dice)
            log_str += f" | Val loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}"

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"epoch": epoch, "model": model.state_dict(), "val_dice": val_dice, "cfg": cfg},
                          os.path.join(cfg.checkpoint_dir, cfg.best_ckpt_name))
                log_str += f"  (best (Dice={best_dice:.4f})"

        logger.info(log_str)
        _save_latest_checkpoint(cfg.checkpoint_dir, epoch, model, optimizer, scheduler, scaler,
                                best_dice, history)

    logger.info(f"Training complete. Best val Dice: {best_dice:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the 2.5D Residual U-Net baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--sanity_check", action="store_true",
                        help="Run a quick forward-pass check with random tensors and exit")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        yml = yaml.safe_load(f)
    cfg = build_config(yml)

    if args.sanity_check:
        sanity_check(cfg)
        return

    seed = yml.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_ds, val_ds = build_datasets(yml, cfg)
    device = yml.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_output_dir(cfg, config_path=args.config)
    logger.info(f"Experiment    : {yml['experiment_name']}")
    logger.info(f"Model         : ResidualUNet2p5D (encoder_type={cfg.encoder_type})")

    train_unet(train_ds, val_ds, cfg=cfg, device=device, logger=logger, config_path=args.config)


if __name__ == "__main__":
    main()
