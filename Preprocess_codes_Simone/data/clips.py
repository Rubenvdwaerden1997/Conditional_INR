"""V-JEPA clips out of the frame store: a label-free `FrameWindowDataset` sample of `frames.window`
frames, augmented by `ClipAugment` (which also crops theta to `pretrain.theta_crop`)."""
from __future__ import annotations

import dataclasses

import torch
from torch.utils.data import DataLoader

from ..config import Config
from .frame_loader import build_frame_loader
from .transforms import ClipAugment


def clip_input_size(cfg: Config) -> tuple[int, int, int]:
    """Model input (T, R, THETA) implied by the config."""
    fc = cfg.frames
    return fc.window, fc.polar_r1 - fc.polar_r0, cfg.pretrain.theta_crop


def build_clip_loader(cfg: Config, train: bool, every: int = 1) -> DataLoader:
    """Train: `pretrain.train_split` anchors, uniformly sampled, augmented. Eval: `pretrain.val_split`, in
    order, theta cropped only, every `every`-th anchor (so an eval bank spans the pullbacks)."""
    pc = cfg.pretrain
    clip_cfg = dataclasses.replace(cfg, frames=dataclasses.replace(cfg.frames, sampler="uniform"))
    transform = ClipAugment(pc.theta_crop, pc.r_jitter, pc.flip_p, pc.time_reverse_p, pc.brightness,
                            pc.contrast, train=train)
    if train:
        return build_frame_loader(clip_cfg, split=pc.train_split, train=True, transform=transform,
                                  with_label=False)
    return build_frame_loader(clip_cfg, split=pc.val_split, train=False, transform=transform,
                              with_label=False, every=every)


def z_frac(batch: dict) -> torch.Tensor:
    """Normalised position of the anchor in its pullback, float32 [B]."""
    return batch["frame"].float() / (batch["n_frames"] - 1).clamp(min=1).float()
