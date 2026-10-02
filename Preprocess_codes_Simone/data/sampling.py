"""Frame-table sampling weights (`frames.sampler`) and the weighted sampler built from them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from ..config import FramesCfg
from .labels import ClassMap, class_presence

EPS = 1e-6      # keeps f_c ** -alpha finite for classes that never occur


def class_balanced_weights(frames_df: pd.DataFrame, cfg: FramesCfg, cm: ClassMap) -> np.ndarray:
    """w_i = min(cap, max over present classes of f_c ** -alpha); 1 when no class is present; mean 1."""
    present = class_presence(frames_df, cm)                            # [N, C]
    if len(frames_df) == 0:
        return np.ones(0)
    frequency = present.mean(axis=0)                                   # f_c over the eligible rows
    per_class = (frequency + EPS) ** (-cfg.sampler_alpha)              # [C]
    rarest = np.where(present, per_class[None, :], 0.0).max(axis=1)    # 0 when nothing is present
    return _normalized(np.where(present.any(axis=1), np.minimum(cfg.sampler_cap, rarest), 1.0))


WEIGHT_SCHEMES = {"class_balanced": class_balanced_weights}


def compute_frame_weights(frames_df: pd.DataFrame, cfg: FramesCfg, cm: ClassMap) -> np.ndarray:
    """Per-row sampling weight for `cfg.sampler` (anything but `uniform`); result has mean 1."""
    spec = str(cfg.sampler)
    if spec not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown sampler {spec!r}: expected 'uniform' or one of {sorted(WEIGHT_SCHEMES)}")
    return WEIGHT_SCHEMES[spec](frames_df, cfg, cm)


def build_weighted_sampler(weights: np.ndarray, num_samples: int, seed: int) -> WeightedRandomSampler:
    """Sampling with replacement from `weights`, reproducible for a given seed."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(np.asarray(weights, dtype=np.float64)),
                                 num_samples=int(num_samples), replacement=True, generator=generator)


def _normalized(weights: np.ndarray) -> np.ndarray:
    """w / w.mean()."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return weights
    mean = weights.mean()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"sampling weights have mean {mean}, cannot normalise")
    return weights / mean
