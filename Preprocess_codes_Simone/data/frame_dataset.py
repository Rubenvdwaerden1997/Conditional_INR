"""Map-style dataset over frame-store windows: polar clips of `window` frames around an anchor frame."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from ..config import FramesCfg
from ..preprocess.polar import PolarMapper
from .labels import ClassMap, build_class_map, map_label
from .sampling import compute_frame_weights
from .store import OCTStore, window_indices

LABEL_SOURCES = ("manual", "pseudo", "all")


class FrameWindowDataset(torch.utils.data.Dataset):
    """One item = one eligible anchor frame plus the `window` frames `anchor + stride * (k - window // 2)`.
    With `z_reference_mm > 0` the stride is multiplied per pullback by `stride_scale` (2 on a 0.1 mm pullback
    for a 0.2 mm reference), so every window spans the same physical length of vessel.

    `split` names the splits the anchors come from ("train", or several as "train,unlabelled"). With
    `with_label=True` only frames that carry the requested `label_source` are anchors; with `with_label=False`
    every frame of the split is (pretraining clips, whole-pullback inference, the consistency regulariser).

    `__getitem__`: read the uint8 window -> map the label through `class_map` -> polar (bilinear image,
    nearest label) -> `transform(sample)` -> normalise -> tensors. The transform sees the **numpy** sample
    after the geometry and before normalisation, so it works on raw pixel values and may change the arrays:

        image       uint8   [T, R, THETA]
        label       uint8   [T, R, THETA]   mapped ids (with_label only)
        is_manual   bool    [T]             (with_label only)
        frames      int64   [T]             absolute frame indices
        frame, n_frames, pullback_idx int;  pullback_id str;  z_mm float

    The tensors returned add the channel axis: `image` float32 [1, T, R, THETA], `label` int64 [T, R, THETA].
    The `PolarMapper`, the RNG and the zarr group are created lazily per worker process.
    """

    def __init__(self, store: OCTStore, cfg: FramesCfg, split: str, train: bool, transform=None,
                 center: tuple[float, float] = (352.0, 352.0), with_label: bool = True, seed: int = 0):
        self.store = store
        self.cfg = cfg
        self.split = str(split)
        self.train = bool(train)
        self.transform = transform
        self.center = (float(center[0]), float(center[1]))
        self.with_label = bool(with_label)
        self.seed = int(seed)

        self.label_source = _one_of(cfg.label_source, LABEL_SOURCES, "label_source")
        self.window = _positive(cfg.window, "window")
        self.strides = _checked_strides(cfg.stride)
        self.z_reference_mm = _non_negative(cfg.z_reference_mm, "z_reference_mm")
        self.class_map: ClassMap = build_class_map(cfg.class_map)
        self.frame_shape = tuple(int(v) for v in store.info["layout"]["frame_shape"])
        self.grid_shape = self._polar_grid()
        self.image_mean, self.image_std = self._norm_stats()

        self.rows = _eligible_rows(store.frames, _checked_splits(self.split, store), cfg, self.with_label)
        self.frames: pd.DataFrame = store.frames.iloc[self.rows]
        self._anchor_pullback = self.frames["pullback_id"].astype(str).to_numpy()
        self._anchor_frame = self.frames["frame"].to_numpy(dtype=np.int64)
        self._anchor_n_frames = self.frames["n_frames"].to_numpy(dtype=np.int64)
        self._anchor_z_mm = self.frames["z_mm"].to_numpy(dtype=np.float32)
        self._anchor_pullback_idx = pd.Index(store.pullbacks["pullback_id"].astype(str)).get_indexer(
            self._anchor_pullback).astype(np.int64)
        self._anchor_stride_scale = stride_scale(
            store.pullbacks["z_spacing_mm"].to_numpy(dtype=float)[self._anchor_pullback_idx], self.z_reference_mm)
        self._table_is_manual = store.frames["is_manual"].to_numpy(dtype=bool)   # whole table: window rows

        self.weights: np.ndarray | None = None
        if self.train and str(cfg.sampler) != "uniform":
            self.weights = compute_frame_weights(self.frames, cfg, self.class_map)

        self._mapper: PolarMapper | None = None
        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, i: int) -> dict:
        pullback_id = str(self._anchor_pullback[i])
        anchor = int(self._anchor_frame[i])
        idx = window_indices(anchor, self.window, self._stride(i), int(self._anchor_n_frames[i]))
        mapper = self._polar_mapper()
        sample = {"image": mapper.to_polar(self.store.read_frames(pullback_id, idx)),
                  "frames": idx, "frame": anchor, "n_frames": int(self._anchor_n_frames[i]),
                  "pullback_id": pullback_id, "pullback_idx": int(self._anchor_pullback_idx[i]),
                  "z_mm": float(self._anchor_z_mm[i])}
        if self.with_label:
            raw = self.store.read_labels(pullback_id, idx)
            sample["label"] = mapper.to_polar_nearest(map_label(raw, self.class_map))
            first_row, _ = self.store.frame_rows(pullback_id)
            sample["is_manual"] = self._table_is_manual[first_row + idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return self._to_tensors(sample)

    def __getstate__(self):
        """Drop the per-worker objects so the dataset pickles into DataLoader workers."""
        state = self.__dict__.copy()
        state["_mapper"] = state["_rng"] = state["_rng_pid"] = None
        return state

    def _stride(self, i: int) -> int:
        """The configured stride (with several, one drawn per sample in train mode, the first otherwise)
        times the z-spacing scale of anchor `i`'s pullback."""
        if len(self.strides) == 1 or not self.train:
            stride = self.strides[0]
        else:
            stride = int(self._worker_rng().choice(self.strides))
        return stride * int(self._anchor_stride_scale[i])

    def _to_tensors(self, sample: dict) -> dict:
        image = np.ascontiguousarray(sample["image"], dtype=np.float32)
        if self.cfg.normalize:
            image = (image - self.image_mean) / self.image_std
        out = {"image": torch.from_numpy(image)[None],                      # [1, T, R, THETA]
               "frames": torch.as_tensor(np.ascontiguousarray(sample["frames"]), dtype=torch.int64),
               "frame": torch.as_tensor(sample["frame"], dtype=torch.int64),
               "n_frames": torch.as_tensor(sample["n_frames"], dtype=torch.int64),
               "pullback_idx": torch.as_tensor(sample["pullback_idx"], dtype=torch.int64),
               "pullback_id": sample["pullback_id"],
               "z_mm": torch.as_tensor(sample["z_mm"], dtype=torch.float32)}
        if "label" in sample:
            out["label"] = torch.as_tensor(np.ascontiguousarray(sample["label"]), dtype=torch.int64)
            out["is_manual"] = torch.as_tensor(np.ascontiguousarray(sample["is_manual"]), dtype=torch.bool)
        return out

    # ---------- lazily built per worker process ----------

    def _polar_mapper(self) -> PolarMapper:
        if self._mapper is None:
            self._mapper = PolarMapper(size=self.frame_shape[0], r0=self.cfg.polar_r0, r1=self.cfg.polar_r1,
                                       n_theta=self.cfg.polar_n_theta, center=self.center)
        return self._mapper

    def _worker_rng(self) -> np.random.Generator:
        """Seeded from `torch.initial_seed()` inside a DataLoader worker (unique per worker and epoch),
        from `seed` elsewhere. The pid guard re-creates the generator in a forked child."""
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            in_worker = torch.utils.data.get_worker_info() is not None
            self._rng = np.random.default_rng(int(torch.initial_seed()) if in_worker else self.seed)
            self._rng_pid = pid
        return self._rng

    # ---------- cfg validation ----------

    def _polar_grid(self) -> tuple[int, int]:
        r0, r1, n_theta = int(self.cfg.polar_r0), int(self.cfg.polar_r1), int(self.cfg.polar_n_theta)
        if not 0 <= r0 < r1:
            raise ValueError(f"polar radii must satisfy 0 <= polar_r0 < polar_r1, got {r0} and {r1}")
        if n_theta < 1:
            raise ValueError(f"polar_n_theta must be >= 1, got {n_theta}")
        return r1 - r0, n_theta

    def _norm_stats(self) -> tuple[float, float]:
        mean, std = float(self.store.info["image_mean"]), float(self.store.info["image_std"])
        if self.cfg.normalize and not std > 0:
            raise ValueError(f"normalize=True needs a positive image_std in dataset_info, got {std}")
        return mean, std


def stride_scale(z_spacing_mm: np.ndarray, z_reference_mm: float) -> np.ndarray:
    """Per-pullback stride multiplier `max(1, round(z_reference_mm / z_spacing_mm))` -> int64 [N].

    30 of the 47 labelled pullbacks have 0.2 mm frame spacing and 17 have 0.1 mm, so an 8-frame stride-1
    window is 1.6 mm of vessel on one and 0.8 mm on the other; with a 0.2 mm reference the 0.1 mm pullbacks
    use twice the stride and every window spans the same length. All ones when `z_reference_mm <= 0` (off)
    and 1 wherever the spacing is unknown (NaN) or not positive.
    """
    z_spacing_mm = np.asarray(z_spacing_mm, dtype=float)
    scale = np.ones(z_spacing_mm.shape, dtype=np.int64)
    if z_reference_mm <= 0:
        return scale
    known = np.isfinite(z_spacing_mm) & (z_spacing_mm > 0)
    scale[known] = np.maximum(1, np.rint(z_reference_mm / z_spacing_mm[known])).astype(np.int64)
    return scale


def _eligible_rows(frames: pd.DataFrame, splits: tuple[str, ...], cfg: FramesCfg, with_label: bool) -> np.ndarray:
    """Anchor rows of the frame table: the requested splits, then the `label_source` filter when a label
    is wanted (`with_label=False` makes every frame of an unlabelled pullback eligible)."""
    keep = frames["split"].isin(splits).to_numpy(dtype=bool)
    if with_label:
        has_label = frames["has_label"].to_numpy(dtype=bool)
        is_manual = frames["is_manual"].to_numpy(dtype=bool)
        keep = keep & {"manual": is_manual, "pseudo": has_label & ~is_manual, "all": has_label}[cfg.label_source]
    return np.flatnonzero(keep)


def _checked_splits(split: str, store: OCTStore) -> tuple[str, ...]:
    """"train" or "train,unlabelled" -> the requested split names, checked against the store."""
    splits = tuple(part.strip() for part in split.split(",") if part.strip())
    if not splits:
        raise ValueError(f"split is empty: {split!r}")
    known = set(store.info.get("splits", {})) | set(store.frames["split"].unique())
    unknown = [s for s in splits if s not in known]
    if unknown:
        raise ValueError(f"unknown split(s) {unknown}; the store has {sorted(known)}")
    return splits


def _checked_strides(stride) -> tuple[int, ...]:
    """An int or a non-empty list of ints >= 1 -> tuple of strides."""
    strides = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,)
    if not strides:
        raise ValueError("stride must be an int or a non-empty list of ints")
    return tuple(_positive(s, "stride") for s in strides)


def _one_of(value, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _positive(value, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _non_negative(value, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value
