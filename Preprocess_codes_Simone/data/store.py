"""Read side of the Cartesian frame store: tables + lazy per-process zarr group."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import zarr

from ..store.writer import open_root

# read_frames/read_labels read one contiguous span [min, max+1] and gather from it, which costs a single
# shard-aligned pass for the windows the sampler actually produces. A strided window can still ask for a span
# far wider than the frames it keeps, so when the span exceeds SPAN_FACTOR x the number of distinct frames
# wanted we switch to orthogonal indexing of just those frames.
SPAN_FACTOR = 2


def window_indices(anchor: int, window: int, stride: int, n_frames: int) -> np.ndarray:
    """Frames `anchor + stride * (k - window // 2)`, clipped to [0, n_frames) -> int64 [T]."""
    k = np.arange(window, dtype=np.int64)
    idx = np.int64(anchor) + np.int64(stride) * (k - window // 2)
    return np.clip(idx, 0, n_frames - 1)


class OCTStore:
    """Loads dataset_info.json, pullbacks.parquet and frames.parquet; opens zarr lazily per pid."""

    def __init__(self, root: str):
        self.root = str(root)
        with open(os.path.join(self.root, "dataset_info.json")) as f:
            self.info = json.load(f)
        self.class_names: tuple[str, ...] = tuple(self.info["class_names"])
        self.pullbacks: pd.DataFrame = pd.read_parquet(os.path.join(self.root, "pullbacks.parquet"))
        self.frames: pd.DataFrame = pd.read_parquet(os.path.join(self.root, "frames.parquet"))
        self._frame_rows = _frame_row_spans(self.frames)
        self._group: zarr.Group | None = None
        self._group_pid: int | None = None

    def group(self) -> zarr.Group:
        """Cached per os.getpid(); use_consolidated=True with fallback to False."""
        pid = os.getpid()
        if self._group is None or self._group_pid != pid:
            self._group = open_root(os.path.join(self.root, "store.zarr"), "r")
            self._group_pid = pid
        return self._group

    def read_frames(self, pullback_id: str, idx: np.ndarray) -> np.ndarray:
        """uint8 [T, 704, 704]; one contiguous slice (or oindex) then gather."""
        return self._read(pullback_id, "image", idx)

    def read_labels(self, pullback_id: str, idx: np.ndarray) -> np.ndarray:
        """uint8 [T, 704, 704]."""
        return self._read(pullback_id, "label", idx)

    def attrs(self, pullback_id: str) -> dict:
        return self.group()[pullback_id].attrs.asdict()

    def frame_rows(self, pullback_id: str) -> tuple[int, int]:
        """(first_row, n_frames) of the pullback in `frames`."""
        return self._frame_rows[pullback_id]

    def __getstate__(self):
        """Drop the cached zarr group so the store pickles into DataLoader workers."""
        state = self.__dict__.copy()
        state["_group"] = None
        state["_group_pid"] = None
        return state

    def _read(self, pullback_id: str, name: str, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        arr = self.group()[pullback_id][name]
        unique = np.unique(idx)
        lo, hi = int(unique[0]), int(unique[-1]) + 1
        if hi - lo <= SPAN_FACTOR * unique.size:
            return arr[lo:hi][idx - lo]
        return arr.oindex[unique][np.searchsorted(unique, idx)]


def _frame_row_spans(frames: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """{pullback_id: (first_row, n_frames)} from the pullback-sorted frame table."""
    ids = frames["pullback_id"].to_numpy()
    if ids.size == 0:
        return {}
    starts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
    ends = np.concatenate((starts[1:], [ids.size]))
    return {str(ids[s]): (int(s), int(e - s)) for s, e in zip(starts, ends)}
