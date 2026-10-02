"""Per-frame class pixel counts and bounding boxes -> frames.parquet rows.

`frame_class_stats` is the only place the per-frame statistics are derived, so the frame table,
the class-guided crops and the sampling weights all see the same numbers. It walks the volume in
blocks of frames and, inside a block, only over the class values that actually occur there, which
keeps the temporaries small (one bool [block, H, W]) and the number of passes low.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.labels import bbox_columns, px_columns


def frame_class_stats(label: np.ndarray, n_classes: int = 17, block: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """label uint8 [T, H, W] -> (counts int32 [T, n_classes], bbox int16 [T, n_classes, 4] as (y0, y1, x0, x1),
    end-exclusive, all -1 when the class is absent)."""
    label = np.asarray(label)
    if label.ndim != 3:
        raise ValueError(f"label must be [T, H, W], got shape {label.shape}")
    n_frames, height, width = label.shape
    step = max(int(block), 1)
    counts = np.zeros((n_frames, n_classes), dtype=np.int32)
    bboxes = np.full((n_frames, n_classes, 4), -1, dtype=np.int16)
    for start in range(0, n_frames, step):
        frames = label[start:start + step]
        for value in np.unique(frames):
            class_id = int(value)
            if class_id >= n_classes:
                raise ValueError(f"label value {class_id} is outside the {n_classes}-class range")
            present = frames == class_id
            in_row = np.count_nonzero(present, axis=2) > 0            # [B, H]
            in_col = np.count_nonzero(present, axis=1) > 0            # [B, W]
            counts[start:start + len(frames), class_id] = np.count_nonzero(present, axis=(1, 2))
            in_frame = in_row.any(axis=1)
            box = np.stack([in_row.argmax(axis=1), height - in_row[:, ::-1].argmax(axis=1),
                            in_col.argmax(axis=1), width - in_col[:, ::-1].argmax(axis=1)], axis=1)
            bboxes[start + np.flatnonzero(in_frame), class_id] = box[in_frame].astype(np.int16)
    return counts, bboxes


def stats_frame(pullback_id: str, patient_id: str, split: str, n_frames: int, manual_frames: list[int],
                has_label: bool, z_spacing: float | None, counts: np.ndarray, bboxes: np.ndarray) -> pd.DataFrame:
    """One frames row per frame (unlabelled: zero counts, -1 boxes, has_label=False)."""
    counts = np.asarray(counts, dtype=np.int32)
    bboxes = np.asarray(bboxes, dtype=np.int16)
    if len(counts) != n_frames or len(bboxes) != n_frames:
        raise ValueError(f"{pullback_id}: statistics cover {len(counts)} frames, expected {n_frames}")
    frame = np.arange(n_frames, dtype=np.int32)
    is_manual = np.zeros(n_frames, dtype=bool)
    is_manual[np.asarray(manual_frames, dtype=np.int64)] = True
    spacing = np.float32(z_spacing) if z_spacing is not None else np.float32("nan")
    out = pd.DataFrame({
        "pullback_id": pd.array([pullback_id] * n_frames, dtype="str"),
        "patient_id": pd.array([patient_id] * n_frames, dtype="str"),
        "split": pd.array([split] * n_frames, dtype="str"),
        "frame": frame,
        "n_frames": np.full(n_frames, n_frames, dtype=np.int32),
        "is_manual": is_manual,
        "has_label": np.full(n_frames, bool(has_label), dtype=bool),
        "z_mm": (frame * spacing).astype(np.float32),
    })
    for column, values in zip(px_columns(), counts.T):
        out[column] = values.astype(np.int32)
    flat_boxes = bboxes.reshape(n_frames, bboxes.shape[1] * bboxes.shape[2])   # -1 cannot infer at n_frames=0
    for column, values in zip(bbox_columns(), flat_boxes.T):
        out[column] = values.astype(np.int16)
    return out
