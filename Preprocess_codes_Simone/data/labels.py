"""Class names, class maps (raw17 / preset12 / user dict) and frame-table helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# label-value order 0..16
CLASS_NAMES: tuple[str, ...] = (
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media", "Catheter", "Sidebranch",
    "Red_thrombus", "White_thrombus", "Dissection", "Plaque_rupture", "Layered_plaque", "Neovascularization",
    "Intraplaque_hemorrhage", "Intramural_hematoma",
)
CLASS_TOKENS: tuple[str, ...] = (
    "background", "lumen", "guidewire", "intima", "lipid", "calcium", "media", "catheter", "sidebranch",
    "redthrombus", "whitethrombus", "dissection", "plaque_rupture", "layered_plaque", "neovascularization",
    "intraplaque_hemorrhage", "intramural_hematoma",
)
N_CLASSES = 17

PRESET12_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 1, 8: 7, 9: 8, 10: 8, 11: 0, 12: 9, 13: 10,
                14: 11, 15: 0, 16: 0}
PRESET12_NAMES = ("Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media", "Sidebranch",
                  "Thrombus", "Plaque_rupture", "Healed_plaque", "Neovascularization")

PRESETS: dict[str, tuple[dict, tuple[str, ...]]] = {"preset12": (PRESET12_MAP, PRESET12_NAMES)}


@dataclass(frozen=True)
class ClassMap:
    lut: np.ndarray            # uint8 [17]: raw label value -> mapped id
    names: tuple[str, ...]     # mapped class names, index = mapped id
    n_classes: int


def build_class_map(spec: str | dict) -> ClassMap:
    """"raw17" | "preset12" | {src_name_or_id: dst_id, ..., "names": [...]} -> ClassMap."""
    if isinstance(spec, str):
        if spec == "raw17":
            return ClassMap(np.arange(N_CLASSES, dtype=np.uint8), CLASS_NAMES, N_CLASSES)
        if spec in PRESETS:
            mapping, names = PRESETS[spec]
            lut = np.zeros(N_CLASSES, dtype=np.uint8)
            for src, dst in mapping.items():
                lut[src] = dst
            return ClassMap(lut, names, len(names))
        known = ", ".join(repr(k) for k in ("raw17", *PRESETS))
        raise ValueError(f"unknown class map {spec!r}: expected one of {known}, or a dict")
    if not isinstance(spec, dict):
        raise TypeError(f"class map spec must be a str or a dict, got {type(spec).__name__}")

    entries = dict(spec)
    raw_names = entries.pop("names", None)
    if raw_names is None:
        raise ValueError("a class-map dict needs a 'names' entry listing the mapped class names")
    names = tuple(str(n) for n in raw_names)
    if not names:
        raise ValueError("a class-map dict needs at least one mapped class name")
    lut = np.zeros(N_CLASSES, dtype=np.uint8)   # unspecified raw ids -> 0 (background)
    for key, dst in entries.items():
        src = _raw_class_id(key)
        dst_id = int(dst)
        if not 0 <= dst_id < len(names):
            raise ValueError(f"class map: destination id {dst_id} for {key!r} is outside 0..{len(names) - 1}")
        lut[src] = dst_id
    return ClassMap(lut, names, len(names))


def map_label(label: np.ndarray, cm: ClassMap) -> np.ndarray:
    """Apply cm.lut to a raw uint8 label array."""
    return cm.lut[label]


def map_counts(counts17: np.ndarray, cm: ClassMap) -> np.ndarray:
    """[..., 17] pixel counts -> [..., C] (sum over merged raw classes)."""
    counts17 = np.asarray(counts17)
    out = np.zeros(counts17.shape[:-1] + (cm.n_classes,), dtype=counts17.dtype)
    for c in range(cm.n_classes):
        merged = cm.lut == c
        if merged.any():
            out[..., c] = counts17[..., merged].sum(axis=-1)
    return out


def px_columns() -> list[str]:
    """["px_background", ..., "px_intramural_hematoma"]."""
    return [f"px_{token}" for token in CLASS_TOKENS]


def bbox_columns() -> list[str]:
    """["bbox_background_y0", "bbox_background_y1", "bbox_background_x0", "bbox_background_x1", ...] (17 x 4)."""
    return [f"bbox_{token}_{side}" for token in CLASS_TOKENS for side in ("y0", "y1", "x0", "x1")]


def frame_counts(frames_df: pd.DataFrame) -> np.ndarray:
    """int32 [N, 17] from the px_ columns."""
    return frames_df[px_columns()].to_numpy(dtype=np.int32)


def class_presence(frames_df: pd.DataFrame, cm: ClassMap) -> np.ndarray:
    """bool [N, C]: the stored frame holds at least one pixel of the mapped class."""
    return map_counts(frame_counts(frames_df), cm) > 0


def _raw_class_id(key) -> int:
    """Raw class name or 0..16 id (int or digit string) -> raw class id."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        raw_id = int(key)
    elif isinstance(key, str) and key in CLASS_NAMES:
        raw_id = CLASS_NAMES.index(key)
    elif isinstance(key, str) and key.strip().isdigit():
        raw_id = int(key)
    else:
        raise ValueError(f"unknown raw class {key!r}: expected a CLASS_NAMES entry or an id 0..{N_CLASSES - 1}")
    if not 0 <= raw_id < N_CLASSES:
        raise ValueError(f"raw class id {raw_id} is outside 0..{N_CLASSES - 1}")
    return raw_id
