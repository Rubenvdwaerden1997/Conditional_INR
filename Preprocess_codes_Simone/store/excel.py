"""Parse the `train_val_test_split` Excel sheet into the planned-pullback table.

The sheet is the only source of the split, of the manually annotated frames and of the
per-class frame counts used as an end-to-end QC target. Nothing here touches pixels.

Conventions chosen here (stated because the plan leaves them open):
- `patient_id` comes from the sheet's own `PatientID` column (267 distinct patients for
  300 pullbacks); no id is parsed out of the pullback name.
- `excel_row` is the 0-based row position in the sheet, `excel_id` the sheet's `ID` value.
- `excel_n_frames` is `Shape[0]` (the volume depth the sheet claims), NOT `Nº frames`
  (which counts the manual frames and is only used for the consistency warning).
- flag columns are int8 with -1 for missing, `dem_age` is float32 with NaN for missing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..data.labels import CLASS_TOKENS

# The 35 headers of the real sheet, in file order ('º' is U+00BA).
EXCEL_HEADERS: tuple[str, ...] = (
    "Patient", "ID", "Nº pullback", "Pullback", "Shape", "Spacing", "Frames", "Nº frames",
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media", "Catheter",
    "Sidebranch", "Redthrombus", "Whitethrombus", "Dissection", "Plaque_rupture", "Layered_plaque",
    "Neovascularization", "Intraplaque_hemorrhage", "Intramural_hematoma",
    "Dem_age", "Dem_Gender", "MH_MI", "MH_DM", "MH_HT", "MH_HC", "MedPresentation_LipidLowTher",
    "Dataset", "PatientID", "Set",
)
# class-count columns in label-value order 0..16 (they pair with CLASS_TOKENS)
EXCEL_CLASS_COLUMNS: tuple[str, ...] = EXCEL_HEADERS[8:25]
XL_FRAMES_COLUMNS: tuple[str, ...] = tuple(f"xl_frames_{t}" for t in CLASS_TOKENS)
# sheet flag column -> output column; all int8, -1 when missing
FLAG_COLUMNS: dict[str, str] = {
    "Dem_Gender": "dem_gender", "MH_MI": "mh_mi", "MH_DM": "mh_dm", "MH_HT": "mh_ht",
    "MH_HC": "mh_hc", "MedPresentation_LipidLowTher": "lipid_lowering", "Dataset": "dataset",
}
SPLIT_FROM_SET: dict[str, str] = {"Training": "train", "Validation": "val", "Testing": "test"}
REQUIRED_COLUMNS: tuple[str, ...] = (
    ("Pullback", "PatientID", "Set", "Shape", "Frames", "Nº frames", "ID")
    + EXCEL_CLASS_COLUMNS + ("Dem_age",) + tuple(FLAG_COLUMNS)
)


def parse_frames(s) -> list[int]:
    """Manual-frame string "1,41, 47" (1-based) -> sorted 0-based [0, 40, 46]; NaN/empty -> []."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return []
    text = str(s).strip()
    if not text:
        return []
    out = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError(f"manual frame indices are 1-based, got {value!r} in {s!r}")
        out.add(value - 1)
    return sorted(out)


def parse_shape(s) -> int:
    """Excel `Shape` string "( 540, 704,704)" -> 540 (first entry); NaN/unparsable -> -1."""
    if s is None or (not isinstance(s, str) and pd.isna(s)):
        return -1
    text = str(s).strip().lstrip("(").split(",")[0].strip()
    try:
        return int(text)
    except ValueError:
        return -1


def load_split_table(path: str, sheet: str) -> pd.DataFrame:
    """One row per Excel pullback with the columns up to the demographics.

    Asserts unique pullback ids and a valid `Set`; warns when `Nº frames` disagrees with `Frames`.
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}[{sheet}] is missing columns: {missing}")

    pullback_id = raw["Pullback"].astype(str).str.strip()
    duplicated = sorted(pullback_id[pullback_id.duplicated()].unique())
    if duplicated:
        raise ValueError(f"{path}[{sheet}] has duplicate pullback ids: {duplicated}")
    bad_set = sorted(set(raw["Set"].astype(str)) - set(SPLIT_FROM_SET))
    if bad_set:
        raise ValueError(f"{path}[{sheet}] has unknown Set values: {bad_set}")

    manual_frames = [parse_frames(s) for s in raw["Frames"]]
    n_manual = np.array([len(m) for m in manual_frames], dtype=np.int32)
    excel_n_manual = raw["Nº frames"].to_numpy(dtype=np.int64)
    mismatch = np.flatnonzero(n_manual != excel_n_manual)
    for i in mismatch:
        warnings.warn(
            f"{pullback_id.iloc[i]}: 'Nº frames' says {excel_n_manual[i]} but 'Frames' lists "
            f"{n_manual[i]} manual frames; using the parsed list",
            UserWarning, stacklevel=2)

    out = pd.DataFrame({
        "pullback_id": pullback_id.to_numpy(dtype=object).astype(str),
        "patient_id": raw["PatientID"].astype(str).str.strip().to_numpy(dtype=object).astype(str),
        "split": [SPLIT_FROM_SET[s] for s in raw["Set"].astype(str)],
        "excel_row": np.arange(len(raw), dtype=np.int32),
        "excel_id": raw["ID"].to_numpy(dtype=np.int64).astype(np.int32),
        "excel_n_frames": np.array([parse_shape(s) for s in raw["Shape"]], dtype=np.int32),
        "n_manual": n_manual,
        "manual_frames": manual_frames,
    })
    for token_col, excel_col in zip(XL_FRAMES_COLUMNS, EXCEL_CLASS_COLUMNS):
        out[token_col] = raw[excel_col].fillna(0).to_numpy(dtype=np.int64).astype(np.int16)
    out["dem_age"] = raw["Dem_age"].to_numpy(dtype=np.float64).astype(np.float32)
    for excel_col, flag_col in FLAG_COLUMNS.items():
        out[flag_col] = raw[excel_col].fillna(-1).to_numpy(dtype=np.float64).astype(np.int8)
    return out
