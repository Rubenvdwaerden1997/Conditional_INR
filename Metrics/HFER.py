#!/usr/bin/env python3
"""
HFER -- High Frequency Energy Ratio of per-class segmentation area along the pullback.

Measures longitudinal (frame-to-frame) flicker of a segmentation, without needing ground
truth. For every pullback and every class:
  1. count the class's pixels in every frame        -> a 1D signal along z
  2. divide by the class's max over frames          -> signal in [0, 1]
  3. power spectrum of that signal (rfft), DC bin dropped
  4. HFER = 100 * (power at freq >= cutoff) / (total AC power)

A smooth, anatomically plausible segmentation concentrates its power at low frequencies
(low HFER); a segmentation whose class area jumps between neighbouring frames scores high.

The definition (per-class max normalisation, DC excluded from the denominator, result in
percent, NaN when the class is absent or the AC power is zero) follows the reference
implementation in W:/Jlucas/tools/create_figures/HFER.py so numbers are comparable.

Cutoff: the reference uses 0.1 cycles/frame, commented as ~1 cycle/mm at 0.1 mm z-spacing.
Pullbacks here do NOT share one z-spacing, so the cutoff is specified physically
(--threshold_cycles_per_mm, default 1.0) and converted per pullback to cycles/frame as
cycles_per_mm * z_spacing_mm. For a 0.1 mm pullback this is exactly the reference's 0.1.
z-spacing is inferred from the pullback's frame count (the prediction .nii.gz header is NOT
reliable: Conditional INR's writer stamps the config's constant z_spacing on every pullback).

This module is also imported by Pullback_prediction*.py (HFER computed in the Predict_3D_Model
runs, saved into test_metrics.xlsx). Run directly, it scores already-written full-pullback
prediction volumes instead.

Notes:
  - Guiding-catheter and artifact frames (predictions there are wrong by definition) are left out,
    using the same Excel as Continuous_3D_metrics.py (--exclusion_excel; default
    Continuous_3D_Metrics/Exclusion_Frames_testset.xlsx; empty string = exclude nothing). An FFT
    needs evenly spaced frames, so the volume is split into unbroken stretches of kept frames; each
    stretch >= --min_stretch_frames (default 32) is scored on its own and the stretches are averaged
    weighted by length. Guiding frames sit at the end of a pullback, so they just shorten the last
    stretch. The reference implementation does NOT exclude any frames.
  - Ground truth is only annotated on some frames, so run this on full-pullback predictions.

Usage (standalone):
    python HFER.py \
        --pred_dirs   /path/to/model_a/predictions /path/to/model_b/predictions \
        --model_names ModelA ModelB \
        [--pattern "*_postprocessed.nii.gz"] \
        [--threshold_cycles_per_mm 1.0] \
        [--z_spacing_mm 0.1] \
        [--min_ac_power 0.0] \
        [--output_dir /path/to/output]
"""
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

# Canonical 12-class post-mapping taxonomy (see Continuous_3D_metrics.py / label_description_conditionalinr.txt).
_CLASS_NAMES_12 = [
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media",
    "Sidebranch", "Thrombus", "Plaque rupture", "Layered plaque", "Neovascularization",
]
_N_CLASSES = len(_CLASS_NAMES_12)

# Pullback ID = everything before the first "_pred" token in the filename stem.
_ID_PATTERN = re.compile(r"^(.+?)_pred", re.IGNORECASE)


def infer_z_spacing_mm(n_frames: int, fallback: float) -> float:
    """Same frame-count heuristic as Continuous_3D_Metrics/Continuous_3D_metrics.py: ~75 mm
    pullbacks run at 350-400 frames, ~54 mm pullbacks at 250-300 or 500-550 frames. Any other
    frame count -> `fallback` (with a warning) rather than a silent guess."""
    if 350 <= n_frames <= 400:
        return 75.0 / n_frames
    if (250 <= n_frames <= 300) or (500 <= n_frames <= 550):
        return 54.0 / n_frames
    print(f"[WARN] Frame count {n_frames} doesn't match a known pullback-length bracket "
          f"(350-400 -> 75mm, 250-300/500-550 -> 54mm) -- using fallback z_spacing={fallback}mm")
    return fallback


def cutoff_cycles_per_frame(n_frames: int, cycles_per_mm: float, fallback_z_spacing_mm: float,
                            z_spacing_override_mm: Optional[float] = None):
    """Per-pullback cutoff in cycles/frame. Returns (z_spacing_mm, cutoff_cycles_per_frame)."""
    z = z_spacing_override_mm if z_spacing_override_mm else infer_z_spacing_mm(n_frames, fallback_z_spacing_mm)
    return z, cycles_per_mm * z


def hfer_from_signal(counts: np.ndarray, cutoff: float, min_ac_power: float = 0.0) -> float:
    """HFER (%) of one class's per-frame pixel counts; `cutoff` in cycles/frame. NaN if the class
    never appears, the AC power is <= min_ac_power (e.g. a constant signal), or the cutoff is
    above Nyquist (0.5 cycles/frame, so there is no high-frequency band to measure)."""
    peak = counts.max()
    if peak <= 0 or cutoff > 0.5:
        return float("nan")
    signal = counts.astype(float) / peak
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal))
    total_ac_power = power[1:].sum()  # exclude DC
    if total_ac_power <= min_ac_power:
        return float("nan")
    return float(power[freqs >= cutoff].sum() / total_ac_power * 100)


def pixels_per_frame(vol: np.ndarray, n_classes: int = _N_CLASSES) -> np.ndarray:
    """vol: [D,H,W] non-negative integer label volume. Returns [D, n_classes] pixel counts per frame."""
    flat = vol.reshape(vol.shape[0], -1)
    return np.stack([np.bincount(frame, minlength=n_classes)[:n_classes] for frame in flat])


def hfer_from_counts(counts: np.ndarray, cutoff: float, min_ac_power: float = 0.0,
                     class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """counts: [D, n_classes] from pixels_per_frame(). Returns {class_name: HFER %}, background skipped."""
    class_names = class_names or _CLASS_NAMES_12
    return {
        name: hfer_from_signal(counts[:, c], cutoff, min_ac_power)
        for c, name in enumerate(class_names) if c > 0  # skip background
    }


# ---------------------------------------------------------------------------
# Helpers shared by Pullback_prediction.py / _unet.py / _unet3d.py
# ---------------------------------------------------------------------------
_DEFAULT_EXCLUSION_EXCEL = str(Path(__file__).resolve().parent.parent / "Continuous_3D_Metrics"
                               / "Exclusion_Frames_testset.xlsx")


def parse_frame_ranges(value) -> List[tuple]:
    """'1-88, 460-540' / '217' / 217 / NaN / -99 -> [(1, 88), (460, 540)] (1-based, inclusive).
    NaN and the -99 'none' sentinel give []."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if text in ("", "-99", "-99.0", "nan"):
        return []
    ranges = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)(?:\.0)?\s*(?:-\s*(\d+)(?:\.0)?)?", token)
        if not m:
            raise ValueError(f"Cannot parse frame range {token!r} (from {value!r})")
        start = int(m.group(1))
        ranges.append((start, int(m.group(2)) if m.group(2) else start))
    return ranges


def load_exclusions(excel_path: Optional[str]) -> Dict[str, List[tuple]]:
    """{pullback: [(start, end), ...]} (1-based, inclusive) = union of the sheet's 'guiding' and
    'artifact' columns -- same file/format as Continuous_3D_metrics.py's --exclusion_excel.
    Empty/None path -> {} (no exclusion)."""
    if not excel_path:
        return {}
    if not Path(excel_path).exists():
        print(f"[WARN] HFER exclusion Excel not found: {excel_path} -- NO frames excluded")
        return {}
    df = pd.read_excel(excel_path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["pullback"]).strip()] = parse_frame_ranges(row.get("guiding")) + parse_frame_ranges(row.get("artifact"))
    print(f"[INFO] HFER: excluding guiding-catheter + artifact frames from {excel_path} ({len(out)} pullbacks listed)")
    return out


def hfer_excluding(counts: np.ndarray, excluded: np.ndarray, cutoff: float, min_ac_power: float,
                   class_names: List[str], min_stretch_frames: int):
    """HFER per class over only the non-excluded frames. The FFT needs evenly spaced frames, so the
    volume is split into unbroken stretches of kept frames; each stretch >= min_stretch_frames is
    scored on its own and the stretches are averaged weighted by length (per class, skipping NaN).
    With nothing excluded this is one stretch = the whole volume. Returns (hfer dict, n_frames_used,
    n_stretches)."""
    keep = ~excluded
    stretches, start = [], None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        elif not k and start is not None:
            stretches.append((start, i))
            start = None
    if start is not None:
        stretches.append((start, len(keep)))
    stretches = [(s, e) for s, e in stretches if e - s >= min_stretch_frames]

    acc = {name: [] for name in class_names[1:]}  # (hfer, length)
    for s, e in stretches:
        for name, h in hfer_from_counts(counts[s:e], cutoff, min_ac_power, class_names).items():
            if not np.isnan(h):
                acc[name].append((h, e - s))
    result = {name: (sum(h * w for h, w in v) / sum(w for _, w in v)) if v else float("nan")
              for name, v in acc.items()}
    return result, sum(e - s for s, e in stretches), len(stretches)


def add_hfer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exclusion_excel", type=str, default=_DEFAULT_EXCLUSION_EXCEL,
                        help="Excel with 'pullback', 'guiding', 'artifact' frame ranges (1-based, '-99' = none), as "
                             "in Continuous_3D_Metrics/Exclusion_Frames_testset.xlsx (the default). Those frames are "
                             "left out of HFER (predictions there are wrong by definition); pixel counts are still "
                             "saved for every frame. Pass an empty string to exclude nothing.")
    parser.add_argument("--hfer_min_stretch_frames", type=int, default=32,
                        help="Unbroken stretches of kept frames shorter than this are skipped (coarse spectrum)")
    parser.add_argument("--skip_hfer", action="store_true",
                        help="Skip the High Frequency Energy Ratio (HFER) of per-class area along each full "
                             "predicted pullback (on by default; cheap, see HFER.py)")
    parser.add_argument("--hfer_threshold_cycles_per_mm", type=float, default=1.0,
                        help="HFER high-frequency cutoff in cycles/mm, converted per pullback to cycles/frame "
                             "via its z-spacing (inferred from frame count). 1.0 = 0.1 cycles/frame at 0.1 mm.")
    parser.add_argument("--hfer_min_ac_power", type=float, default=0.0,
                        help="HFER is NaN when a class's total AC power is <= this (default 0: only exactly "
                             "constant signals)")


def compute_pullback_hfer(pred: np.ndarray, pid: str, class_names: List[str], cycles_per_mm: float,
                          min_ac_power: float, fallback_z_spacing_mm: float,
                          exclusions: Optional[Dict[str, List[tuple]]] = None, min_stretch_frames: int = 32,
                          z_spacing_override_mm: Optional[float] = None):
    """pred: full predicted volume [D,H,W] (postprocessed if the run postprocesses).
    exclusions: from load_exclusions(); frames listed for `pid` are left out of HFER.
    Returns (hfer_row, pixels_per_frame_df) for one pullback. The pixel table keeps EVERY frame, with an
    Inclusion column (0 = excluded from HFER)."""
    n_frames = pred.shape[0]
    counts = pixels_per_frame(pred, len(class_names))
    z, cutoff = cutoff_cycles_per_frame(n_frames, cycles_per_mm, fallback_z_spacing_mm, z_spacing_override_mm)

    excluded = np.zeros(n_frames, dtype=bool)
    if exclusions:
        if pid not in exclusions:
            print(f"[WARN] HFER: {pid} not listed in the exclusion Excel -- using all {n_frames} frames")
        for start, end in exclusions.get(pid, []):
            excluded[max(start - 1, 0):min(end, n_frames)] = True  # 1-based inclusive -> 0-based slice

    hfer, n_used, n_stretches = hfer_excluding(counts, excluded, cutoff, min_ac_power, class_names, min_stretch_frames)
    row = {"pullback": pid, "n_frames": n_frames, "n_frames_used": n_used,
           "n_frames_excluded": int(excluded.sum()), "n_stretches": n_stretches,
           "z_spacing_mm": z, "cutoff_cycles_per_frame": cutoff, **hfer}
    df_px = pd.DataFrame(counts, columns=class_names)
    df_px.insert(0, "Inclusion", (~excluded).astype(int))
    df_px.insert(0, "frame_1based", np.arange(1, n_frames + 1))
    df_px.insert(0, "pullback", pid)
    return row, df_px


def build_hfer_tables(hfer_rows: List[dict], pixel_frames: List[pd.DataFrame], class_names: List[str]):
    """Returns (per-pullback df, overall df, pixels-per-frame df); all None if nothing was computed."""
    if not hfer_rows:
        return None, None, None
    df_hfer = pd.DataFrame(hfer_rows)
    cols = class_names[1:]
    df_overall = pd.DataFrame({
        "class_name": cols,
        "N_pullbacks": [int(df_hfer[c].notna().sum()) for c in cols],
        "HFER_mean_%": [df_hfer[c].mean() for c in cols],
        "HFER_median_%": [df_hfer[c].median() for c in cols],
    })
    return df_hfer, df_overall, pd.concat(pixel_frames, ignore_index=True)


def find_predictions(folder: Path, pattern: str) -> Dict[str, Path]:
    found = {}
    for path in sorted(folder.glob(pattern)):
        m = _ID_PATTERN.match(path.name)
        if not m:
            print(f"  WARNING: cannot extract pullback ID from {path.name}, skipping")
            continue
        found.setdefault(m.group(1), path)
    return found


def score_model(model_name: str, pred_dir: Path, pattern: str, cycles_per_mm: float,
                z_spacing_override_mm: Optional[float], min_ac_power: float,
                exclusions: Optional[Dict[str, List[tuple]]] = None, min_stretch_frames: int = 32) -> List[dict]:
    preds = find_predictions(pred_dir, pattern)
    if not preds:
        print(f"  WARNING: no files matching '{pattern}' in {pred_dir}")
    rows = []
    for pid, path in preds.items():
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # [D,H,W], native small int dtype (no int64 copy)
        if vol.min() < 0 or vol.max() >= _N_CLASSES:
            print(f"  WARNING: {path.name} has labels outside 0..{_N_CLASSES - 1} "
                  f"(min {vol.min()}, max {vol.max()}) -- not the 12-class taxonomy, skipping")
            continue
        row, _ = compute_pullback_hfer(vol, pid, _CLASS_NAMES_12, cycles_per_mm, min_ac_power, 0.1,
                                       exclusions, min_stretch_frames, z_spacing_override_mm)
        row = {"model": model_name, **row}
        rows.append(row)
        print(f"  {model_name} / {pid} ({row['n_frames_used']}/{vol.shape[0]} frames used, "
              f"z={row['z_spacing_mm']:.3f} mm, cutoff={row['cutoff_cycles_per_frame']:.3f} cycles/frame)")
    return rows


def boxplot_table(df: pd.DataFrame, class_cols: List[str]) -> pd.DataFrame:
    """Long-form table for an Excel Box and Whisker chart: column A = class, then one column per model
    (one row per pullback, blank where the class is absent), pullback LAST so A..N is contiguous."""
    long = df.melt(id_vars=["model", "pullback"], value_vars=class_cols,
                   var_name="class", value_name="HFER").dropna(subset=["HFER"])
    wide = long.pivot_table(index=["class", "pullback"], columns="model", values="HFER", aggfunc="mean")
    wide = wide.reset_index()
    wide["class"] = pd.Categorical(wide["class"], categories=class_cols, ordered=True)
    wide = wide.sort_values(["class", "pullback"]).reset_index(drop=True)
    wide["class"] = wide["class"].astype(str)
    model_cols = [c for c in wide.columns if c not in ("class", "pullback")]
    return wide[["class"] + model_cols + ["pullback"]]


def main():
    ap = argparse.ArgumentParser(description="High Frequency Energy Ratio (HFER) per class per pullback.")
    ap.add_argument("--from_excel", default=None,
                    help="Existing hfer_<date>.xlsx: skip scoring and only add the 'Boxplot data' sheet "
                         "(written to <name>_boxplot.xlsx next to it)")
    ap.add_argument("--pred_dirs", nargs="+", help="Folder(s) of full-pullback prediction .nii.gz")
    ap.add_argument("--model_names", nargs="+", help="One name per --pred_dirs entry")
    ap.add_argument("--pattern", default="*.nii.gz", help="Glob for prediction files inside each folder")
    ap.add_argument("--threshold_cycles_per_mm", type=float, default=1.0,
                    help="High-frequency cutoff in cycles/mm, converted per pullback to cycles/frame via its "
                         "z-spacing. Default 1.0 (= the reference's 0.1 cycles/frame at 0.1 mm spacing).")
    ap.add_argument("--z_spacing_mm", type=float, default=None,
                    help="Force one z-spacing for every pullback instead of inferring it from frame count.")
    ap.add_argument("--min_ac_power", type=float, default=0.0,
                    help="Return NaN when total AC power is <= this (guards near-constant signals). "
                         "Default 0 = only exactly-constant signals are NaN, as in the reference.")
    ap.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "HFER_output"))
    ap.add_argument("--exclusion_excel", type=str, default=_DEFAULT_EXCLUSION_EXCEL,
                    help="Guiding/artifact frame ranges to leave out (default: the repo's Exclusion_Frames_testset.xlsx; "
                         "empty string = exclude nothing)")
    ap.add_argument("--min_stretch_frames", type=int, default=32,
                    help="Unbroken stretches of kept frames shorter than this are skipped")
    args = ap.parse_args()

    if args.from_excel:
        src = Path(args.from_excel)
        old = pd.read_excel(src, sheet_name="HFER per pullback")
        cols = [c for c in _CLASS_NAMES_12 if c != "Background" and c in old.columns]
        out = src.with_name(src.stem + "_boxplot.xlsx")
        boxplot_table(old, cols).to_excel(out, sheet_name="Boxplot data", index=False)
        print(f"Saved: {out}")
        return
    if not args.pred_dirs or not args.model_names:
        ap.error("--pred_dirs and --model_names are required (unless --from_excel is used)")
    if len(args.pred_dirs) != len(args.model_names):
        ap.error("--pred_dirs and --model_names must have the same length")
    if args.threshold_cycles_per_mm <= 0:
        ap.error("--threshold_cycles_per_mm must be > 0")

    exclusions = load_exclusions(args.exclusion_excel)
    rows = []
    for name, folder in zip(args.model_names, args.pred_dirs):
        print(f"Scoring {name}: {folder}")
        rows += score_model(name, Path(folder), args.pattern, args.threshold_cycles_per_mm,
                            args.z_spacing_mm, args.min_ac_power, exclusions, args.min_stretch_frames)
    if not rows:
        raise SystemExit("No pullbacks scored.")

    df = pd.DataFrame(rows)
    class_cols = [c for c in _CLASS_NAMES_12 if c != "Background"]
    summary = df.groupby("model")[class_cols].agg(["mean", "median", "count"])  # NaN skipped

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hfer_{datetime.now():%d_%m_%Y}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="HFER per pullback", index=False)
        summary.to_excel(writer, sheet_name="Summary per model")
        boxplot_table(df, class_cols).to_excel(writer, sheet_name="Boxplot data", index=False)
        pd.DataFrame({"threshold_cycles_per_mm": [args.threshold_cycles_per_mm],
                      "z_spacing_override_mm": [args.z_spacing_mm],
                      "min_ac_power": [args.min_ac_power],
                      "pattern": [args.pattern]}).to_excel(writer, sheet_name="Settings", index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
