#!/usr/bin/env python3
"""
Continuous per-frame 3D metrics across whole pullbacks, for one or more model
prediction folders -- e.g. Conditional INR vs. 3D U-Net vs. OCTAID vs. MLP.

Unlike Metrics/Pullback_prediction.py (which only scores the sparse annotated
frames against ground truth), this script does NOT need ground truth or a
model checkpoint. It consumes already-written full-pullback prediction
volumes (*.nii.gz, one per pullback, 12-class post-mapping taxonomy) and
derives, for EVERY frame of EVERY pullback:
  - per-class area (mm^2)
  - lipid arc (deg) and fibrous cap thickness (FCT, um)
  - calcium arc (deg), depth (um), thickness (um)
via nnunetv2/Codes/utils/plaque_quantification.py's quantification_lipid()/
quantification_calcium() (reused as-is, same as Pullback_prediction.py).

On top of the continuous FCT/lipid-arc curves, it derives TCFA (thin-cap
fibroatheroma) presence: FCT < --tcfa_fct_threshold_um AND lipid arc >=
--tcfa_lipid_arc_threshold_deg, sustained over >= --tcfa_min_consecutive_frames
consecutive frames (the "continuity" requirement -- a single flickering frame
does not count as a lesion).

Optionally, --exclusion_excel points at a sheet with "pullback", "guiding",
"artifact" columns (frame ranges per pullback, e.g. "12-100", comma-separated
for multiple ranges). Frames falling in those ranges are still written as a
row (Inclusion=0, Exclusion_Reason="Guiding"/"Artifact") but get no computed
values -- they are excluded from all metrics, TCFA lesion detection, and
summary means (which skip NaN). When --exclusion_excel is given, this also
writes a visualization-ready copy of every prediction volume under
<output_dir>/masked_predictions/<model_name>/ -- excluded frames blanked
entirely to background (0), everything else untouched -- so a viewer shows no
prediction at all on guiding-catheter/artifact frames. Reuses the volume
already loaded for the metrics themselves rather than re-reading the file.

Multiple model folders are analysed in one run via repeated --pred_dirs /
--model_names, so results across models land in the same long-format tables
directly (a "model" column), rather than requiring separate runs to be
stitched together by hand. Pullback-ID matching is filename-prefix-based
(everything before the first "_pred..." token) since different model
pipelines use different suffix conventions
(e.g. "{pid}_predictions.nii.gz" vs "{pid}_pred_overlap0.5.nii.gz").

Usage:
    python Continuous_3D_metrics.py \
        --pred_dirs   /path/to/conditional_inr/predictions /path/to/octaid/predictions \
        --model_names ConditionalINR OCTAID \
        [--label_file /path/to/label_description.txt] \
        [--env local|cluster] \
        [--tcfa_fct_threshold_um 65.0] \
        [--tcfa_lipid_arc_threshold_deg 90.0] \
        [--tcfa_min_consecutive_frames 3] \
        [--output_dir /path/to/output]
"""
import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Same dual local/cluster sys.path pattern as Metrics/Pullback_prediction.py --
# whichever path exists on the current machine resolves the import.
sys.path.insert(1, r"W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
sys.path.insert(1, "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
from plaque_quantification import quantification_lipid, quantification_calcium, load_label_map

_DEFAULT_LABEL_FILE = os.path.join(REPO_ROOT, "Metrics", "label_description_conditionalinr.txt")

# Canonical 12-class post-mapping taxonomy shared across the model pipelines
# (see Metrics/Pullback_prediction.py's _CLASS_NAMES_12 / label_description_conditionalinr.txt).
_CLASS_NAMES_12 = [
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media",
    "Sidebranch", "Thrombus", "Plaque rupture", "Layered plaque", "Neovascularization",
]

# Pullback ID = everything before the first "_pred" token in the filename stem.
# Covers both "{pid}_predictions.nii.gz" (nnU-Net/OCTAID family) and
# "{pid}_pred_overlap0.5[_postprocessed].nii.gz" (Conditional INR) without
# needing a per-folder naming convention.
_DEFAULT_ID_PATTERN = re.compile(r"^(.+?)_pred", re.IGNORECASE)


def _strip_niigz(name: str) -> str:
    return name[:-len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def find_predictions(folder: Path, id_pattern: re.Pattern,
                      exclude_substrings: List[str] = ("_raw",),
                      prefer_substrings: List[str] = ("_postprocessed",)) -> Dict[str, Path]:
    """Maps pullback_id -> .nii.gz path for every prediction file in `folder`.

    Some pipelines write more than one file per pullback. `exclude_substrings` drops
    those outright (e.g. UNET3D's pre-postprocessing "{pid}_predictions_raw.nii.gz").
    If a pid still has more than one candidate afterwards (e.g. Conditional INR writes
    both "{pid}_predictions_overlap0.5.nii.gz" and "..._postprocessed.nii.gz" side by
    side), whichever contains a `prefer_substrings` match wins; if that's still not
    unique, raises rather than silently guessing."""
    candidates: Dict[str, List[Path]] = {}
    for f in sorted(folder.glob("*.nii.gz")):
        if any(sub.lower() in f.name.lower() for sub in exclude_substrings):
            continue
        stem = _strip_niigz(f.name)
        m = id_pattern.match(stem)
        pid = m.group(1) if m else stem
        candidates.setdefault(pid, []).append(f)

    out: Dict[str, Path] = {}
    for pid, files in candidates.items():
        if len(files) == 1:
            out[pid] = files[0]
            continue
        preferred = [f for f in files if any(sub.lower() in f.name.lower() for sub in prefer_substrings)]
        if len(preferred) == 1:
            out[pid] = preferred[0]
        else:
            names = ", ".join(f.name for f in files)
            raise ValueError(f"Ambiguous pullback id '{pid}' in {folder}: {names} -- narrow --id_pattern, "
                              f"--exclude_substrings, or --prefer_substrings")
    return out


# Same raw-class merges label_description_conditionalinr.txt's header documents as already
# baked into the canonical 12-class taxonomy (Training_model/dataset.py _DEFAULT_LABEL_MAPPING):
# Catheter -> Lumen, {Red,White} Thrombus -> the single "Red Thrombus" canonical class,
# Dissection/Intraplaque hemorrhage/Intramural hematoma -> Intima. A source model that still
# keeps these as separate classes (e.g. OCTAIDlite's "White Thrombus"/"Catheter") resolves
# through this table instead of requiring the label file itself to be hand-edited.
_KNOWN_ALIASES = {
    "catheter": "Lumen",
    "white thrombus": "Red Thrombus",
    "thrombus": "Red Thrombus",
    "dissection": "Intima",
    "intraplaque hemorrhage": "Intima",
    "intramural hematoma": "Intima",
}


def build_index_remap(source_label_file: str, canonical_label_file: str) -> Dict[int, int]:
    """Maps a model's OWN raw class indices -> the canonical taxonomy's indices, by
    matching label NAMES between the two ITK-SnAP label files (case-insensitive, falling
    back to _KNOWN_ALIASES for classes the canonical taxonomy already merged elsewhere).
    Used for models (e.g. OCTAIDlite) whose class indices/count differ from the
    canonical 12-class taxonomy every other folder is assumed to already use.
    Raises loudly if a source class has no canonical match, rather than silently
    dropping/miscounting it."""
    src_name_to_idx, _ = load_label_map(source_label_file)
    canon_name_to_idx, _ = load_label_map(canonical_label_file)
    canon_lower = {name.lower(): idx for name, idx in canon_name_to_idx.items()}
    remap: Dict[int, int] = {}
    for name, src_idx in src_name_to_idx.items():
        canon_idx = canon_lower.get(name.lower())
        if canon_idx is None:
            alias = _KNOWN_ALIASES.get(name.lower())
            if alias:
                canon_idx = canon_lower.get(alias.lower())
        if canon_idx is None:
            raise ValueError(f"Class '{name}' (idx {src_idx}) in {source_label_file} has no name match (and no "
                              f"known alias) in canonical label file {canonical_label_file} -- add an entry to "
                              f"_KNOWN_ALIASES or rename it to what the canonical file calls it before this "
                              f"folder can be processed")
        remap[src_idx] = canon_idx
    return remap


def remap_volume(vol: np.ndarray, remap: Dict[int, int]) -> np.ndarray:
    """Preserves `vol`'s own dtype (e.g. int16) -- fancy-indexing with a LUT returns an
    array typed like the LUT, not like `vol`, so a hardcoded LUT dtype would silently
    upcast every remapped volume (confirmed: this previously forced int64/"long long"
    output, which ITK-SnAP refuses to open -- 'Error unsupported voxel type (long long)')."""
    lut_size = max(int(vol.max()), max(remap.keys())) + 1
    lut = np.arange(lut_size, dtype=vol.dtype)
    for old, new in remap.items():
        lut[old] = new
    return lut[vol]


def _parse_ranges(value) -> List["tuple[int, int]"]:
    """Parses a cell like '12-100' or '12-100, 150-160' (or a bare '42') into
    inclusive 1-based (start, end) frame tuples. Blank/NaN -> no ranges. A bare
    negative number (e.g. '-99', the same "not applicable" sentinel
    plaque_quantification.py itself uses) also means no ranges -- checked via a
    plain int() parse FIRST, before the dash-as-range-separator logic, so '-99'
    isn't misread as a malformed range with an empty start ('', '99')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    ranges = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            f = int(part)  # bare integer, including a leading '-' (e.g. -99 sentinel)
        except ValueError:
            pass
        else:
            if f >= 0:
                ranges.append((f, f))
            continue  # negative -> sentinel for "not applicable", no range added
        if "-" not in part:
            raise ValueError(f"Could not parse frame range/number: {part!r}")
        a, b = part.split("-", 1)
        ranges.append((int(a.strip()), int(b.strip())))
    return ranges


def load_exclusions(path: str) -> Dict[str, Dict[int, str]]:
    """Returns {pullback_id: {frame_1based: reason}} for frames to exclude
    (guiding catheter / artifact), read from an excel with "pullback",
    "guiding", "artifact" columns."""
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "pullback" in df.columns, "--exclusion_excel must contain a 'pullback' column"
    reason_cols = [c for c in ("guiding", "artifact") if c in df.columns]
    if not reason_cols:
        raise ValueError("--exclusion_excel must contain a 'guiding' and/or 'artifact' column")

    out: Dict[str, Dict[int, str]] = {}
    for _, row in df.iterrows():
        pid = str(row["pullback"])
        frame_reasons: Dict[int, str] = {}
        for col in reason_cols:
            for start, end in _parse_ranges(row.get(col)):
                for f in range(start, end + 1):
                    label = col.capitalize()
                    frame_reasons[f] = f"{frame_reasons[f]}, {label}" if f in frame_reasons else label
        out[pid] = frame_reasons
    return out


def infer_z_spacing_mm(n_frames: int, fallback: float) -> float:
    """OCT pullback z-spacing is NOT one fixed value across the dataset -- different
    pullback-length protocols run at different frame rates. Same heuristic as the
    DICOM-preprocessing pipeline's z-spacing inference (there it prefers DICOM tag
    0018|0050 "Spacing Between Slices" when present, falling back to this), applied here
    directly to a prediction volume's frame count since Continuous_3D_metrics.py never
    has the source DICOM in hand -- it only ever reads already-written prediction
    .nii.gz files. Known brackets: ~75mm pullbacks run at 350-400 frames; ~54mm pullbacks
    run at either of two frame rates, 250-300 or 500-550 frames (same physical length,
    different resolution). Falls back to `fallback` (with a warning) for any frame count
    outside these three ranges, rather than silently guessing."""
    if 350 <= n_frames <= 400:
        return 75.0 / n_frames
    if (250 <= n_frames <= 300) or (500 <= n_frames <= 550):
        return 54.0 / n_frames
    print(f"[WARN] Frame count {n_frames} doesn't match a known pullback-length bracket "
          f"(350-400 -> 75mm, 250-300/500-550 -> 54mm) -- using fallback z_spacing={fallback}mm")
    return fallback


def _parse_quant(value) -> float:
    """quantification_lipid/calcium return a numeric string, or the literal int -99
    (their sentinel for 'structure not present in this frame'). -99 -> NaN."""
    v = float(value)
    return float("nan") if v <= -99 else v


def per_frame_metrics(pid: str, path: Path, label_file: str, quant_font: str,
                       fct_thr: float, arc_thr: float,
                       excluded: Optional[Dict[int, str]] = None,
                       remap: Optional[Dict[int, int]] = None,
                       xy_spacing_override: Optional[float] = None,
                       z_spacing_override: Optional[float] = None,
                       mask_output_path: Optional[Path] = None) -> pd.DataFrame:
    """One row per frame of the pullback: per-class area (mm^2) + continuous
    plaque-quantification metrics, computed over EVERY frame (not just annotated ones).
    Frames listed in `excluded` (guiding catheter / artifact) get Inclusion=0 and no
    computed values -- nothing is measured or quantified for them.

    Is_TCFA_Frame is evaluated per frame here, independent of lesion grouping/length --
    a single isolated TCFA+ frame is always True, never hidden by a continuity requirement
    (that requirement only applies later, to which lesions count as "significant").

    `remap` (from build_index_remap), if given, is applied right after loading -- before
    area counts or quantification -- so a model with a different raw taxonomy (e.g.
    OCTAIDlite's 14 classes) is converted to the canonical indices first, and everything
    downstream is identical to any other folder.

    `xy_spacing_override`/`z_spacing_override`, if given, replace whatever spacing is (or
    isn't) baked into the .nii.gz header. Confirmed necessary in practice: only Conditional
    INR's own writer sets correct spacing -- OCTAID/OCTAIDlite/UNET3D's .nii.gz files all
    carry the SimpleITK default identity spacing (1,1,1)mm, which silently inflated FCT/
    area/length by ~100x for those three until this override was added.

    `mask_output_path`, if given, writes a visualization-ready copy of the (already
    remapped/re-spaced) volume alongside the metrics -- every frame in `excluded` blanked
    to background (0), everything else untouched -- reusing the volume already in memory
    here rather than re-reading the file a second time. Written with the resolved
    xy_spacing/z_spacing (override or original) baked in as correct spacing metadata."""
    excluded = excluded or {}
    img = sitk.ReadImage(str(path))
    vol = sitk.GetArrayFromImage(img).astype(np.int16)  # [D,H,W]
    if remap:
        vol = remap_volume(vol, remap)
    sx, sy, sz = img.GetSpacing()  # (x,y,z), matches (W,H,D) axis order
    xy_spacing = xy_spacing_override if xy_spacing_override is not None else (sx + sy) / 2.0
    if z_spacing_override is not None:
        sz = z_spacing_override
    else:
        # Header sz is passed as the fallback -- correct if some future model's writer
        # sets it properly and the frame count doesn't match a known bracket; broken
        # (1,1,1)-spacing headers just make the fallback less good, not silently wrong,
        # since the two known brackets are checked first regardless of header quality.
        sz = infer_z_spacing_mm(vol.shape[0], fallback=sz)
    px_area_mm2 = xy_spacing * xy_spacing

    rows = []
    for z in range(vol.shape[0]):
        frame_1based = z + 1
        reason = excluded.get(frame_1based)
        row = {
            "pullback": pid, "frame_1based": frame_1based, "z_spacing_mm": sz,
            "Inclusion": 0 if reason else 1, "Exclusion_Reason": reason or "",
        }
        if reason:
            for name in _CLASS_NAMES_12:
                row[f"Area_{name}_mm2"] = float("nan")
            row["FCT_um"] = float("nan")
            row["Lipid_Arc_deg"] = float("nan")
            row["Calcium_Depth_um"] = float("nan")
            row["Calcium_Arc_deg"] = float("nan")
            row["Calcium_Thickness_um"] = float("nan")
            row["Is_TCFA_Frame"] = False
            rows.append(row)
            continue

        frame = vol[z]
        counts = np.bincount(frame.ravel(), minlength=len(_CLASS_NAMES_12))
        for c, name in enumerate(_CLASS_NAMES_12):
            row[f"Area_{name}_mm2"] = float(counts[c]) * px_area_mm2 if c < len(counts) else 0.0

        _, _, fct, lipid_arc, _ = quantification_lipid(
            frame, label_file=label_file, xy_spacing=xy_spacing,
            font=quant_font, filename=f"{pid}_frame{frame_1based:04d}")
        _, _, ca_depth, ca_arc, ca_thick, _ = quantification_calcium(
            frame, label_file=label_file, xy_spacing=xy_spacing,
            font=quant_font, filename=f"{pid}_frame{frame_1based:04d}")

        row["FCT_um"] = _parse_quant(fct)
        row["Lipid_Arc_deg"] = _parse_quant(lipid_arc)
        row["Calcium_Depth_um"] = _parse_quant(ca_depth)
        row["Calcium_Arc_deg"] = _parse_quant(ca_arc)
        row["Calcium_Thickness_um"] = _parse_quant(ca_thick)
        row["Is_TCFA_Frame"] = bool(row["FCT_um"] < fct_thr and row["Lipid_Arc_deg"] >= arc_thr) \
            if not (np.isnan(row["FCT_um"]) or np.isnan(row["Lipid_Arc_deg"])) else False
        rows.append(row)

    if mask_output_path is not None:
        # uint8 explicitly, regardless of vol's own dtype -- 12 classes fit trivially in
        # 0-255, and this guarantees a type every NIfTI viewer (ITK-SnAP included) can
        # open. ITK-SnAP refuses "long long" (int64) outright ("Error unsupported voxel
        # type"); don't rely on vol/remap_volume happening to already be narrow enough.
        masked_vol = vol.copy().astype(np.uint8)
        for frame_1based in excluded:
            z = frame_1based - 1
            if 0 <= z < masked_vol.shape[0]:
                masked_vol[z] = 0
        out_img = sitk.GetImageFromArray(masked_vol)
        out_img.SetSpacing((xy_spacing, xy_spacing, sz))
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out_img, str(mask_output_path))

    return pd.DataFrame(rows)


def _process_one(job: dict) -> Tuple[str, str, pd.DataFrame, float]:
    """One (model, pullback) work unit -- must be a top-level function (not a closure)
    so ProcessPoolExecutor can pickle it, including on Windows (spawn re-imports this
    module in each worker, hence the `if __name__ == "__main__"` guard at the bottom)."""
    t0 = time.time()
    df_pb = per_frame_metrics(
        job["pid"], job["path"], job["label_file"], job["quant_font"],
        job["fct_thr"], job["arc_thr"], job["excluded"], job["remap"],
        job["xy_spacing_override"], job["z_spacing_override"],
        job["mask_output_path"],
    )
    df_pb.insert(0, "model", job["model_name"])
    return job["model_name"], job["pid"], df_pb, time.time() - t0


def find_tcfa_lesions(df_pb: pd.DataFrame, min_frames: int) -> List[dict]:
    """Groups consecutive TCFA-positive frames (Is_TCFA_Frame, within one model+pullback)
    into lesions. This is a summary/grouping view only -- it never changes whether an
    individual frame counts as TCFA+ (that's decided once, per frame, in per_frame_metrics)."""
    df_pb = df_pb.sort_values("frame_1based")
    frames = df_pb["frame_1based"].to_numpy()
    z_spacing = df_pb["z_spacing_mm"].iloc[0] if len(df_pb) else float("nan")
    is_tcfa = df_pb["Is_TCFA_Frame"].to_numpy()

    lesions = []
    i = 0
    n = len(is_tcfa)
    while i < n:
        if not is_tcfa[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and is_tcfa[j + 1]:
            j += 1
        n_frames = j - i + 1
        lesions.append({
            "start_frame": int(frames[i]),
            "end_frame": int(frames[j]),
            "n_frames": n_frames,
            "length_mm": n_frames * z_spacing,
            "min_FCT_um": float(df_pb["FCT_um"].iloc[i:j + 1].min()),
            "max_Lipid_Arc_deg": float(df_pb["Lipid_Arc_deg"].iloc[i:j + 1].max()),
            "meets_min_length": n_frames >= min_frames,
        })
        i = j + 1
    return lesions


def main():
    parser = argparse.ArgumentParser("Continuous per-frame 3D metrics across full pullbacks, multi-model")
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True,
                         help="One or more folders, each containing one *.nii.gz prediction per pullback")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                         help="Label per folder, same order/length as --pred_dirs (default: folder basename)")
    parser.add_argument("--label_file", type=str, default=_DEFAULT_LABEL_FILE,
                         help="ITK-SnAP label description for the CANONICAL 12-class post-mapping taxonomy "
                              "(default: Metrics/label_description_conditionalinr.txt). All output (Area_* "
                              "columns, quantification) uses this taxonomy, regardless of --label_files below")
    parser.add_argument("--label_files", type=str, nargs="+", default=None,
                         help="Per-folder ITK-SnAP label file (parallel to --pred_dirs), describing that "
                              "folder's OWN raw class indices/names, ONLY needed for a folder whose taxonomy "
                              "differs from --label_file (e.g. OCTAIDlite's 14-class output). That folder's "
                              "volumes are remapped by NAME to the canonical taxonomy before any metric is "
                              "computed. Pass '' for folders that are already canonical")
    parser.add_argument("--id_pattern", type=str, default=None,
                         help="Regex overriding pullback-ID extraction from filenames (default matches "
                              "everything before the first '_pred' token)")
    parser.add_argument("--exclude_substrings", type=str, nargs="+", default=["_raw"],
                         help="Skip prediction files whose name contains any of these substrings (default: "
                              "'_raw', e.g. UNET3D's '{pid}_predictions_raw.nii.gz' pre-postprocessing dumps)")
    parser.add_argument("--prefer_substrings", type=str, nargs="+", default=["_postprocessed"],
                         help="If a pullback still has multiple candidate files after --exclude_substrings "
                              "(e.g. Conditional INR's '..._overlap0.5.nii.gz' + '..._postprocessed.nii.gz' "
                              "side by side), the one matching one of these substrings wins (default: "
                              "'_postprocessed')")
    parser.add_argument("--env", type=str, default=None, choices=["local", "cluster"],
                         help="Selects plaque_quantification.py's font path. Default: auto-detect from "
                              "which nnunetv2 utils path exists on this machine")
    parser.add_argument("--xy_spacing_mm", type=float, default=None,
                         help="Override in-plane pixel spacing (mm/px) instead of trusting each .nii.gz's own "
                              "header. Use this when a folder's writer doesn't set correct spacing (confirmed "
                              "for OCTAID/OCTAIDlite/UNET3D here -- their files carry the SimpleITK default "
                              "identity spacing (1,1,1)mm, which silently inflates FCT/area by ~100x if trusted)")
    parser.add_argument("--z_spacing_mm", type=float, default=None,
                         help="Force ONE frame spacing (mm/frame) for every pullback, overriding both the "
                              "_.nii.gz header and the frame-count-bracket inference below. Leave unset "
                              "(default) unless you specifically want a single fixed value -- z-spacing "
                              "genuinely varies by pullback-length protocol (75mm pullbacks run at 350-400 "
                              "frames, 54mm pullbacks at 250-300 or 500-550 frames), so a single constant is "
                              "wrong for part of the dataset. When unset, per-pullback frame count picks the "
                              "right bracket automatically (see infer_z_spacing_mm), falling back to the "
                              ".nii.gz header only for a frame count outside all three known brackets")
    parser.add_argument("--tcfa_fct_threshold_um", type=float, default=65.0)
    parser.add_argument("--tcfa_lipid_arc_threshold_deg", type=float, default=90.0)
    parser.add_argument("--tcfa_min_consecutive_frames", type=int, default=1,
                         help="Minimum run length (frames) of sustained FCT+arc criteria for a lesion to count "
                              "towards Per_Pullback_Summary's n_TCFA_lesions/total_TCFA_length_mm. Default 1 = no "
                              "continuity filter -- every run is kept in TCFA_Lesions (with its own n_frames/"
                              "length_mm) regardless, so a stricter cutoff can be applied post-hoc without rerunning")
    parser.add_argument("--exclusion_excel", type=str, default=None,
                         help="Optional excel with 'pullback', 'guiding', 'artifact' columns (frame ranges per "
                              "pullback, e.g. '12-100', comma-separated for multiple). Frames in those ranges are "
                              "still written as a row (Inclusion=0, Exclusion_Reason set) but get no computed values")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Default: Continuous_3D_Metrics/output next to this script")
    parser.add_argument("--max_pullbacks", type=int, default=None, help="Only process the first N pullbacks per folder (debugging)")
    parser.add_argument("--n_procs", type=int, default=1,
                         help="Number of (model, pullback) jobs to run in parallel (CPU-bound, no GPU used). "
                              "Default 1 = sequential, same behavior as before. Match to --cpus-per-task")
    args = parser.parse_args()

    if args.model_names and len(args.model_names) != len(args.pred_dirs):
        raise ValueError("--model_names must have the same length as --pred_dirs")
    model_names = args.model_names or [Path(d).name for d in args.pred_dirs]

    if args.label_files and len(args.label_files) != len(args.pred_dirs):
        raise ValueError("--label_files must have the same length as --pred_dirs")
    label_files = args.label_files or [""] * len(args.pred_dirs)
    remaps: Dict[str, Optional[Dict[int, int]]] = {}
    for model_name, folder_label_file in zip(model_names, label_files):
        if folder_label_file:
            remaps[model_name] = build_index_remap(folder_label_file, args.label_file)
            print(f"[INFO] {model_name}: remapping {folder_label_file} -> canonical {args.label_file} "
                  f"({len(remaps[model_name])} classes)")
        else:
            remaps[model_name] = None

    id_pattern = re.compile(args.id_pattern, re.IGNORECASE) if args.id_pattern else _DEFAULT_ID_PATTERN

    if args.env:
        quant_font = "mine" if args.env == "local" else "cluster"
    else:
        quant_font = "mine" if os.path.isdir("W:/rubenvdw") else "cluster"

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclusions: Dict[str, Dict[int, str]] = {}
    masked_dir: Optional[Path] = None
    if args.exclusion_excel:
        exclusions = load_exclusions(args.exclusion_excel)
        n_excl_frames = sum(len(v) for v in exclusions.values())
        print(f"[INFO] Loaded exclusions for {len(exclusions)} pullback(s), {n_excl_frames} frame(s) total "
              f"from {args.exclusion_excel}")
        # Masked prediction volumes only get written when an exclusion excel is actually
        # provided -- otherwise there's nothing meaningful to mask. Lands in the same
        # output_dir as the metrics xlsx, one subfolder per model.
        masked_dir = output_dir / "masked_predictions"
        print(f"[INFO] Masked prediction volumes will be written under {masked_dir} "
              f"(every frame in `excluded` blanked to background; pullbacks with no "
              f"exclusions are still copied through, correctly spaced)")

    pids_per_model: Dict[str, set] = {}
    jobs: List[dict] = []

    for folder_str, model_name in zip(args.pred_dirs, model_names):
        folder = Path(folder_str)
        predictions = find_predictions(folder, id_pattern, args.exclude_substrings, args.prefer_substrings)
        if args.max_pullbacks:
            predictions = dict(list(predictions.items())[:args.max_pullbacks])
        print(f"[INFO] {model_name}: {len(predictions)} pullbacks found in {folder}")
        pids_per_model[model_name] = set(predictions.keys())

        for pid, path in predictions.items():
            jobs.append({
                "model_name": model_name, "pid": pid, "path": path,
                "label_file": args.label_file, "quant_font": quant_font,
                "fct_thr": args.tcfa_fct_threshold_um, "arc_thr": args.tcfa_lipid_arc_threshold_deg,
                "excluded": exclusions.get(pid), "remap": remaps[model_name],
                "xy_spacing_override": args.xy_spacing_mm, "z_spacing_override": args.z_spacing_mm,
                "mask_output_path": (masked_dir / model_name / path.name) if masked_dir else None,
            })

    if not jobs:
        print("[ERROR] No pullbacks were processed -- nothing to write.")
        return

    print(f"[INFO] {len(jobs)} (model, pullback) job(s) total, running with --n_procs {args.n_procs}")
    frame_dfs: List[pd.DataFrame] = []
    if args.n_procs > 1:
        # Each job is fully independent (own file, own frames, no shared state) -- runs
        # as-completed rather than in submission order, so progress numbers/timings below
        # reflect actual finish order, not the original per-folder listing order.
        with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
            futures = [executor.submit(_process_one, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                model_name, pid, df_pb, elapsed = future.result()
                frame_dfs.append(df_pb)
                print(f"[{i}/{len(jobs)}] {model_name} {pid}: {len(df_pb)} frames ({elapsed:.1f}s)")
    else:
        for i, job in enumerate(jobs, 1):
            model_name, pid, df_pb, elapsed = _process_one(job)
            frame_dfs.append(df_pb)
            print(f"[{i}/{len(jobs)}] {model_name} {pid}: {len(df_pb)} frames ({elapsed:.1f}s)")

    df_frames = pd.concat(frame_dfs, ignore_index=True)

    # --- TCFA lesions, per model+pullback ---
    lesion_rows = []
    for (model_name, pid), g in df_frames.groupby(["model", "pullback"], sort=False):
        for lesion in find_tcfa_lesions(g, args.tcfa_min_consecutive_frames):
            lesion.update({"model": model_name, "pullback": pid})
            lesion_rows.append(lesion)
    df_lesions = pd.DataFrame(lesion_rows)
    if not df_lesions.empty:
        df_lesions = df_lesions[[
            "model", "pullback", "start_frame", "end_frame", "n_frames", "length_mm",
            "min_FCT_um", "max_Lipid_Arc_deg", "meets_min_length",
        ]]

    # --- Per-pullback summary, per model (all pullbacks, not just the common ones) ---
    area_cols = [c for c in df_frames.columns if c.startswith("Area_")]
    summary_rows = []
    for (model_name, pid), g in df_frames.groupby(["model", "pullback"], sort=False):
        lesions_pb = df_lesions[(df_lesions["model"] == model_name) & (df_lesions["pullback"] == pid)] \
            if not df_lesions.empty else pd.DataFrame()
        sig_lesions = lesions_pb[lesions_pb["meets_min_length"]] if not lesions_pb.empty else lesions_pb
        row = {"model": model_name, "pullback": pid, "n_frames": len(g),
               "n_frames_excluded": int((g["Inclusion"] == 0).sum()),
               "n_TCFA_frames": int(g["Is_TCFA_Frame"].sum())}
        for c in area_cols:
            row[f"Mean_{c}"] = float(g[c].mean())
        row["Mean_FCT_um"] = float(g["FCT_um"].mean())
        row["Mean_Lipid_Arc_deg"] = float(g["Lipid_Arc_deg"].mean())
        row["n_TCFA_lesions"] = int(len(sig_lesions))
        row["total_TCFA_length_mm"] = float(sig_lesions["length_mm"].sum()) if len(sig_lesions) else 0.0
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)

    # --- Overall per-model comparison, restricted to pullbacks common to ALL folders (fair comparison) ---
    common_pids = set.intersection(*pids_per_model.values()) if pids_per_model else set()
    print(f"\n[INFO] {len(common_pids)} pullback(s) common to all {len(model_names)} folder(s) "
          f"-- used for the Overall_Per_Model comparison")
    df_common_summary = df_summary[df_summary["pullback"].isin(common_pids)]
    numeric_cols = [c for c in df_common_summary.columns if c not in ("model", "pullback")]
    df_overall = df_common_summary.groupby("model", sort=False)[numeric_cols].mean().reset_index() \
        if not df_common_summary.empty else pd.DataFrame(columns=["model"] + numeric_cols)

    excel_path = output_dir / f"continuous_3d_metrics_{datetime.now().strftime('%d_%m_%Y')}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_overall.to_excel(writer, sheet_name="Overall_Per_Model", index=False)
        df_summary.to_excel(writer, sheet_name="Per_Pullback_Summary", index=False)
        df_lesions.to_excel(writer, sheet_name="TCFA_Lesions", index=False)
        df_frames.to_excel(writer, sheet_name="Frame_Level", index=False)

    print(f"\n[INFO] Wrote {excel_path} "
          f"(sheets: Overall_Per_Model, Per_Pullback_Summary, TCFA_Lesions, Frame_Level)")
    if not df_overall.empty:
        print("\n=== Overall comparison (mean over pullbacks common to all folders) ===")
        print(df_overall[["model", "n_TCFA_lesions", "total_TCFA_length_mm", "Mean_FCT_um", "Mean_Lipid_Arc_deg"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
