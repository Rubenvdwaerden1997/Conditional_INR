#!/usr/bin/env python3
"""
Test-set metrics for a trained Conditional INR model.

Given a model folder (best_model.pt/latest.pt + its yaml config), this script:
  1. Reads the split Excel referenced by the config (paths.<env>.split_excel),
     takes the rows with Set == "Testing", and parses the 1-based annotated
     frame numbers in the "Frames" column.
  2. For each test pullback, loads its preprocessed .npz (Volume_input_image +
     Segmentation_image) from paths.<env>.data_root, remaps the raw
     segmentation labels with the config's class mapping (same remap used at
     training time), and predicts the full pullback via
     Pipeline_ConditionalINR.conditional_inr_inference.predict_conditionalinr
     (reused as-is -- same resize/inference/upsample path as the batch
     pipeline, not duplicated here).
  3. At every annotated frame only (the sole ground truth available), compares
     prediction vs. ground truth per class:
       - frame-level presence counts (TP/FP/FN/TN, i.e. does the class appear
         in that frame at all, in GT and/or prediction)
       - Sensitivity, Specificity, PPV, NPV derived from those counts
       - Dice, averaged over frames where the class is present in GT and/or
         prediction (pure true-negative frames excluded, as Dice is undefined
         there)
       - TP-Dice, the same Dice restricted to frames where the class is
         present in BOTH GT and prediction -- isolates segmentation/boundary
         quality from detection performance.
  4. Also computes continuous plaque-quantification metrics per frame -- lipid
     arc, fibrous cap thickness (FCT), calcium arc, calcium depth, calcium
     thickness -- via nnunetv2/Codes/utils/plaque_quantification.py's
     quantification_lipid()/quantification_calcium() (reused as-is), for both
     GT and prediction, using a 12-class label file matching this model's
     post-mapping taxonomy (Metrics/label_description_conditionalinr.txt).
     Skippable with --skip_continuous_metrics (real per-frame runtime cost
     when lipid/calcium is present).

Usage:
    python Pullback_prediction.py \
        --model_dir  ../saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense \
        --checkpoint best \
        --overlap    0.5 \
        --device     cuda \
        [--postprocess] \
        [--output_dir /path/to/output] \
        [--env local|cluster] \
        [--max_pullbacks 3]
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Pipeline_ConditionalINR moved out of this repo (2026-07-31) into the shared
# CARA_Pipelines_Segmentationmodels location, alongside the other model pipelines.
# Dual local/cluster paths, same pattern as conditional_inr_inference.py's own
# cross-repo import of Training_model -- whichever exists on this machine resolves.
_PIPELINE_DIR_LOCAL   = "W:/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
_PIPELINE_DIR_CLUSTER = "/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
PIPELINE_DIR = _PIPELINE_DIR_LOCAL if os.path.isdir(_PIPELINE_DIR_LOCAL) else _PIPELINE_DIR_CLUSTER
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from conditional_inr_inference import load_conditional_inr_model, predict_conditionalinr
from postprocessing_conditionalinr import load_classes_pixels

# Same dual local/cluster sys.path insertion pattern as
# Teacher_student/Codes/Predict/Predict_Evaluate_dict_onefold.py -- whichever
# path exists on the current machine resolves the import, the other is a no-op.
sys.path.insert(1, r"W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
sys.path.insert(1, "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
from plaque_quantification import quantification_lipid, quantification_calcium

# create_html_report.py lives in this same Metrics/ folder -- reused here (rather
# than duplicated) for the optional --html_report pass, so the two scripts share
# one definition of the color map / JPEG-rendering / HTML-building logic. It is
# intentionally torch-free, so importing it here adds no extra heavy deps beyond
# opencv-python (cv2).
from create_html_report import to_uint8_display, colorize, blend_overlay, save_jpg, get_class_colors, build_html

_DEFAULT_POSTPROCESS_CONFIG = os.path.join(PIPELINE_DIR, "postprocessing_classes_conditionalinr.txt")
_DEFAULT_LABEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_description_conditionalinr.txt")

# Canonical post-mapping taxonomy (0-11) -- matches trainer.py's _ITKSNAP_LABELS
# / the "mapping" comments in the Config/Conditional/*.yaml files.
_CLASS_NAMES_12 = [
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media",
    "Sidebranch", "Thrombus", "Plaque rupture", "Layered plaque", "Neovascularization",
]


def get_class_names(num_classes: int) -> List[str]:
    if num_classes == len(_CLASS_NAMES_12):
        return list(_CLASS_NAMES_12)
    return [f"Class {i}" for i in range(num_classes)]


# -----------------------------
# Excel / label-mapping helpers
# -----------------------------
def load_test_split(yml: dict) -> Dict[str, List[int]]:
    """Returns {pullback_id: [0-based annotated frame indices]} for Set == 'Testing'."""
    env   = yml["env"]
    paths = yml["paths"][env]
    df    = pd.read_excel(paths["split_excel"])
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("pullback", "set", "frames"):
        assert col in df.columns, f"Excel must contain '{col}' column"

    df_test = df[df["set"].str.lower() == "testing"]

    pullback_to_frames: Dict[str, List[int]] = {}
    for _, row in df_test.iterrows():
        pid    = str(row["pullback"])
        frames = [
            int(f.strip()) - 1
            for f in str(row["frames"]).split(",")
            if f.strip().isdigit()
        ]
        pullback_to_frames.setdefault(pid, []).extend(frames)
    return pullback_to_frames


def load_label_mapping(yml: dict) -> Optional[Dict[int, int]]:
    seg = yml["model_seg_decoder"]
    if not bool(seg.get("mapping_activated", True)) or not seg.get("mapping"):
        return None
    return {int(k): int(v) for k, v in seg["mapping"].items()}


def remap_labels(seg: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    lut_size = max(int(seg.max()), max(mapping.keys())) + 1
    lut = np.arange(lut_size, dtype=np.int64)
    for old, new in mapping.items():
        lut[old] = new
    return lut[seg]


def find_pullback_npz(pid: str, data_roots: List[str]) -> Optional[Path]:
    for folder in data_roots:
        candidate = Path(folder) / f"{pid}_circ_gray.npz"
        if candidate.exists():
            return candidate
    return None


# -----------------------------
# Per-frame comparison
# -----------------------------
def compare_frame(gt_frame: np.ndarray, pred_frame: np.ndarray, num_classes: int) -> List[dict]:
    """One dict per class with presence + pixel counts for this single frame."""
    rows = []
    for c in range(num_classes):
        gt_mask   = gt_frame == c
        pred_mask = pred_frame == c
        gt_px     = int(gt_mask.sum())
        pred_px   = int(pred_mask.sum())
        inter_px  = int((gt_mask & pred_mask).sum())
        denom     = gt_px + pred_px
        dice      = (2.0 * inter_px / denom) if denom > 0 else float("nan")
        rows.append({
            "class_idx":         c,
            "gt_present":        gt_px > 0,
            "pred_present":      pred_px > 0,
            "gt_pixels":         gt_px,
            "pred_pixels":       pred_px,
            "intersection_pixels": inter_px,
            "dice":              dice,
        })
    return rows


def aggregate_confusion(df: pd.DataFrame, group_cols: List[str], class_names: List[str]) -> pd.DataFrame:
    """Turn frame-level rows into per-group (e.g. per pullback+class, or per class only) summary metrics."""
    out_rows = []
    for keys, g in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        gt_present   = g["gt_present"].to_numpy()
        pred_present = g["pred_present"].to_numpy()

        tp = int(np.sum(gt_present & pred_present))
        fp = int(np.sum(~gt_present & pred_present))
        fn = int(np.sum(gt_present & ~pred_present))
        tn = int(np.sum(~gt_present & ~pred_present))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv         = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

        # Cohen's Kappa (same formula as Teacher_student/Codes/utils/metrics/Classification_evaluation.py)
        kappa_denom = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
        kappa       = (2 * (tp * tn - fn * fp) / kappa_denom) if kappa_denom > 0 else float("nan")

        dice_all_frames = g.loc[gt_present | pred_present, "dice"]
        dice_mean       = float(dice_all_frames.mean()) if len(dice_all_frames) > 0 else float("nan")

        tp_mask   = gt_present & pred_present
        dice_tp   = g.loc[tp_mask, "dice"]
        tp_dice_mean = float(dice_tp.mean()) if len(dice_tp) > 0 else float("nan")

        row = dict(zip(group_cols, keys))
        class_idx = row["class_idx"]
        row.update({
            "class_name":     class_names[class_idx],
            "n_frames":       len(g),
            "n_gt_present":   tp + fn,
            "n_pred_present": tp + fp,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": sensitivity, "Specificity": specificity,
            "PPV": ppv, "NPV": npv, "Kappa": kappa,
            "Dice": dice_mean, "TP_Dice": tp_dice_mean,
        })
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# -----------------------------
# Continuous plaque-quantification metrics (lipid arc/FCT, calcium arc/depth/thickness)
# -----------------------------
def _parse_quant(value) -> float:
    """quantification_lipid/calcium return either a formatted numeric string (e.g. '198')
    or the literal int -99 (their own sentinel for 'structure not present in this frame').
    Both float()-convert cleanly; -99 is remapped to NaN so it doesn't pollute means/MAE."""
    v = float(value)
    return float("nan") if v <= -99 else v


_CONTINUOUS_METRICS = [
    # (summary metric name, GT column, Pred column)
    ("FCT_um",               "GT_FCT_um",               "Pred_FCT_um"),
    ("Lipid_Arc_deg",        "GT_Lipid_Arc_deg",        "Pred_Lipid_Arc_deg"),
    ("Calcium_Depth_um",     "GT_Calcium_Depth_um",     "Pred_Calcium_Depth_um"),
    ("Calcium_Arc_deg",      "GT_Calcium_Arc_deg",      "Pred_Calcium_Arc_deg"),
    ("Calcium_Thickness_um", "GT_Calcium_Thickness_um", "Pred_Calcium_Thickness_um"),
]


def aggregate_continuous(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Per group (e.g. per pullback, or none = overall), per continuous metric:
    N (frames where BOTH GT and prediction show the structure present -- mirrors
    TP-Dice's "both present" convention), MAE, signed Bias (pred-gt), and the
    unrestricted GT/Pred means (computed independently, not just over the N above)."""
    groups = df.groupby(group_cols, sort=False) if group_cols else [((), df)]
    out_rows = []
    for keys, g in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric, gt_col, pred_col in _CONTINUOUS_METRICS:
            both = g[[gt_col, pred_col]].dropna()
            row = dict(zip(group_cols, keys))
            row.update({
                "metric":    metric,
                "N":         len(both),
                "MAE":       float((both[pred_col] - both[gt_col]).abs().mean()) if len(both) else float("nan"),
                "Bias":      float((both[pred_col] - both[gt_col]).mean()) if len(both) else float("nan"),
                "Mean_GT":   float(g[gt_col].mean()) if g[gt_col].notna().any() else float("nan"),
                "Mean_Pred": float(g[pred_col].mean()) if g[pred_col].notna().any() else float("nan"),
            })
            out_rows.append(row)
    return pd.DataFrame(out_rows)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Compute test-set metrics for a Conditional INR model")
    parser.add_argument("--model_dir",   type=str, required=True, help="Folder with best_model.pt/latest.pt + yaml config")
    parser.add_argument("--checkpoint",  type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--output_dir",  type=str, default=None, help="Default: <model_dir>/test_metrics")
    parser.add_argument("--env",         type=str, default=None, choices=["local", "cluster"],
                         help="Override the yaml's 'env' field (which paths.* block to use)")
    parser.add_argument("--overlap",     type=float, default=0.5, help="Sliding z-patch overlap fraction [0.0-1.0]")
    parser.add_argument("--chunk_size",  type=int, default=65536, help="Voxels per INR forward pass")
    parser.add_argument("--use_amp",     action="store_true", help="Encoder-only bfloat16 autocast")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--postprocess", action="store_true", help="Apply small-region cleanup before scoring")
    parser.add_argument("--postprocess_config", type=str, default=_DEFAULT_POSTPROCESS_CONFIG)
    parser.add_argument("--postprocess_n_procs", type=int, default=4)
    parser.add_argument("--max_pullbacks", type=int, default=None, help="Only process the first N test pullbacks (debugging)")
    parser.add_argument("--skip_continuous_metrics", action="store_true",
                         help="Skip lipid arc/FCT/calcium arc/depth/thickness (plaque_quantification.py is slow "
                              "per frame when the structure is present -- adds real runtime on a full test set)")
    parser.add_argument("--label_file", type=str, default=_DEFAULT_LABEL_FILE,
                         help="ITK-SnAP label description matching this model's 12-class post-mapping taxonomy "
                              "(default: Metrics/label_description_conditionalinr.txt)")
    parser.add_argument("--html_report", action="store_true",
                         help="Also render an HTML QC report (OCT | GT | prediction JPEGs, browsable per "
                              "pullback/frame) alongside the metrics -- see create_html_report.py. Reuses the "
                              "volume/GT/prediction already in memory from this same run, so it adds no extra "
                              "inference or file reads, just JPEG encoding per annotated frame.")
    parser.add_argument("--html_alpha", type=float, default=0.45,
                         help="Overlay transparency for the HTML report's GT/prediction color coding")
    parser.add_argument("--html_jpeg_quality", type=int, default=90)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    yaml_path = next(iter(list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))), None)
    if yaml_path is None:
        raise FileNotFoundError(f"No yaml config found in {model_dir}")
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    if args.env:
        yml["env"] = args.env

    # Default folder name encodes --overlap (0.5 -> "test_metrics_05", 1.0 -> "test_metrics_10")
    # so different overlap runs land in separate folders instead of overwriting each other.
    overlap_suffix = str(args.overlap).replace(".", "")
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / f"test_metrics_{overlap_suffix}"
    pred_dir   = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    label_mapping = load_label_mapping(yml)
    pullback_to_frames = load_test_split(yml)
    if args.max_pullbacks:
        pullback_to_frames = dict(list(pullback_to_frames.items())[:args.max_pullbacks])
    print(f"[INFO] {len(pullback_to_frames)} test pullbacks found in split Excel (env={yml['env']})")

    data_roots = yml["paths"][yml["env"]]["data_root"]

    print(f"[INFO] Loading model from {model_dir} (checkpoint={args.checkpoint})...")
    model, cfg, device = load_conditional_inr_model(str(model_dir), args.checkpoint, args.device)
    num_classes  = cfg.num_classes
    class_names  = get_class_names(num_classes)

    html_manifest: List[dict] = []
    class_colors = None
    html_dir = output_dir / "html_report"
    if args.html_report:
        class_colors = get_class_colors(num_classes)
        (html_dir / "images").mkdir(parents=True, exist_ok=True)
        print(f"[INFO] HTML report enabled -> {html_dir / 'report.html'}")

    postprocess_classes = None
    if args.postprocess:
        postprocess_classes = load_classes_pixels(args.postprocess_config)
        print(f"[INFO] Postprocessing enabled, thresholds from {args.postprocess_config}")

    # plaque_quantification.py's font param only accepts 'mine' (hardcoded Windows path)
    # or 'cluster' (hardcoded /data/diag path) -- derive from the same env this run uses.
    quant_font = "mine" if yml["env"] == "local" else "cluster"
    # Native-resolution pixel spacing (mm/px), recovered the same way
    # predict_conditionalinr() computes it for the saved .nii.gz spacing metadata --
    # both seg[z] and pred[z] are always native-resolution arrays (see module docstring).
    native_xy_spacing = cfg.xy_spacing * (cfg.resize_to / cfg.native_xy_size) if cfg.resize_to else cfg.xy_spacing
    if not args.skip_continuous_metrics:
        print(f"[INFO] Continuous metrics enabled (label_file={args.label_file}, font={quant_font}, "
              f"xy_spacing={native_xy_spacing:.6f} mm/px) -- adds runtime per frame with lipid/calcium present")

    frame_rows: List[dict] = []
    continuous_rows: List[dict] = []
    n_done = 0
    for pid, ann_frames in pullback_to_frames.items():
        t0 = time.time()
        npz_path = find_pullback_npz(pid, data_roots)
        if npz_path is None:
            print(f"[WARN] {pid}: no .npz found in {data_roots} -- skipping")
            continue

        data = np.load(npz_path, allow_pickle=False)
        if "Segmentation_image" not in data.files:
            print(f"[WARN] {pid}: no Segmentation_image in {npz_path.name} -- skipping")
            continue

        vol = data["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]
        volume = vol.astype(np.float32)

        seg = data["Segmentation_image"]
        if seg.ndim == 4:
            seg = seg[0]
        seg = seg.astype(np.int64)
        if label_mapping:
            seg = remap_labels(seg, label_mapping)

        D = volume.shape[0]
        valid_frames = [z for z in ann_frames if 0 <= z < D]
        skipped = sorted(set(ann_frames) - set(valid_frames))
        if skipped:
            print(f"[WARN] {pid}: annotated frame(s) {[z + 1 for z in skipped]} out of range (D={D}) -- skipping those")
        if not valid_frames:
            print(f"[WARN] {pid}: no valid annotated frames -- skipping pullback")
            continue

        raw_pred_path  = pred_dir / f"{pid}_pred_overlap{args.overlap}.nii.gz"
        postproc_path  = pred_dir / f"{pid}_pred_overlap{args.overlap}_postprocessed.nii.gz"

        predict_conditionalinr(
            volume, model, cfg, device, str(raw_pred_path),
            overlap_frac=args.overlap, chunk_size=args.chunk_size, use_amp=args.use_amp,
            postprocess_classes=postprocess_classes,
            postprocess_output_path=str(postproc_path) if postprocess_classes is not None else None,
            postprocess_n_procs=args.postprocess_n_procs,
        )

        score_path = postproc_path if (args.postprocess and postproc_path.exists()) else raw_pred_path
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(score_path)))  # [D,H,W]

        if args.html_report:
            pid_img_dir = html_dir / "images" / pid
            pid_img_dir.mkdir(parents=True, exist_ok=True)

        for z in valid_frames:
            frame_class_rows = compare_frame(seg[z], pred[z], num_classes)
            for row in frame_class_rows:
                row["pullback"]    = pid
                row["frame_1based"] = z + 1
                frame_rows.append(row)

            if args.html_report:
                gt_frame   = seg[z]
                pred_frame = pred[z]

                # Reuse the per-class dice just computed above -- excl. background,
                # same "present in GT and/or prediction" convention as aggregate_confusion().
                dice_vals = [r["dice"] for r in frame_class_rows if r["class_idx"] != 0 and not math.isnan(r["dice"])]
                frame_dice = float(np.mean(dice_vals)) if dice_vals else float("nan")

                raw_rgb      = np.stack([to_uint8_display(volume[z])] * 3, axis=-1)
                overlay_gt   = blend_overlay(raw_rgb, colorize(gt_frame, class_colors), gt_frame > 0, args.html_alpha)
                overlay_pred = blend_overlay(raw_rgb, colorize(pred_frame, class_colors), pred_frame > 0, args.html_alpha)

                frame_1based = z + 1
                raw_name  = f"frame{frame_1based:04d}_raw.jpg"
                gt_name   = f"frame{frame_1based:04d}_gt.jpg"
                pred_name = f"frame{frame_1based:04d}_pred.jpg"
                save_jpg(raw_rgb,      pid_img_dir / raw_name,  args.html_jpeg_quality)
                save_jpg(overlay_gt,   pid_img_dir / gt_name,   args.html_jpeg_quality)
                save_jpg(overlay_pred, pid_img_dir / pred_name, args.html_jpeg_quality)

                html_manifest.append({
                    "pullback": pid,
                    "frame":    frame_1based,
                    "dice":     None if math.isnan(frame_dice) else round(frame_dice, 4),
                    "raw":      f"images/{pid}/{raw_name}",
                    "gt":       f"images/{pid}/{gt_name}",
                    "pred":     f"images/{pid}/{pred_name}",
                })

            if not args.skip_continuous_metrics:
                gt_frame   = seg[z].astype(np.int16)
                pred_frame = pred[z].astype(np.int16)
                _, _, gt_fct,   gt_lipid_arc,   _ = quantification_lipid(
                    gt_frame, label_file=args.label_file, xy_spacing=native_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_gt")
                _, _, pred_fct, pred_lipid_arc, _ = quantification_lipid(
                    pred_frame, label_file=args.label_file, xy_spacing=native_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_pred")
                _, _, gt_ca_depth,   gt_ca_arc,   gt_ca_thick,   _ = quantification_calcium(
                    gt_frame, label_file=args.label_file, xy_spacing=native_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_gt")
                _, _, pred_ca_depth, pred_ca_arc, pred_ca_thick, _ = quantification_calcium(
                    pred_frame, label_file=args.label_file, xy_spacing=native_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_pred")
                continuous_rows.append({
                    "pullback": pid, "frame_1based": z + 1,
                    "GT_FCT_um": _parse_quant(gt_fct), "Pred_FCT_um": _parse_quant(pred_fct),
                    "GT_Lipid_Arc_deg": _parse_quant(gt_lipid_arc), "Pred_Lipid_Arc_deg": _parse_quant(pred_lipid_arc),
                    "GT_Calcium_Depth_um": _parse_quant(gt_ca_depth), "Pred_Calcium_Depth_um": _parse_quant(pred_ca_depth),
                    "GT_Calcium_Arc_deg": _parse_quant(gt_ca_arc), "Pred_Calcium_Arc_deg": _parse_quant(pred_ca_arc),
                    "GT_Calcium_Thickness_um": _parse_quant(gt_ca_thick), "Pred_Calcium_Thickness_um": _parse_quant(pred_ca_thick),
                })

        n_done += 1
        print(f"[{n_done}/{len(pullback_to_frames)}] {pid}: {len(valid_frames)} annotated frames scored "
              f"({time.time() - t0:.1f}s)")

    if not frame_rows:
        print("[ERROR] No frames were scored -- nothing to write.")
        return

    df_frames = pd.DataFrame(frame_rows)
    df_frames["class_name"] = df_frames["class_idx"].map(lambda c: class_names[c])
    df_frames = df_frames[[
        "pullback", "frame_1based", "class_idx", "class_name",
        "gt_present", "pred_present", "gt_pixels", "pred_pixels", "intersection_pixels", "dice",
    ]]

    df_per_pullback = aggregate_confusion(df_frames, ["pullback", "class_idx"], class_names)
    df_per_pullback = df_per_pullback[[
        "pullback", "class_idx", "class_name", "n_frames", "n_gt_present", "n_pred_present",
        "TP", "FP", "FN", "TN", "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "Dice", "TP_Dice",
    ]]

    df_overall = aggregate_confusion(df_frames, ["class_idx"], class_names)
    df_overall = df_overall[[
        "class_idx", "class_name", "n_frames", "n_gt_present", "n_pred_present",
        "TP", "FP", "FN", "TN", "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "Dice", "TP_Dice",
    ]]

    sheet_names = ["Overall", "Per_Pullback", "Frame_Level"]
    df_continuous_frames = df_continuous_per_pullback = df_continuous_overall = None
    if continuous_rows:
        df_continuous_frames = pd.DataFrame(continuous_rows)
        df_continuous_per_pullback = aggregate_continuous(df_continuous_frames, ["pullback"])
        df_continuous_overall      = aggregate_continuous(df_continuous_frames, [])
        sheet_names += ["Continuous_Overall", "Continuous_Per_Pullback", "Continuous_Frame_Level"]

    excel_path = output_dir / "test_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_overall.to_excel(writer, sheet_name="Overall", index=False)
        df_per_pullback.to_excel(writer, sheet_name="Per_Pullback", index=False)
        df_frames.to_excel(writer, sheet_name="Frame_Level", index=False)
        if continuous_rows:
            df_continuous_overall.to_excel(writer, sheet_name="Continuous_Overall", index=False)
            df_continuous_per_pullback.to_excel(writer, sheet_name="Continuous_Per_Pullback", index=False)
            df_continuous_frames.to_excel(writer, sheet_name="Continuous_Frame_Level", index=False)

    print(f"\n[INFO] Wrote {excel_path} (sheets: {', '.join(sheet_names)})")

    print("\n=== Overall test-set metrics (pooled across all annotated frames), classes 1..N (background excluded) ===")
    non_bg = df_overall[df_overall["class_idx"] != 0]
    print(non_bg[["class_name", "n_gt_present", "n_pred_present", "Sensitivity", "Specificity", "PPV", "NPV", "Kappa", "Dice", "TP_Dice"]]
          .to_string(index=False, float_format=lambda x: "nan" if math.isnan(x) else f"{x:.3f}"))
    print(f"\nMean Dice (excl. background): {non_bg['Dice'].mean():.4f}")
    print(f"Mean TP-Dice (excl. background): {non_bg['TP_Dice'].mean():.4f}")

    if df_continuous_overall is not None:
        print("\n=== Overall continuous plaque-quantification metrics (N = frames with structure in BOTH GT and prediction) ===")
        print(df_continuous_overall[["metric", "N", "MAE", "Bias", "Mean_GT", "Mean_Pred"]]
              .to_string(index=False, float_format=lambda x: "nan" if math.isnan(x) else f"{x:.2f}"))

    if args.html_report:
        if html_manifest:
            report_path = html_dir / "report.html"
            build_html(html_manifest, class_names, class_colors, report_path)
            print(f"\n[INFO] Wrote HTML QC report: {report_path} ({len(html_manifest)} frames)")
        else:
            print("\n[WARN] --html_report was set but no frames were rendered -- nothing to write.")


if __name__ == "__main__":
    main()
