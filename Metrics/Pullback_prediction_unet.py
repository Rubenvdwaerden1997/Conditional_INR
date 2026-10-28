#!/usr/bin/env python3
"""
Test-set metrics for a trained ResidualUNet2p5D model.

Mirrors Pullback_prediction.py's pipeline exactly (test-split loading, label
remapping, per-frame/per-class comparison, Excel/HTML report, continuous
plaque-quantification metrics) -- all of that is model-agnostic, so it's
imported and reused as-is, not duplicated. Only two things differ, swapped via
residualunet2p5d_inference.py: model loading and the inference call itself
(ResidualUNet2p5D has no coordinate query, so there's no --eval_resolution,
--chunk_size or --use_amp here -- see residualunet2p5d_inference.py's
docstring for why: the model always predicts at exactly its own encoder input
resolution, there's no native-resolution query to opt into).

Usage:
    python Pullback_prediction_unet.py \
        --model_dir  ../saved_models_unet2p5D_conditional/residualunet2p5d_encoder64_depth5_zarrstore_zjitter_perframe \
        --checkpoint best \
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
from typing import List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "Training_model"))

from residualunet2p5d_inference import load_residualunet2p5d_model, predict_residualunet2p5d
from postprocessing_conditionalinr import load_classes_pixels
from model import resize_labels, resize_volume

# Reuse Pullback_prediction.py's model-agnostic pieces directly -- same folder,
# plain import, nothing duplicated.
from Pullback_prediction import (
    get_class_names, load_test_split, load_label_mapping, remap_labels,
    find_pullback_npz, compare_frame, aggregate_confusion, aggregate_continuous,
    _parse_quant, _CONTINUOUS_METRICS, _DEFAULT_LABEL_FILE,
)

sys.path.insert(1, r"W:/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
sys.path.insert(1, "/data/diag/rubenvdw/nnunetv2/nnUNet/nnunetv2/Codes/utils")
from plaque_quantification import quantification_lipid, quantification_calcium

from create_html_report import to_uint8_display, colorize, blend_overlay, save_jpg, get_class_colors, build_html
from HFER import add_hfer_args, compute_pullback_hfer, build_hfer_tables, load_exclusions

_PIPELINE_DIR_LOCAL   = "W:/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
_PIPELINE_DIR_CLUSTER = "/data/diag/rubenvdw/CARA_Pipelines_Segmentationmodels/Pipeline_ConditionalINR"
PIPELINE_DIR = _PIPELINE_DIR_LOCAL if os.path.isdir(_PIPELINE_DIR_LOCAL) else _PIPELINE_DIR_CLUSTER
_DEFAULT_POSTPROCESS_CONFIG = os.path.join(PIPELINE_DIR, "postprocessing_classes_conditionalinr.txt")


def main():
    parser = argparse.ArgumentParser("Compute test-set metrics for a ResidualUNet2p5D model")
    parser.add_argument("--model_dir",   type=str, required=True, help="Folder with best_model.pt/latest.pt + yaml config")
    parser.add_argument("--checkpoint",  type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--output_dir",  type=str, default=None, help="Default: <model_dir>/test_metrics")
    parser.add_argument("--env",         type=str, default=None, choices=["local", "cluster"],
                         help="Override the yaml's 'env' field (which paths.* block to use)")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--postprocess", action="store_true", help="Apply small-region cleanup before scoring")
    parser.add_argument("--postprocess_config", type=str, default=_DEFAULT_POSTPROCESS_CONFIG)
    parser.add_argument("--postprocess_n_procs", type=int, default=4)
    parser.add_argument("--max_pullbacks", type=int, default=None, help="Only process the first N test pullbacks (debugging)")
    parser.add_argument("--skip_existing_predictions", action="store_true",
                         help="If this pullback's prediction .nii.gz already exists (the postprocessed one if "
                              "--postprocess is set, else the raw one), reuse it instead of re-running inference. "
                              "Off by default: always recompute, same as before.")
    parser.add_argument("--skip_continuous_metrics", action="store_true",
                         help="Skip lipid arc/FCT/calcium arc/depth/thickness (plaque_quantification.py is slow "
                              "per frame when the structure is present -- adds real runtime on a full test set)")
    add_hfer_args(parser)
    parser.add_argument("--label_file", type=str, default=_DEFAULT_LABEL_FILE)
    parser.add_argument("--html_report", action="store_true")
    parser.add_argument("--html_alpha", type=float, default=0.45)
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

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "test_metrics"
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
    model, cfg, device = load_residualunet2p5d_model(str(model_dir), args.checkpoint, args.device)
    num_classes  = cfg.num_classes
    class_names  = get_class_names(num_classes)

    html_manifest: List[dict] = []
    class_colors = None
    html_dir = output_dir / "html_report"
    report_name = "report.html"
    if args.html_report:
        class_colors = get_class_colors(num_classes)
        (html_dir / "images").mkdir(parents=True, exist_ok=True)
        print(f"[INFO] HTML report enabled -> {html_dir / report_name}")

    postprocess_classes = None
    if args.postprocess:
        postprocess_classes = load_classes_pixels(args.postprocess_config)
        print(f"[INFO] Postprocessing enabled, thresholds from {args.postprocess_config}")

    quant_font = "mine" if yml["env"] == "local" else "cluster"
    # ResidualUNet2p5D always predicts at exactly its own encoder input resolution
    # (cfg.resize_to, or native if resize_to=0) -- cfg.xy_spacing is already correct
    # for that grid (Config.__post_init__ rescales it from native_xy_size whenever
    # resize_to is set). Ground truth is downsampled to match below, same as
    # ConditionalINR's eval_resolution="encoder" mode -- there is no "native" mode
    # for this model, it has no coordinate query to decouple output resolution from
    # encoder input resolution with.
    eval_xy_spacing = cfg.xy_spacing
    print(f"[INFO] ResidualUNet2p5D predicts at its own encoder resolution "
          f"(resize_to={cfg.resize_to or 'native'}) -- ground truth downsampled to match.")
    if not args.skip_continuous_metrics:
        print(f"[INFO] Continuous metrics enabled (label_file={args.label_file}, font={quant_font}, "
              f"xy_spacing={eval_xy_spacing:.6f} mm/px) -- adds runtime per frame with lipid/calcium present")

    frame_rows: List[dict] = []
    continuous_rows: List[dict] = []
    hfer_rows: List[dict] = []
    pixel_frames: List[pd.DataFrame] = []
    hfer_exclusions = {} if args.skip_hfer else load_exclusions(args.exclusion_excel)
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

        # Always downsample ground truth to cfg.resize_to (nearest-neighbour) --
        # ResidualUNet2p5D has no native-resolution query, see note above. No-op
        # when cfg.resize_to == 0.
        if cfg.resize_to:
            seg_eval    = resize_labels(torch.from_numpy(seg), cfg.resize_to).numpy()
            volume_eval = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
        else:
            seg_eval    = seg
            volume_eval = volume

        D = volume.shape[0]
        valid_frames = [z for z in ann_frames if 0 <= z < D]
        skipped = sorted(set(ann_frames) - set(valid_frames))
        if skipped:
            print(f"[WARN] {pid}: annotated frame(s) {[z + 1 for z in skipped]} out of range (D={D}) -- skipping those")
        if not valid_frames:
            print(f"[WARN] {pid}: no valid annotated frames -- skipping pullback")
            continue

        raw_pred_path  = pred_dir / f"{pid}_pred.nii.gz"
        postproc_path  = pred_dir / f"{pid}_pred_postprocessed.nii.gz"

        needed_path = postproc_path if postprocess_classes is not None else raw_pred_path
        if args.skip_existing_predictions and needed_path.exists():
            print(f"[INFO] {pid}: {needed_path.name} already exists -- skipping inference, reusing it")
        else:
            predict_residualunet2p5d(
                volume, model, cfg, device, str(raw_pred_path),
                postprocess_classes=postprocess_classes,
                postprocess_output_path=str(postproc_path) if postprocess_classes is not None else None,
                postprocess_n_procs=args.postprocess_n_procs,
            )

        score_path = postproc_path if (args.postprocess and postproc_path.exists()) else raw_pred_path
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(score_path)))  # [D,H,W]
        assert pred.shape[1:] == seg_eval.shape[1:], (
            f"{pid}: pred grid {pred.shape[1:]} != ground-truth grid {seg_eval.shape[1:]} -- "
            "predict_residualunet2p5d's resize and this script's ground-truth downsampling "
            "have gone out of sync, would silently corrupt every Dice number."
        )

        if not args.skip_hfer:
            # Whole predicted volume (not just annotated frames); `pred` is the postprocessed file when
            # --postprocess is set. Pixel counts are saved too so HFER can be recomputed at another cutoff.
            hfer_row, df_px = compute_pullback_hfer(
                pred, pid, class_names, args.hfer_threshold_cycles_per_mm, args.hfer_min_ac_power,
                getattr(cfg, "z_spacing", 0.1), hfer_exclusions, args.hfer_min_stretch_frames)
            hfer_rows.append(hfer_row)
            pixel_frames.append(df_px)

        if args.html_report:
            pid_img_dir = html_dir / "images" / pid
            pid_img_dir.mkdir(parents=True, exist_ok=True)

        for z in valid_frames:
            frame_class_rows = compare_frame(seg_eval[z], pred[z], num_classes)
            for row in frame_class_rows:
                row["pullback"]    = pid
                row["frame_1based"] = z + 1
                frame_rows.append(row)

            if args.html_report:
                gt_frame   = seg_eval[z]
                pred_frame = pred[z]

                dice_vals = [r["dice"] for r in frame_class_rows if r["class_idx"] != 0 and not math.isnan(r["dice"])]
                frame_dice = float(np.mean(dice_vals)) if dice_vals else float("nan")

                raw_rgb      = np.stack([to_uint8_display(volume_eval[z])] * 3, axis=-1)
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
                gt_frame   = seg_eval[z].astype(np.int16)
                pred_frame = pred[z].astype(np.int16)
                _, _, gt_fct,   gt_lipid_arc,   _ = quantification_lipid(
                    gt_frame, label_file=args.label_file, xy_spacing=eval_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_gt")
                _, _, pred_fct, pred_lipid_arc, _ = quantification_lipid(
                    pred_frame, label_file=args.label_file, xy_spacing=eval_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_pred")
                _, _, gt_ca_depth,   gt_ca_arc,   gt_ca_thick,   _ = quantification_calcium(
                    gt_frame, label_file=args.label_file, xy_spacing=eval_xy_spacing,
                    font=quant_font, filename=f"{pid}_frame{z + 1:04d}_gt")
                _, _, pred_ca_depth, pred_ca_arc, pred_ca_thick, _ = quantification_calcium(
                    pred_frame, label_file=args.label_file, xy_spacing=eval_xy_spacing,
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

    df_hfer, df_hfer_overall, df_pixels = build_hfer_tables(hfer_rows, pixel_frames, class_names)
    if df_hfer is not None:
        sheet_names += ["HFER_Per_Pullback", "HFER_Overall", "Pixels_Per_Frame"]

    excel_path = output_dir / "test_metrics.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_overall.to_excel(writer, sheet_name="Overall", index=False)
        df_per_pullback.to_excel(writer, sheet_name="Per_Pullback", index=False)
        df_frames.to_excel(writer, sheet_name="Frame_Level", index=False)
        if continuous_rows:
            df_continuous_overall.to_excel(writer, sheet_name="Continuous_Overall", index=False)
            df_continuous_per_pullback.to_excel(writer, sheet_name="Continuous_Per_Pullback", index=False)
            df_continuous_frames.to_excel(writer, sheet_name="Continuous_Frame_Level", index=False)
        if df_hfer is not None:
            df_hfer.to_excel(writer, sheet_name="HFER_Per_Pullback", index=False)
            df_hfer_overall.to_excel(writer, sheet_name="HFER_Overall", index=False)
            df_pixels.to_excel(writer, sheet_name="Pixels_Per_Frame", index=False)

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

    if df_hfer_overall is not None:
        print("\n=== HFER (high-frequency energy ratio of per-class area along z; lower = smoother), mean over pullbacks ===")
        print(df_hfer_overall.to_string(index=False, float_format=lambda x: "nan" if math.isnan(x) else f"{x:.2f}"))

    if args.html_report:
        if html_manifest:
            report_path = html_dir / report_name
            # long_manifest=[] -- this script only renders per-frame overlays, never the
            # separate longitudinal-angle-sweep view create_html_report.py's own main() builds;
            # the HTML's own JS handles an empty LONG_MANIFEST gracefully (just disables that
            # view's toggle), so this isn't a missing feature, just one this script never had.
            build_html(html_manifest, [], class_names, class_colors, report_path)
            print(f"\n[INFO] Wrote HTML QC report: {report_path} ({len(html_manifest)} frames)")
        else:
            print("\n[WARN] --html_report was set but no frames were rendered -- nothing to write.")


if __name__ == "__main__":
    main()
