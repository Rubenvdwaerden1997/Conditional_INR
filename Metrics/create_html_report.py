#!/usr/bin/env python3
"""
Lightweight HTML QC report for a trained Conditional INR model: OCT frame |
ground truth | prediction, side by side, for every annotated test-set frame.

Replaces the old matplotlib-PDF approach (nnunetv2/Codes/Predictions_PDF/
create_imgs_pdf.py) which was slow (huge 50x50in figures) and produced a
single very large, hard-to-browse PDF. Instead this script:
  - writes plain JPEGs via OpenCV (fast, small files) instead of matplotlib
    figures,
  - builds one small report.html that references those JPEGs by relative
    path (images are not embedded/base64'd, so the HTML itself stays tiny
    and opens instantly even with thousands of frames),
  - lets you browse pullback -> frame in the browser (click, or arrow keys),
    with a quick per-frame Dice badge to help you jump straight to the worst
    predictions.
  - also builds a rotating longitudinal ("L-mode") view per pullback: for a
    set of rotation angles around the pullback axis, it takes the diameter
    line through each frame's center and stacks those across every z-frame
    into one 2D image. This uses the full 3D prediction (not just annotated
    frames), so it's the way to check whether predictions stay consistent
    along the length of a pullback rather than flickering frame-to-frame.
    Toggle it with the "Longitudinal view" button in the report (prediction
    only -- no GT longitudinal, since GT is only annotated at sparse frames).
    Disable with --no_longitudinal, or change the rotation granularity with
    --long_angle_step (default 15 deg -> 12 angles spanning 0-165 deg).

It does NOT run inference itself -- it reads the .nii.gz predictions already
written by Pullback_prediction.py (in <model_dir>/test_metrics/predictions),
so re-generating the visual report after tweaking colors/alpha is cheap and
doesn't require the model/GPU at all.

Usage:
    python create_html_report.py \
        --model_dir ../saved_models_3D_conditional/conditional_3D_relu_cedice_trilinear_encoder64_depth5_nodense \
        --overlap 0.5 \
        [--use_postprocessed] \
        [--output_dir /path/to/output] \
        [--env local|cluster] \
        [--max_pullbacks 3] \
        [--eval_resolution native|encoder]

--eval_resolution must match whatever Pullback_prediction.py was run with for these
predictions: "native" (default) reads native-resolution predictions and ground truth;
"encoder" reads predictions saved at the model's own encoder input resolution
(cfg.resize_to) and downsamples the ground truth to match before rendering. Output
lands in a resolution-specific folder/filename (see below) so the two never collide.
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml
import cv2

# Only needed for --eval_resolution=encoder (downsampling ground truth/raw image to
# cfg.resize_to before rendering) and to read cfg.resize_to itself -- reuses the exact
# same resize_labels/resize_volume Pullback_prediction.py and training use, rather than
# reimplementing a second, possibly-divergent resize here. CPU-only (no GPU needed for
# these two functions), so this doesn't change the "no --gpus-per-task" note in
# create_html_report.sh -- it does mean torch must be importable in whatever
# environment runs this script (already true in the train_monai container the .sh
# script uses).
_CONDITIONAL_INR_REPO_LOCAL   = Path("W:/rubenvdw/Conditional_INR")
_CONDITIONAL_INR_REPO_CLUSTER = Path("/data/diag/rubenvdw/Conditional_INR")
for _repo_root in (_CONDITIONAL_INR_REPO_LOCAL, _CONDITIONAL_INR_REPO_CLUSTER):
    sys.path.insert(0, str(_repo_root / "Training_model"))
    sys.path.insert(0, str(_repo_root))
from model import resize_labels, resize_volume                     # Training_model/model.py
from predict_singlepullback_conditional import build_config_from_yaml

# Mirrors Pullback_prediction.py's class-name convention (post label-mapping,
# 12-class taxonomy) -- duplicated rather than imported so this script stays
# a light, torch-free tool (Pullback_prediction.py's import of
# conditional_inr_inference pulls in the model/torch stack, which this script
# has no need for).
_CLASS_NAMES_12 = [
    "Background", "Lumen", "Guidewire", "Intima", "Lipid", "Calcium", "Media",
    "Sidebranch", "Thrombus", "Plaque rupture", "Layered plaque", "Neovascularization",
]

# Colors from Metrics/label_description_conditionalinr.txt (the ITK-SnAP
# description matching this model's 12-class post-mapping taxonomy) -- kept
# in sync with that file so the report's legend matches what you'd see in
# ITK-SnAP.
_CLASS_COLORS_12 = [
    (0, 0, 0),        # Background
    (255, 0, 0),      # Lumen
    (63, 63, 63),     # Guidewire
    (0, 0, 255),      # Intima
    (255, 255, 0),    # Lipid
    (210, 210, 210),  # Calcium
    (255, 0, 255),    # Media
    (255, 123, 0),    # Sidebranch
    (230, 141, 230),  # Thrombus
    (208, 190, 161),  # Plaque rupture
    (0, 255, 0),       # Layered/Healed plaque
    (162, 162, 162),  # Neovascularization
]


def get_class_names(num_classes: int) -> List[str]:
    if num_classes == len(_CLASS_NAMES_12):
        return list(_CLASS_NAMES_12)
    return [f"Class {i}" for i in range(num_classes)]


def get_class_colors(num_classes: int) -> List[Tuple[int, int, int]]:
    if num_classes == len(_CLASS_COLORS_12):
        return list(_CLASS_COLORS_12)
    colors = [(0, 0, 0)]
    for i in range(1, num_classes):
        h = (i - 1) / max(num_classes - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


# -----------------------------
# Excel / label-mapping helpers (mirrors Pullback_prediction.py)
# -----------------------------
def load_test_split(yml: dict) -> Dict[str, List[int]]:
    """Returns {pullback_id: [0-based annotated frame indices]} for Set == 'Testing'."""
    env = yml["env"]
    paths = yml["paths"][env]
    df = pd.read_excel(paths["split_excel"])
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("pullback", "set", "frames"):
        assert col in df.columns, f"Excel must contain '{col}' column"

    df_test = df[df["set"].str.lower() == "testing"]

    pullback_to_frames: Dict[str, List[int]] = {}
    for _, row in df_test.iterrows():
        pid = str(row["pullback"])
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


def find_pred_file(pid: str, pred_dir: Path, overlap: float, use_postprocessed: bool) -> Optional[Path]:
    raw_path = pred_dir / f"{pid}_pred_overlap{overlap}.nii.gz"
    postproc_path = pred_dir / f"{pid}_pred_overlap{overlap}_postprocessed.nii.gz"
    if use_postprocessed and postproc_path.exists():
        return postproc_path
    if raw_path.exists():
        return raw_path
    # Fall back to whatever overlap value was actually used, so the report
    # still works if --overlap doesn't match the Pullback_prediction.py run.
    candidates = sorted(pred_dir.glob(f"{pid}_pred_overlap*.nii.gz"))
    if candidates:
        preferred = [c for c in candidates if use_postprocessed == ("_postprocessed" in c.stem)]
        chosen = preferred[0] if preferred else candidates[0]
        print(f"[WARN] {pid}: exact overlap={overlap} prediction not found, using {chosen.name} instead")
        return chosen
    return None


# -----------------------------
# Image helpers
# -----------------------------
def to_uint8_display(frame: np.ndarray) -> np.ndarray:
    """Percentile contrast-stretch to 0-255 gray, robust to whatever
    normalization the underlying dataset used (plain 0-255 grayscale,
    foreground z-scored, etc.) -- see label mapping note in module docstring."""
    lo, hi = np.percentile(frame, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((frame.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def colorize(label_img: np.ndarray, colors: List[Tuple[int, int, int]]) -> np.ndarray:
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        if idx == 0:
            continue
        mask = label_img == idx
        if mask.any():
            out[mask] = color
    return out


def blend_overlay(raw_gray_rgb: np.ndarray, colored: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = raw_gray_rgb.copy()
    if mask.any():
        out[mask] = (
            raw_gray_rgb[mask].astype(np.float32) * (1 - alpha)
            + colored[mask].astype(np.float32) * alpha
        ).astype(np.uint8)
    return out


def save_jpg(rgb_img: np.ndarray, path: Path, quality: int) -> None:
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def frame_mean_dice(gt: np.ndarray, pred: np.ndarray, num_classes: int) -> float:
    """Mean Dice over classes present in GT and/or prediction, excluding background --
    same convention as Pullback_prediction.py's aggregate_confusion() Dice column, just
    for a single frame."""
    dices = []
    for c in range(1, num_classes):
        gt_m = gt == c
        pred_m = pred == c
        if not gt_m.any() and not pred_m.any():
            continue
        denom = gt_m.sum() + pred_m.sum()
        inter = np.logical_and(gt_m, pred_m).sum()
        dices.append(2.0 * inter / denom if denom > 0 else np.nan)
    return float(np.nanmean(dices)) if dices else float("nan")


# -----------------------------
# Longitudinal (L-mode) view
# -----------------------------
# A pullback frame is a circular cross-section with the catheter at its center. Rotating
# the frame by `angle_deg` around that center and reading off the row through the middle
# is equivalent to sampling the *original* frame along a diameter line at that angle --
# the same construction IVUS/OCT consoles use for their longitudinal ("L-mode") pullback
# view. Stacking that line across every z-frame gives one 2D image (z vertical, diameter
# horizontal) that makes frame-to-frame (in)consistency of the 3D prediction visible at a
# glance, which a per-frame-only report can't show.
def extract_diameter_line(frame2d: np.ndarray, angle_deg: float, is_label: bool) -> np.ndarray:
    h, w = frame2d.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    rotated = cv2.warpAffine(frame2d.astype(np.float32), M, (w, h), flags=interp, borderValue=0.0)
    line = rotated[h // 2, :]
    return np.rint(line).astype(np.int64) if is_label else line


def build_longitudinal_images(
    volume: np.ndarray,
    pred: np.ndarray,
    angles: List[int],
    colors: List[Tuple[int, int, int]],
    alpha: float,
    pid: str,
    out_dir: Path,
    jpeg_quality: int,
) -> List[dict]:
    """volume/pred: full [D,H,W] pullback (all z, not just annotated frames) -- the
    prediction is a continuous 3D field regardless of where GT annotations happen to be."""
    D, H, W = volume.shape
    entries = []
    for angle in angles:
        raw_lines = np.zeros((D, W), dtype=np.float32)
        pred_lines = np.zeros((D, W), dtype=np.int64)
        for z in range(D):
            raw_lines[z] = extract_diameter_line(volume[z], angle, is_label=False)
            pred_lines[z] = extract_diameter_line(pred[z], angle, is_label=True)

        # Transpose so the pullback (z) axis runs horizontally and the diameter axis
        # runs vertically -- stretches the L-mode image out wide instead of tall, which
        # is how IVUS/OCT consoles display it and is much easier to scan frame-to-frame.
        raw_lines = raw_lines.T
        pred_lines = pred_lines.T

        raw_rgb = np.stack([to_uint8_display(raw_lines)] * 3, axis=-1)
        colored_pred = colorize(pred_lines, colors)
        overlay_pred = blend_overlay(raw_rgb, colored_pred, pred_lines > 0, alpha)

        raw_name = f"long_angle{angle:03d}_raw.jpg"
        pred_name = f"long_angle{angle:03d}_pred.jpg"
        save_jpg(raw_rgb, out_dir / raw_name, jpeg_quality)
        save_jpg(overlay_pred, out_dir / pred_name, jpeg_quality)

        entries.append({
            "pullback": pid,
            "angle": angle,
            "raw": f"images/{pid}/{raw_name}",
            "pred": f"images/{pid}/{pred_name}",
        })
    return entries


# -----------------------------
# HTML report
# -----------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Conditional INR -- Prediction QC report</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background: #14161a; color: #e6e6e6; display: flex; height: 100vh; overflow: hidden;
  }
  #sidebar {
    width: 280px; min-width: 280px; background: #1b1e24; border-right: 1px solid #2c2f36;
    overflow-y: auto; padding: 10px;
  }
  #sidebar h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: #9aa0aa; margin: 6px 4px 8px; }
  .pullback-item { padding: 7px 8px; border-radius: 6px; cursor: pointer; font-size: 13px; margin-bottom: 2px; }
  .pullback-item:hover { background: #262a32; }
  .pullback-item.active { background: #2f5d8f; color: #fff; }
  .pullback-item .count { float: right; color: #8892a0; font-size: 11px; }
  #main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  #topbar {
    padding: 10px 16px; border-bottom: 1px solid #2c2f36; display: flex; align-items: center;
    gap: 14px; background: #1b1e24; flex-wrap: wrap;
  }
  #topbar .title { font-size: 15px; font-weight: 600; }
  #topbar .stat { font-size: 12px; color: #9aa0aa; }
  #modeToggle { display: flex; gap: 4px; background: #262a32; border-radius: 6px; padding: 3px; }
  #modeToggle button {
    border: none; background: transparent; color: #9aa0aa; font-size: 12px; padding: 5px 10px;
    border-radius: 4px; cursor: pointer;
  }
  #modeToggle button.active { background: #2f5d8f; color: #fff; }
  #frameStrip { display: flex; gap: 4px; padding: 8px 16px; overflow-x: auto; background: #17191f; border-bottom: 1px solid #2c2f36; }
  .frame-btn {
    min-width: 42px; padding: 5px 6px; text-align: center; border-radius: 5px; font-size: 12px;
    cursor: pointer; background: #262a32; color: #cdd2da; border: 1px solid transparent;
  }
  .frame-btn:hover { border-color: #4c7fb8; }
  .frame-btn.active { background: #2f5d8f; color: #fff; }
  .frame-btn .d { display: block; font-size: 10px; margin-top: 1px; }
  .dice-good { color: #6fe08a; } .dice-mid { color: #f0c95a; } .dice-bad { color: #f0705a; }
  #viewer { flex: 1; display: flex; align-items: center; justify-content: center; gap: 10px; padding: 14px; overflow: auto; min-height: 0; }
  #viewer.long-mode { flex-direction: column; gap: 16px; align-items: stretch; justify-content: flex-start; overflow: hidden; }
  .panel { flex: 1; max-width: 33%; text-align: center; }
  .panel.wide { max-width: 100%; width: 100%; flex: 1 1 0; min-height: 0; display: flex; flex-direction: column; }
  .panel.hidden { display: none; }
  .panel img { max-width: 100%; max-height: 70vh; border-radius: 6px; background: #000; border: 1px solid #2c2f36; }
  .panel.wide img { flex: 1; min-height: 0; width: 100%; height: 100%; max-height: none; object-fit: fill; }
  .panel .label { font-size: 12px; color: #9aa0aa; margin-top: 6px; flex: none; }
  #legend { padding: 8px 16px; border-top: 1px solid #2c2f36; background: #1b1e24; display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: #cdd2da; }
  .swatch { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  #navHint { font-size: 11px; color: #6b7280; margin-left: auto; }
</style>
</head>
<body>
  <div id="sidebar">
    <h2>Pullbacks</h2>
    <div id="pullbackList"></div>
  </div>
  <div id="main">
    <div id="topbar">
      <div class="title" id="titleText">--</div>
      <div class="stat" id="statText"></div>
      <div id="modeToggle">
        <button id="modeFrameBtn">Frame view</button>
        <button id="modeLongBtn">Longitudinal view</button>
      </div>
      <div id="navHint">&larr; / &rarr; = prev/next</div>
    </div>
    <div id="frameStrip"></div>
    <div id="viewer">
      <div class="panel" id="panelRaw"><img id="imgRaw"><div class="label" id="labelRaw">OCT frame</div></div>
      <div class="panel" id="panelGt"><img id="imgGt"><div class="label">Ground truth</div></div>
      <div class="panel" id="panelPred"><img id="imgPred"><div class="label" id="labelPred">Prediction</div></div>
    </div>
    <div id="legend"></div>
  </div>

<script>
const MANIFEST = __MANIFEST_JSON__;
const LONG_MANIFEST = __LONG_MANIFEST_JSON__;
const CLASS_NAMES = __CLASS_NAMES_JSON__;
const CLASS_COLORS = __CLASS_COLORS_JSON__;

const byPullback = {};
for (const row of MANIFEST) {
  (byPullback[row.pullback] = byPullback[row.pullback] || []).push(row);
}
const byPullbackLong = {};
for (const row of LONG_MANIFEST) {
  (byPullbackLong[row.pullback] = byPullbackLong[row.pullback] || []).push(row);
}
const pullbackIds = Object.keys(byPullback);
const hasLongitudinal = LONG_MANIFEST.length > 0;

let mode = 'frame';   // 'frame' | 'long'
let curPullback = pullbackIds[0];
let curIdx = 0;       // index into byPullback[curPullback] (frame mode) or byPullbackLong[curPullback] (long mode)

function diceClass(d) {
  if (d === null || isNaN(d)) return '';
  if (d >= 0.7) return 'dice-good';
  if (d >= 0.4) return 'dice-mid';
  return 'dice-bad';
}
function fmtDice(d) { return (d === null || isNaN(d)) ? 'n/a' : d.toFixed(2); }

function curRows() {
  return mode === 'long' ? (byPullbackLong[curPullback] || []) : (byPullback[curPullback] || []);
}

function renderSidebar() {
  const el = document.getElementById('pullbackList');
  el.innerHTML = '';
  for (const pid of pullbackIds) {
    const div = document.createElement('div');
    div.className = 'pullback-item' + (pid === curPullback ? ' active' : '');
    const count = mode === 'long' ? (byPullbackLong[pid] || []).length : byPullback[pid].length;
    div.innerHTML = pid + '<span class="count">' + count + '</span>';
    div.onclick = () => { curPullback = pid; curIdx = 0; render(); };
    el.appendChild(div);
  }
}

function renderFrameStrip() {
  const el = document.getElementById('frameStrip');
  el.innerHTML = '';
  curRows().forEach((row, i) => {
    const div = document.createElement('div');
    div.className = 'frame-btn' + (i === curIdx ? ' active' : '');
    if (mode === 'long') {
      div.innerHTML = row.angle + '&deg;';
    } else {
      div.innerHTML = row.frame + '<span class="d ' + diceClass(row.dice) + '">' + fmtDice(row.dice) + '</span>';
    }
    div.onclick = () => { curIdx = i; render(); };
    el.appendChild(div);
  });
}

function renderLegend() {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  CLASS_NAMES.forEach((name, i) => {
    const c = CLASS_COLORS[i];
    const span = document.createElement('span');
    span.innerHTML = '<span class="swatch" style="background: rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')"></span>' + name;
    el.appendChild(span);
  });
}

function setMode(newMode) {
  if (newMode === 'long' && !hasLongitudinal) return;
  mode = newMode;
  curIdx = 0;
  document.getElementById('modeFrameBtn').className = mode === 'frame' ? 'active' : '';
  document.getElementById('modeLongBtn').className = mode === 'long' ? 'active' : '';
  document.getElementById('panelGt').className = 'panel' + (mode === 'long' ? ' hidden' : '');
  document.getElementById('panelRaw').className = 'panel' + (mode === 'long' ? ' wide' : '');
  document.getElementById('panelPred').className = 'panel' + (mode === 'long' ? ' wide' : '');
  document.getElementById('viewer').className = mode === 'long' ? 'long-mode' : '';
  document.getElementById('labelRaw').textContent = mode === 'long' ? 'OCT (longitudinal)' : 'OCT frame';
  document.getElementById('labelPred').textContent = mode === 'long' ? 'Prediction (longitudinal)' : 'Prediction';
  render();
}

function render() {
  const rows = curRows();
  if (rows.length === 0) { renderSidebar(); renderFrameStrip(); return; }
  const row = rows[curIdx];
  document.getElementById('imgRaw').src = row.raw;
  document.getElementById('imgPred').src = row.pred;
  if (mode === 'long') {
    document.getElementById('titleText').textContent = curPullback + ' -- rotation ' + row.angle + '°';
    document.getElementById('statText').textContent =
      'Angle ' + (curIdx + 1) + ' / ' + rows.length + ' (' + row.angle + '° diameter cut through pullback axis)';
  } else {
    document.getElementById('imgGt').src = row.gt;
    document.getElementById('titleText').textContent = curPullback + ' -- frame ' + row.frame;
    document.getElementById('statText').textContent =
      'Frame ' + (curIdx + 1) + ' / ' + rows.length + '   |   Dice (excl. bg): ' + fmtDice(row.dice);
  }
  renderSidebar();
  renderFrameStrip();
}

document.getElementById('modeFrameBtn').onclick = () => setMode('frame');
document.getElementById('modeLongBtn').onclick = () => setMode('long');
if (!hasLongitudinal) { document.getElementById('modeLongBtn').style.display = 'none'; }

document.addEventListener('keydown', (e) => {
  const rows = curRows();
  if (e.key === 'ArrowRight') { curIdx = Math.min(curIdx + 1, rows.length - 1); render(); }
  else if (e.key === 'ArrowLeft') { curIdx = Math.max(curIdx - 1, 0); render(); }
});

renderLegend();
setMode('frame');
</script>
</body>
</html>
"""


def build_html(
    manifest: List[dict],
    long_manifest: List[dict],
    class_names: List[str],
    class_colors: List[Tuple[int, int, int]],
    out_path: Path,
) -> None:
    html = (
        _HTML_TEMPLATE
        .replace("__MANIFEST_JSON__", json.dumps(manifest))
        .replace("__LONG_MANIFEST_JSON__", json.dumps(long_manifest))
        .replace("__CLASS_NAMES_JSON__", json.dumps(class_names))
        .replace("__CLASS_COLORS_JSON__", json.dumps(class_colors))
    )
    out_path.write_text(html, encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Build an HTML QC report (OCT | GT | prediction) for a Conditional INR model")
    parser.add_argument("--model_dir", type=str, required=True, help="Folder with the yaml config (+ test_metrics/predictions)")
    parser.add_argument("--pred_dir", type=str, default=None, help="Default: <model_dir>/test_metrics/predictions")
    parser.add_argument("--output_dir", type=str, default=None, help="Default: <model_dir>/test_metrics/html_report")
    parser.add_argument("--env", type=str, default=None, choices=["local", "cluster"])
    parser.add_argument("--overlap", type=float, default=0.5, help="Must match the overlap used when predictions were generated")
    parser.add_argument("--use_postprocessed", action="store_true", help="Prefer *_postprocessed.nii.gz predictions when present")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay transparency for GT/prediction color coding")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--max_pullbacks", type=int, default=None, help="Only process the first N test pullbacks (debugging)")
    parser.add_argument("--pullbacks", type=str, nargs="+", default=None, help="Only process these specific pullback IDs")
    parser.add_argument("--no_longitudinal", action="store_true",
                         help="Skip building the rotating longitudinal (L-mode) prediction view (on by default)")
    parser.add_argument("--long_angle_step", type=int, default=15,
                         help="Degrees between longitudinal rotation angles, spanning 0-165 deg (default 15 -> 12 angles). "
                              "A diameter cut at angle a is identical to the one at a+180, so only 0-180 is needed.")
    parser.add_argument("--eval_resolution", type=str, default="native", choices=["native", "encoder"],
                         help="Must match the --eval_resolution Pullback_prediction.py used to generate these "
                              "predictions. 'native' (default): predictions and ground truth are both native "
                              "resolution, unchanged behaviour. 'encoder': predictions are at the model's own "
                              "encoder input resolution (cfg.resize_to) -- ground truth (and the raw OCT image) "
                              "are downsampled to match (nearest-neighbour for labels, bilinear for the image, "
                              "same functions training/Pullback_prediction.py use) before rendering.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    yaml_path = next(iter(list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))), None)
    if yaml_path is None:
        raise FileNotFoundError(f"No yaml config found in {model_dir}")
    with open(yaml_path) as f:
        yml = yaml.safe_load(f)
    if args.env:
        yml["env"] = args.env
    cfg = build_config_from_yaml(yml)

    # Mirrors Pullback_prediction.py's own output-folder naming exactly (overlap suffix,
    # plus "_encoderres" for --eval_resolution=encoder) so this script's default pred_dir
    # actually finds what that script wrote, and so the two eval_resolution modes' outputs
    # (predictions AND this report) never land in, or overwrite, the same folder.
    overlap_suffix   = str(args.overlap).replace(".", "")
    eval_dir_suffix  = "_encoderres" if args.eval_resolution == "encoder" else ""
    test_metrics_dir = model_dir / f"test_metrics_{overlap_suffix}{eval_dir_suffix}"

    pred_dir   = Path(args.pred_dir) if args.pred_dir else test_metrics_dir / "predictions"
    output_dir = Path(args.output_dir) if args.output_dir else test_metrics_dir / "html_report"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Distinct filename too (not just a distinct folder) -- belt-and-suspenders so an
    # explicit --output_dir pointed at the same place for both modes still can't clobber
    # the other mode's report.
    report_name = "report.html" if args.eval_resolution == "native" else f"report_{args.eval_resolution}res.html"

    if not pred_dir.exists():
        raise FileNotFoundError(
            f"{pred_dir} does not exist -- run Pullback_prediction.py first with the same "
            f"--eval_resolution {args.eval_resolution} to generate predictions "
            f"(or pass --pred_dir to point at an existing folder of *_pred_overlap*.nii.gz files)"
        )

    label_mapping = load_label_mapping(yml)
    pullback_to_frames = load_test_split(yml)
    if args.pullbacks:
        pullback_to_frames = {k: v for k, v in pullback_to_frames.items() if k in args.pullbacks}
    if args.max_pullbacks:
        pullback_to_frames = dict(list(pullback_to_frames.items())[: args.max_pullbacks])
    print(f"[INFO] {len(pullback_to_frames)} test pullbacks selected (env={yml['env']})")

    data_roots = yml["paths"][yml["env"]]["data_root"]
    num_classes = int(yml["model_seg_decoder"]["out_channels"])
    class_names = get_class_names(num_classes)
    class_colors = get_class_colors(num_classes)

    manifest: List[dict] = []
    long_manifest: List[dict] = []
    n_done = 0
    for pid, ann_frames in pullback_to_frames.items():
        npz_path = find_pullback_npz(pid, data_roots)
        if npz_path is None:
            print(f"[WARN] {pid}: no .npz found in {data_roots} -- skipping")
            continue

        pred_path = find_pred_file(pid, pred_dir, args.overlap, args.use_postprocessed)
        if pred_path is None:
            print(f"[WARN] {pid}: no prediction .nii.gz found in {pred_dir} -- skipping "
                  f"(run Pullback_prediction.py first)")
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

        # Same eval_resolution="encoder" ground-truth downsampling as Pullback_prediction.py:
        # nearest-neighbour for labels (crisp boundaries, no invented intermediate classes),
        # bilinear for the raw image (rendering only). No-op for "native" or when
        # cfg.resize_to == 0 (e.g. the native704 model has nothing to downsample to).
        if args.eval_resolution == "encoder" and cfg.resize_to:
            seg_eval    = resize_labels(torch.from_numpy(seg), cfg.resize_to).numpy()
            volume_eval = resize_volume(torch.from_numpy(volume), cfg.resize_to).numpy()
        else:
            seg_eval    = seg
            volume_eval = volume

        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))  # [D,H,W]
        assert pred.shape[1:] == seg_eval.shape[1:], (
            f"{pid}: pred grid {pred.shape[1:]} != ground-truth grid {seg_eval.shape[1:]} "
            f"(eval_resolution={args.eval_resolution}, cfg.resize_to={cfg.resize_to}) -- "
            f"was {pred_path.name} generated with --eval_resolution {args.eval_resolution}?"
        )

        D = volume.shape[0]
        valid_frames = sorted(z for z in set(ann_frames) if 0 <= z < D)
        skipped = sorted(set(ann_frames) - set(valid_frames))
        if skipped:
            print(f"[WARN] {pid}: annotated frame(s) {[z + 1 for z in skipped]} out of range (D={D}) -- skipping those")
        if not valid_frames:
            print(f"[WARN] {pid}: no valid annotated frames -- skipping pullback")
            continue

        pid_img_dir = images_dir / pid
        pid_img_dir.mkdir(parents=True, exist_ok=True)

        for z in valid_frames:
            gt_frame = seg_eval[z]
            pred_frame = pred[z]

            raw_rgb = np.stack([to_uint8_display(volume_eval[z])] * 3, axis=-1)
            colored_gt = colorize(gt_frame, class_colors)
            colored_pred = colorize(pred_frame, class_colors)
            overlay_gt = blend_overlay(raw_rgb, colored_gt, gt_frame > 0, args.alpha)
            overlay_pred = blend_overlay(raw_rgb, colored_pred, pred_frame > 0, args.alpha)

            frame_1based = z + 1
            raw_name = f"frame{frame_1based:04d}_raw.jpg"
            gt_name = f"frame{frame_1based:04d}_gt.jpg"
            pred_name = f"frame{frame_1based:04d}_pred.jpg"

            save_jpg(raw_rgb, pid_img_dir / raw_name, args.jpeg_quality)
            save_jpg(overlay_gt, pid_img_dir / gt_name, args.jpeg_quality)
            save_jpg(overlay_pred, pid_img_dir / pred_name, args.jpeg_quality)

            dice = frame_mean_dice(gt_frame, pred_frame, num_classes)
            manifest.append({
                "pullback": pid,
                "frame": frame_1based,
                "dice": None if np.isnan(dice) else round(dice, 4),
                "raw": f"images/{pid}/{raw_name}",
                "gt": f"images/{pid}/{gt_name}",
                "pred": f"images/{pid}/{pred_name}",
            })

        if not args.no_longitudinal:
            angles = list(range(0, 180, args.long_angle_step))
            long_entries = build_longitudinal_images(
                volume_eval, pred, angles, class_colors, args.alpha, pid, pid_img_dir, args.jpeg_quality,
            )
            long_manifest.extend(long_entries)
            print(f"    + longitudinal view: {len(angles)} angles x {volume_eval.shape[0]} frames")

        n_done += 1
        print(f"[{n_done}/{len(pullback_to_frames)}] {pid}: {len(valid_frames)} frames rendered")

    if not manifest:
        print("[ERROR] No frames were rendered -- nothing to write.")
        return

    report_path = output_dir / report_name
    build_html(manifest, long_manifest, class_names, class_colors, report_path)
    print(f"\n[INFO] Wrote {report_path} ({len(manifest)} frames across {n_done} pullbacks)")
    print(f"[INFO] Open it directly in a browser (file://{report_path.resolve()})")


if __name__ == "__main__":
    main()
