"""
preprocess_frames.py — pre-split annotated frames/patches to disk.

2D mode (--mode 2d):
    For each annotated frame z, saves a .npz with:
        frame   : [2*context+1, H, W]  float32  — centre frame + context neighbours
        label   : [H, W]               int64    — segmentation label (remapped if mapping on)

    Boundary frames are edge-repeated, not zero-padded.

3D mode (--mode 3d):
    For each annotated frame z, saves a .npz with:
        patch   : [patch_z, H, W]      float32  — z-window centred on annotated frame
        label   : [H, W]               int64    — segmentation label (remapped if mapping on)
        z_local : scalar               int64    — index of annotated frame within the patch

    patch_z and label mapping are read from the config yaml.
    Files without Segmentation_image are silently skipped.

Usage:
    python preprocess_frames.py --config Config/config.yaml   --mode 2d
    python preprocess_frames.py --config Config/config_3D.yaml --mode 3d
    python preprocess_frames.py --config Config/config.yaml   --mode 2d --sets training validation testing
    python preprocess_frames.py --config Config/config_3D.yaml --mode 3d --n_procs 6
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Circular mask
# ---------------------------------------------------------------------------

def _create_outside_mask(h: int, w: int) -> np.ndarray:
    """Return a [H, W] bool array: True = outside the circular OCT field-of-view.

    The circle is centered at the image centre and just touches the image edges.
    Pixels outside this circle are the Abbott watermark / black corners and
    should be zeroed out in both the image and the segmentation.
    """
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    radius = min(cx, cy, w - cx, h - cy)
    Y, X   = np.ogrid[:h, :w]
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) >= radius + 0.5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_pullback_to_frames(df: pd.DataFrame, sets) -> dict:
    """Return {pullback_id: [0-based frame indices]} filtered to given sets."""
    df = df[df["set"].str.lower().isin([s.lower() for s in sets])]
    pullback_to_frames: dict = {}
    for _, row in df.iterrows():
        pid    = str(row["pullback"])
        frames = [
            int(f.strip()) - 1
            for f in str(row["frames"]).split(",")
            if f.strip().isdigit()
        ]
        pullback_to_frames.setdefault(pid, []).extend(frames)
    return pullback_to_frames


def _build_label_lut(mapping: dict, seg_max: int) -> np.ndarray:
    """Build a lookup table that remaps raw labels to model class indices."""
    lut_size = max(seg_max, max(mapping.keys())) + 1
    lut = np.arange(lut_size, dtype=np.int64)
    for old, new in mapping.items():
        lut[old] = new
    return lut


# ---------------------------------------------------------------------------
# 2D preprocessing
# ---------------------------------------------------------------------------

def preprocess_2d(yml: dict, args) -> None:
    env     = yml["env"]
    paths   = yml["paths"][env]
    context = yml.get("context_frames", 0)

    frames_folder = Path(paths.get("frames_folder", ""))
    if not str(frames_folder):
        raise ValueError("paths.frames_folder is not set in the config")
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Label mapping
    seg               = yml["model_seg_decoder"]
    mapping_activated = bool(seg.get("mapping_activated", True))
    label_mapping     = (
        {int(k): int(v) for k, v in seg["mapping"].items()}
        if mapping_activated and seg.get("mapping") else None
    )

    df = pd.read_excel(paths["split_excel"])
    df.columns = [c.strip().lower() for c in df.columns]
    pullback_to_frames = _build_pullback_to_frames(df, args.sets)

    folders = paths["data_root"]
    if isinstance(folders, str):
        folders = [folders]

    total_saved = 0
    for folder in folders:
        for npz_path in sorted(Path(folder).glob("*.npz")):
            pid = npz_path.stem.replace("_circ_gray", "")
            if pid not in pullback_to_frames:
                continue

            print(f"Processing {pid} …")
            data = np.load(npz_path, allow_pickle=False)

            vol = data["Volume_input_image"]
            if vol.ndim == 4:
                vol = vol[0]
            vol = vol.astype(np.float32)

            segm = data["Segmentation_image"]
            if segm.ndim == 4:
                segm = segm[0]
            segm = segm.astype(np.int64)

            if label_mapping:
                lut  = _build_label_lut(label_mapping, int(segm.max()))
                segm = lut[segm]

            D = vol.shape[0]
            for z in pullback_to_frames[pid]:
                out_path = frames_folder / f"{pid}_frame{z:04d}.npz"
                if out_path.exists():
                    continue

                stack = np.stack([
                    vol[max(0, min(D - 1, zi))]
                    for zi in range(z - context, z + context + 1)
                ], axis=0)                      # [2*context+1, H, W]

                np.savez_compressed(out_path, frame=stack, label=segm[z])
                total_saved += 1

    print(f"\nDone. {total_saved} frame files saved to {frames_folder}")


# ---------------------------------------------------------------------------
# 3D preprocessing
# ---------------------------------------------------------------------------

def _process_one_pullback_3d(
    npz_path: Path,
    pid: str,
    ann_frames: list,
    patch_z: int,
    buf_size: int,
    patches_folder: Path,
    label_mapping: dict,
) -> tuple:
    """Process one pullback's annotated frames into 3D patch files.

    Returns (pid, n_saved, status) where status is one of
    "ok" / "no_segmentation" / f"error: {msg}" — never raises, so a single
    corrupt/unexpected pullback can't take down the whole run (parallel or not).
    """
    try:
        raw = np.load(npz_path, allow_pickle=False)
        if "Segmentation_image" not in raw.files:
            return pid, 0, "no_segmentation"

        vol = raw["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]
        vol = vol.astype(np.float32)

        segm = raw["Segmentation_image"]
        if segm.ndim == 4:
            segm = segm[0]
        segm = segm.astype(np.int64)

        if label_mapping:
            lut  = _build_label_lut(label_mapping, int(segm.max()))
            segm = lut[segm]

        D = vol.shape[0]
        n_saved = 0

        for z in ann_frames:
            out_path = patches_folder / f"{pid}_frame{z:04d}_patch3d.npz"
            if out_path.exists():
                continue

            if not (0 <= z < D):
                print(f"  Frame {z} out of range [0, {D}) for {pid} — skipping")
                continue

            # Centre the double buffer on z, clamp to volume bounds
            z_start = int(np.clip(z - buf_size // 2, 0, max(0, D - buf_size)))
            z_end   = z_start + buf_size
            z_local = z - z_start          # position of annotated frame in buffer

            patch = vol[z_start:z_end].copy()   # [buf_size or less, H, W]

            # Edge-repeat pad when pullback is shorter than buf_size
            if patch.shape[0] < buf_size:
                pad   = np.broadcast_to(
                    patch[-1:],
                    (buf_size - patch.shape[0],) + patch.shape[1:],
                )
                patch = np.concatenate([patch, pad.copy()], axis=0)

            # Build labels volume: 255 everywhere, fill ALL annotated frames in buffer
            H, W         = patch.shape[1], patch.shape[2]
            patch_labels = np.full((buf_size, H, W), 255, dtype=np.int64)
            for z_ann in ann_frames:
                if z_start <= z_ann < z_end:
                    patch_labels[z_ann - z_start] = segm[z_ann]

            # Apply circular mask: zero image outside FOV, set labels outside to background (0)
            outside      = _create_outside_mask(H, W)               # [H, W] bool
            patch[:, outside] = 0.0
            for zi in range(buf_size):
                if (patch_labels[zi] != 255).any():                  # annotated frame
                    patch_labels[zi][outside] = 0

            np.savez_compressed(
                out_path,
                patch   = patch,             # [2*patch_z, H, W]
                labels  = patch_labels,      # [2*patch_z, H, W] — 255 for unannotated frames
                z_local = np.array(z_local, dtype=np.int64),
            )
            n_saved += 1

        return pid, n_saved, "ok"

    except Exception as e:                # noqa: BLE001 -- deliberately broad: one bad
        return pid, 0, f"error: {e}"       # pullback must never kill the whole run


# Globals set by _mp_init, read by _mp_process -- avoids re-pickling the same
# (small) read-only config into every task, same pattern as
# postprocessing_conditionalinr.py's pixel_post_parallel.
_MP_PATCH_Z = _MP_BUF_SIZE = _MP_PATCHES_FOLDER = _MP_LABEL_MAPPING = None


def _mp_init(patch_z, buf_size, patches_folder, label_mapping):
    global _MP_PATCH_Z, _MP_BUF_SIZE, _MP_PATCHES_FOLDER, _MP_LABEL_MAPPING
    _MP_PATCH_Z, _MP_BUF_SIZE = patch_z, buf_size
    _MP_PATCHES_FOLDER, _MP_LABEL_MAPPING = patches_folder, label_mapping


def _mp_process(args) -> tuple:
    npz_path, pid, ann_frames = args
    print(f"Processing {pid} …")
    return _process_one_pullback_3d(
        npz_path, pid, ann_frames, _MP_PATCH_Z, _MP_BUF_SIZE, _MP_PATCHES_FOLDER, _MP_LABEL_MAPPING,
    )


def preprocess_3d(yml: dict, args) -> None:
    env     = yml["env"]
    paths   = yml["paths"][env]
    tr      = yml["training_segmentation"]
    patch_z = int(tr.get("patch_z", 32))

    patches_folder = Path(paths.get("patches_folder_3d", ""))
    if not str(patches_folder):
        raise ValueError("paths.patches_folder_3d is not set in the config")
    patches_folder.mkdir(parents=True, exist_ok=True)

    # Label mapping
    seg               = yml["model_seg_decoder"]
    mapping_activated = bool(seg.get("mapping_activated", True))
    label_mapping     = (
        {int(k): int(v) for k, v in seg["mapping"].items()}
        if mapping_activated and seg.get("mapping") else None
    )

    df = pd.read_excel(paths["split_excel"])
    df.columns = [c.strip().lower() for c in df.columns]
    pullback_to_frames = _build_pullback_to_frames(df, args.sets)

    folders = paths["data_root"]
    if isinstance(folders, str):
        folders = [folders]

    buf_size = 2 * patch_z   # double buffer so annotated frame can land anywhere in [0, patch_z-1]

    # Resolve which npz files to process up front (same pid-in-split filtering as before)
    tasks = []
    for folder in folders:
        for npz_path in sorted(Path(folder).glob("*.npz")):
            pid = npz_path.stem.replace("_circ_gray", "")
            if pid in pullback_to_frames:
                tasks.append((npz_path, pid, pullback_to_frames[pid]))

    total_saved = total_skip = total_error = 0
    n_procs = max(1, args.n_procs)

    if n_procs == 1:
        for npz_path, pid, ann_frames in tasks:
            print(f"Processing {pid} …")
            _, n_saved, status = _process_one_pullback_3d(
                npz_path, pid, ann_frames, patch_z, buf_size, patches_folder, label_mapping,
            )
            total_saved += n_saved
            if status == "no_segmentation":
                print(f"  Skipping {pid} — no Segmentation_image")
                total_skip += 1
            elif status != "ok":
                print(f"  ERROR processing {pid}: {status}")
                total_error += 1
    else:
        print(f"Processing {len(tasks)} pullbacks with {n_procs} parallel workers...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_procs, initializer=_mp_init,
                      initargs=(patch_z, buf_size, patches_folder, label_mapping)) as pool:
            for pid, n_saved, status in pool.imap_unordered(_mp_process, tasks):
                total_saved += n_saved
                if status == "no_segmentation":
                    print(f"  Skipping {pid} — no Segmentation_image")
                    total_skip += 1
                elif status != "ok":
                    print(f"  ERROR processing {pid}: {status}")
                    total_error += 1

    print(
        f"\nDone. {total_saved} patch files saved to {patches_folder}"
        + (f"  ({total_skip} pullbacks skipped — no segmentation)" if total_skip else "")
        + (f"  ({total_error} pullbacks FAILED — see ERROR lines above)" if total_error else "")
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-extract annotated frames/patches from pullback .npz files"
    )
    parser.add_argument("--config",  default="Config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--mode",    default="2d", choices=["2d", "3d"],
                        help="'2d': extract single frames with context  |  '3d': extract z-patches")
    parser.add_argument("--sets",    nargs="+", default=["training", "validation"],
                        help="Which dataset splits to preprocess")
    parser.add_argument("--n_procs", type=int, default=1,
                        help="Parallel workers across pullbacks (--mode 3d only; --mode 2d stays "
                             "single-process). 1 = original sequential behavior. Each worker holds "
                             "one full pullback volume in memory (~1-2GB) at a time.")
    args = parser.parse_args()

    with open(args.config) as f:
        yml = yaml.safe_load(f)

    if args.mode == "2d":
        preprocess_2d(yml, args)
    else:
        preprocess_3d(yml, args)
