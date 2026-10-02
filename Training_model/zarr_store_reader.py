"""Thin read-only access to a zarr frame store built by Preprocess_codes_Simone/run_build_store.py.

Deliberately self-contained (does not import Preprocess_codes_Simone) so Training_model and
Preprocess_codes_Simone stay independent of each other.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pandas as pd
import zarr


def open_store(store_path: str) -> zarr.Group:
    """Open the store's zarr root read-only (consolidated metadata, falling back if absent).

    `store_path` is the build's --dst (holds pullbacks.parquet, dataset_info.json, ...);
    the actual zarr group is one level deeper, at `store_path/store.zarr` (see
    Preprocess_codes_Simone/store/build.py: `store_path = os.path.join(dst, "store.zarr")`).
    """
    zarr_root_path = os.path.join(store_path, "store.zarr")
    try:
        return zarr.open_group(zarr_root_path, mode="r", use_consolidated=True)
    except ValueError:
        return zarr.open_group(zarr_root_path, mode="r", use_consolidated=False)


def load_pullbacks_table(store_path: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(store_path, "pullbacks.parquet"))


def load_norm_stats(store_path: str) -> tuple[float, float]:
    with open(os.path.join(store_path, "dataset_info.json")) as f:
        info = json.load(f)
    return float(info["image_mean"]), float(info["image_std"])


def load_pullback_norm_stats(store_path: str) -> dict[str, tuple[float, float]]:
    """Per-pullback (mean, std) over foreground pixels, derived from manifest.jsonl's already-
    computed image_sum/image_sumsq/n_pixels (build.py's _stream_image, same disc mask used for
    the corpus-wide image_mean/image_std) -- no rebuild, no re-reading any image pixels, just
    arithmetic on numbers already on disk from the original build.

    population std (ddof=0, same as numpy's default .std()): sqrt(E[x^2] - E[x]^2), matching
    the .npz pipeline's per-pullback foreground normalisation (process_volume() in
    Preprocessing_dataconversion.py: foreground_values.mean() / .std() over the whole volume,
    pooled across every frame) -- same formula, same scope (one pullback, all its frames
    pooled), so normalizing against this instead of the single global image_mean/image_std
    makes the zarr-backed path numerically equivalent to the .npz pipeline's normalisation,
    not just similar.

    Last record wins for a pullback_id that appears more than once (resumed builds) --
    matches store/build.py's own _read_manifest dedup behaviour.
    """
    records: dict[str, dict] = {}
    with open(os.path.join(store_path, "manifest.jsonl")) as f:
        for line in f:
            record = json.loads(line)
            records[record["pullback_id"]] = record

    stats: dict[str, tuple[float, float]] = {}
    for pullback_id, record in records.items():
        if record["status"] != "ok":
            continue
        n = record["n_pixels"]
        if n <= 0:
            continue
        mean = record["image_sum"] / n
        var  = record["image_sumsq"] / n - mean * mean
        std  = var ** 0.5 if var > 0 else 0.0
        stats[pullback_id] = (float(mean), float(std))
    return stats


def _store_bytes(store_path: str) -> int:
    with open(os.path.join(store_path, "dataset_info.json")) as f:
        info = json.load(f)
    return int(info["bytes"]["image"]) + int(info["bytes"]["label"])


def ensure_local_copy(store_path: str, local_cache_dir: str) -> str:
    """rsync `store_path` -> `local_cache_dir` once (node-local NVMe), return the path to read from.

    Mirrors oct-fm's train_seg.sbatch: copy the store to node-local scratch once per job, then
    read from there for the rest of training (network storage is otherwise the bottleneck).
    Skips the copy if already done (dataset_info.json present at the destination). Falls back to
    `store_path` unchanged — logging a warning, not raising — if there isn't enough free space,
    so a misconfigured/undersized local cache never crashes the job.
    """
    info_path = os.path.join(local_cache_dir, "dataset_info.json")
    if os.path.exists(info_path):
        print(f"zarr_local_cache: already present at {local_cache_dir}, skipping copy")
        return local_cache_dir

    needed = _store_bytes(store_path)
    os.makedirs(local_cache_dir, exist_ok=True)
    free = shutil.disk_usage(local_cache_dir).free
    headroom = 1.1
    if free < needed * headroom:
        print(f"WARNING zarr_local_cache: {local_cache_dir} has {free / 1e9:.0f}GB free, "
              f"need ~{needed * headroom / 1e9:.0f}GB for the store — "
              f"falling back to reading directly from {store_path}")
        return store_path

    print(f"zarr_local_cache: rsync {store_path} -> {local_cache_dir} "
          f"(~{needed / 1e9:.0f}GB, one-time)")
    subprocess.run(["rsync", "-a", "--info=progress2", f"{store_path}/", f"{local_cache_dir}/"],
                   check=True)
    print("zarr_local_cache: rsync done")
    return local_cache_dir


def read_image_window(root: zarr.Group, pullback_id: str, z0: int, z1: int):
    return root[pullback_id]["image"][z0:z1]


def read_label_frame(root: zarr.Group, pullback_id: str, z: int):
    return root[pullback_id]["label"][z]
