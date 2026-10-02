"""Find image / label sources for planned pullbacks.

`find_image` looks a pullback up in a two-level index of each root (`<root>/<id>.<suf>` and
`<root>/*/<id>.<suf>`) instead of globbing per id: one scan of the real root costs ~0.1 s,
300 globs cost ~30 s. The index is cached per realpath'd root for the life of the process
(source trees are read-only during a build); `clear_index_cache()` drops it.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from ..config import StoreCfg
from .excel import XL_FRAMES_COLUMNS

IMAGE_SUFFIXES = (".nii.gz", ".nii", ".dcm", ".npz")
# coarse format from the file name; the worker replaces it with the exact
# `probe_image` format (niigz-rgb24 | niigz-uint8 | niigz-nibabel | dcm-rgb | dcm-gray | npz)
FORMAT_FROM_SUFFIX = {".nii.gz": "niigz", ".nii": "niigz", ".dcm": "dcm", ".npz": "npz"}

_ROOT_INDEX: dict[str, dict[str, list[str]]] = {}


def clear_index_cache() -> None:
    """Forget the cached directory indices (only needed when a source tree changes)."""
    _ROOT_INDEX.clear()


def _root_index(root: str) -> dict[str, list[str]]:
    """{file name: [<root>/<name>, then <root>/<subdir>/<name> sorted]} for one root."""
    key = os.path.realpath(root)
    index = _ROOT_INDEX.get(key)
    if index is not None:
        return index
    direct: dict[str, list[str]] = {}
    nested: dict[str, list[str]] = {}
    subdirs: list[str] = []
    try:
        with os.scandir(key) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    subdirs.append(entry.path)
                else:
                    direct.setdefault(entry.name, []).append(entry.path)
    except OSError as ex:
        # an unmounted / unreadable root must not masquerade as "every pullback is missing",
        # and must not be cached: the mount may come back within the process
        warnings.warn(f"cannot scan source root {root}: {type(ex).__name__}: {ex}",
                      UserWarning, stacklevel=3)
        return {}
    for subdir in sorted(subdirs):
        try:
            with os.scandir(subdir) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        nested.setdefault(entry.name, []).append(entry.path)
        except OSError:
            continue
    index = {name: paths + nested.get(name, []) for name, paths in direct.items()}
    for name, paths in nested.items():
        index.setdefault(name, paths)
    _ROOT_INDEX[key] = index
    return index


def find_image(pullback_id: str, roots) -> str | None:
    """Root order, then suffix order, `<root>/<id>.<suf>` before `<root>/*/<id>.<suf>`; warn on duplicates."""
    if isinstance(roots, str):
        roots = (roots,)
    hits: list[str] = []
    for root in roots:
        index = _root_index(root)
        for suffix in IMAGE_SUFFIXES:
            hits.extend(index.get(pullback_id + suffix, ()))
    if not hits:
        return None
    if len(hits) > 1:
        warnings.warn(f"{pullback_id}: {len(hits)} candidate images {hits}; using {hits[0]}",
                      UserWarning, stacklevel=2)
    return hits[0]


def find_label(pullback_id: str, label_root: str, suffix: str) -> str | None:
    """`<label_root>/<id><suffix>.nii.gz` or None."""
    path = os.path.join(label_root, f"{pullback_id}{suffix}.nii.gz")
    return path if os.path.exists(path) else None


def source_format_from_path(path: str) -> str:
    """Coarse source format from the file name ("niigz" | "dcm" | "npz" | "")."""
    for suffix, fmt in FORMAT_FROM_SUFFIX.items():
        if path.endswith(suffix):
            return fmt
    return ""


def resolve_sources(table: pd.DataFrame, cfg: StoreCfg) -> pd.DataFrame:
    """Add `source_image`, `source_label`, `source_format`, `build_status` (missing_image / missing_label / ok)."""
    out = table.copy()
    splits = out["split"] if "split" in out.columns else ["train"] * len(out)
    images, labels, formats, statuses = [], [], [], []
    for pullback_id, split in zip(out["pullback_id"], splits):
        image = find_image(pullback_id, cfg.image_roots)
        label = None if split == "unlabelled" else find_label(
            pullback_id, cfg.label_root, cfg.label_suffix)
        images.append(image or "")
        labels.append(label or "")
        formats.append(source_format_from_path(image) if image else "")
        if image is None:
            statuses.append("missing_image")
        elif label is None and split != "unlabelled":
            statuses.append("missing_label")
        else:
            statuses.append("ok")
    out["source_image"] = images
    out["source_label"] = labels
    out["source_format"] = formats
    out["build_status"] = statuses
    return out


def enumerate_unlabelled(roots, known_ids, skip_dirs=()) -> pd.DataFrame:
    """Every image under the roots (`<root>/<id>.<suf>`, `<root>/*/<id>.<suf>`) whose id is not in
    `known_ids`: one row each with split="unlabelled", patient_id = parent dir name, sorted by id.
    Files inside any of `skip_dirs` (the label root, which sits under the image root on the
    real tree) are never images and are ignored.

    The id is the file name without its suffix. When the same file name sits in several directories
    (`RUMC-00474-1/IMG001.nii.gz`, `RUMC-00744-1/IMG001.nii.gz`, ...) every one of them is kept under
    `<parent dir>__<name>`; a name with a single directory keeps its plain id. Within one directory
    the suffix order of `IMAGE_SUFFIXES` decides. An id that still collides after prefixing (the same
    parent name under two roots) keeps the first root's file, with a warning.

    Same columns as `resolve_sources(load_split_table(...))` so the two tables concatenate: the
    Excel-only fields carry sentinels (`excel_row`/`excel_id`/`excel_n_frames` -1, no manual frames,
    zero class counts, `dem_age` NaN, flags -1). `sidecar_n_frames` is the frame count of the
    `<id>.json` sidecar next to the image (`nifti.dim[2]`), -1 without one; it only feeds the
    size estimate of the build.
    """
    if isinstance(roots, str):
        roots = (roots,)
    known = set(known_ids)
    skip = tuple(os.path.realpath(d) for d in skip_dirs)
    hits: dict[str, dict[str, str]] = {}          # name -> {parent dir: image path}
    for root in roots:
        index = _root_index(root)
        for suffix in IMAGE_SUFFIXES:
            for file_name, paths in index.items():
                if not file_name.endswith(suffix):
                    continue
                name = file_name[: -len(suffix)]
                if name in known:
                    continue
                per_dir = hits.setdefault(name, {})
                for path in paths:
                    if os.path.realpath(os.path.dirname(path)) in skip:
                        continue
                    per_dir.setdefault(os.path.dirname(path), path)
    images: dict[str, str] = {}
    for name, per_dir in hits.items():
        if len(per_dir) == 1:
            images[name] = next(iter(per_dir.values()))
            continue
        for parent, path in per_dir.items():
            pullback_id = f"{os.path.basename(parent)}__{name}"
            if pullback_id in images:
                warnings.warn(f"{pullback_id}: {path} collides with {images[pullback_id]}; keeping the latter",
                              UserWarning, stacklevel=2)
                continue
            images[pullback_id] = path
    ids = sorted(images)
    paths = [images[i] for i in ids]
    n = len(ids)
    out = pd.DataFrame({
        "pullback_id": pd.array(ids, dtype="str"),
        "patient_id": pd.array([os.path.basename(os.path.dirname(p)) for p in paths], dtype="str"),
        "split": pd.array(["unlabelled"] * n, dtype="str"),
        "excel_row": np.full(n, -1, dtype=np.int32),
        "excel_id": np.full(n, -1, dtype=np.int32),
        "excel_n_frames": np.full(n, -1, dtype=np.int32),
        "n_manual": np.zeros(n, dtype=np.int32),
        "manual_frames": [[] for _ in range(n)],
    })
    for column in XL_FRAMES_COLUMNS:
        out[column] = np.zeros(n, dtype=np.int16)
    out["dem_age"] = np.full(n, np.nan, dtype=np.float32)
    for column in ("dem_gender", "mh_mi", "mh_dm", "mh_ht", "mh_hc", "lipid_lowering", "dataset"):
        out[column] = np.full(n, -1, dtype=np.int8)
    out["source_image"] = pd.array(paths, dtype="str")
    out["source_label"] = pd.array([""] * n, dtype="str")
    out["source_format"] = pd.array([source_format_from_path(p) for p in paths], dtype="str")
    out["build_status"] = pd.array(["ok"] * n, dtype="str")
    out["sidecar_n_frames"] = np.array([sidecar_n_frames(p) for p in paths], dtype=np.int32)
    return out


def sidecar_n_frames(image_path: str) -> int:
    """Frame count from the `<id>.json` sidecar written by dcm2niigz (`nifti.dim[2]`), -1 if absent."""
    for suffix in IMAGE_SUFFIXES:
        if image_path.endswith(suffix):
            sidecar = image_path[: -len(suffix)] + ".json"
            break
    else:
        return -1
    try:
        with open(sidecar) as f:
            return int(json.load(f)["nifti"]["dim"][2])
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return -1


def refuse_dst_inside_sources(dst: str, roots) -> None:
    """sys.exit when the realpath of `dst` lies inside any source root (never write under /mnt)."""
    if isinstance(roots, str):
        roots = (roots,)
    dst_real = os.path.realpath(dst)
    for root in roots:
        if not root:
            continue
        root_real = os.path.realpath(root)
        if dst_real == root_real or dst_real.startswith(root_real + os.sep):
            sys.exit(f"REFUSING: store destination {dst} is inside source root {root}")
