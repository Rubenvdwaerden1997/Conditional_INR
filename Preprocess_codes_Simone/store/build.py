"""`oct-fm build-store`: plan -> preflight -> parallel per-pullback workers -> finalize.

One pullback is one unit of work and of resume: a worker streams its image straight from the raw
source into `store.zarr/<id>/image` in shard-aligned blocks (never holding the volume), writes its
`frames_parts/<id>.parquet` rows and one manifest record. Nothing but the destination guard and the
preflight stops the run: an unreadable source is `excluded`, any other failure is an `error`, and
both are recorded, listed in `excluded.json` and printed at the end. A worker killed by the OS
writes no record at all: its id is printed, the run continues and a resume rebuilds that pullback.

The parent process never touches pixels. It opens the root group once (so `dst` exists), writes
`lut.npz`, and hands every worker a plain job dict; each worker opens its own zarr handle and
builds the LUT once per process in the pool initializer.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Config, StoreCfg
from ..data.labels import CLASS_NAMES, CLASS_TOKENS, N_CLASSES, bbox_columns, px_columns
from ..preprocess.colormap import build_vendor_lut, load_lut, save_lut
from ..preprocess.polar import disc_mask
from .excel import load_split_table
from .frame_stats import frame_class_stats, stats_frame
from .locate import enumerate_unlabelled, refuse_dst_inside_sources, resolve_sources
from .qc import orientation_ok, save_overlay_png
from .readers import (BrokenSourceError, derive_z_spacing, iter_image_gray, probe_image,
                      read_label_volume)
from .writer import (FRAME_SHAPE, consolidate, create_pullback, open_root, verify_frames,
                     write_block)

SCHEMA_VERSION = 1
BYTES_PER_FRAME = 275 * 1024          # measured zstd size of one gray Cartesian frame
DICTIONARY_COLUMNS = ("pullback_id", "patient_id", "split")


class BuildError(Exception):
    """A pullback the build refuses (bad geometry, label/image mismatch): status `error`, reason verbatim."""


def main(cfg: Config, jobs: int = 4, limit: int = 0, only: str = "", dry_run: bool = False,
         finalize_only: bool = False) -> None:
    store = cfg.store
    dst = store.dst
    refuse_dst_inside_sources(dst, _source_roots(store))
    if finalize_only:
        finalize(cfg)
        return

    planned = plan(cfg, limit=limit, only=only)
    _print_plan(planned, dst)
    if dry_run:
        print("dry-run: nothing written")
        return

    buildable = planned[planned["build_status"] == "ok"]
    _preflight(dst, int(_frame_estimate(buildable)) * BYTES_PER_FRAME, store)
    os.makedirs(dst, exist_ok=True)
    parts_dir, qc_dir = os.path.join(dst, "frames_parts"), os.path.join(dst, "qc")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)
    store_path = os.path.join(dst, "store.zarr")
    open_root(store_path, "a")                 # create the root group here; every worker opens its own
    lut_path = _write_lut(store, dst)

    records = _read_manifest(dst)
    todo = [_job(row, cfg, store_path, parts_dir, qc_dir, lut_path) for row in buildable.itertuples()
            if not _already_built(records.get(row.pullback_id), parts_dir, row.pullback_id)]
    print(f"{len(todo)} pullbacks to build, jobs={jobs} "
          f"({len(buildable) - len(todo)} already built)", flush=True)

    started = time.time()
    manifest_path = os.path.join(dst, "manifest.jsonl")
    tally = {"ok": 0, "excluded": 0, "error": 0, "died": 0}
    # spawn, not fork: the parent holds an open zarr group (and its io thread), and forking a
    # multi-threaded process is deprecated and deadlock-prone. Workers are self-contained
    # (module-level functions, picklable job dicts, LUT rebuilt by the initializer).
    with open(manifest_path, "a", buffering=1) as manifest, \
            ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                initializer=_worker_init, initargs=(lut_path,)) as pool:
        futures = {pool.submit(build_pullback, job): job["pullback_id"] for job in todo}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except Exception as ex:        # the worker died (OOM kill, segfault): it wrote no record
                tally["died"] += 1
                print(f"!! worker died for {futures[future]}: {type(ex).__name__}: {ex}", flush=True)
                continue
            tally[record["status"]] += 1
            manifest.write(json.dumps(record) + "\n")
            mark = "  " if record["status"] == "ok" else "!!"
            print(f"[{done}/{len(todo)}] {mark} {record['pullback_id']} {record['status']} "
                  f"{record['n_frames']} frames {record['elapsed']:.1f}s "
                  f"{record['reason']}".rstrip(), flush=True)
    print(f"built {len(todo)} pullbacks in {(time.time() - started) / 60:.1f} min: "
          f"{tally['ok']} ok, {tally['excluded']} excluded, {tally['error']} error, "
          f"{tally['died']} workers died")
    finalize(cfg)


def plan(cfg: Config, limit: int = 0, only: str = "") -> pd.DataFrame:
    """Excel table + resolved sources, then every other image under the roots as `unlabelled`
    (`store.include_unlabelled`), filtered by --only/--limit.

    `--only` filters every row; `--limit` then keeps the first N *buildable* rows, so a small trial
    run builds N pullbacks whatever the rows whose image is missing. `sidecar_n_frames` (-1 for the
    sheet rows) only feeds the size estimate.
    """
    store = cfg.store
    planned = resolve_sources(load_split_table(store.excel, store.excel_sheet), store)
    planned["sidecar_n_frames"] = np.full(len(planned), -1, dtype=np.int32)
    if store.include_unlabelled:
        unlabelled = enumerate_unlabelled(store.image_roots, planned["pullback_id"],
                                          skip_dirs=(store.label_root,))
        planned = pd.concat([planned, unlabelled], ignore_index=True)
    if only:
        planned = planned[planned["pullback_id"].str.contains(only, regex=False)]
    if limit:
        planned = planned[planned["build_status"] == "ok"].head(limit)
    return planned.reset_index(drop=True)


def build_pullback(job: dict) -> dict:
    """Worker: ingest one pullback into the store; returns the manifest record (status ok/excluded/error)."""
    started = time.time()
    record = {"pullback_id": job["pullback_id"], "status": "ok", "reason": "",
              "source_image": job["source_image"], "source_label": job["source_label"],
              "n_frames": -1, "attrs": {}, "image_sum": 0.0, "image_sumsq": 0.0, "n_pixels": 0,
              "bytes_image": 0, "bytes_label": 0, "qc": [], "orientation_ok": False,
              "excel_shape_mismatch": False, "elapsed": 0.0}
    root = None
    try:
        root = open_root(job["store_path"], "a")
        _ingest(job, record, root)
    except BrokenSourceError as ex:
        record.update(status="excluded", reason=str(ex))
        _remove_group(root, job["pullback_id"])
    except BuildError as ex:
        record.update(status="error", reason=str(ex))
        _remove_group(root, job["pullback_id"])
    except Exception as ex:                                    # anything else: never stop the run
        record.update(status="error", reason=f"{type(ex).__name__}: {ex}")
        _remove_group(root, job["pullback_id"])
    record["elapsed"] = time.time() - started
    return record


def finalize(cfg: Config) -> None:
    """manifest + plan -> pullbacks.parquet, frames.parquet, qc/orientation.json, excluded.json,
    dataset_info.json; consolidate last."""
    store = cfg.store
    dst = store.dst
    planned = plan(cfg)
    records = _read_manifest(dst)
    parts_dir = os.path.join(dst, "frames_parts")

    parts = []
    for pullback_id in sorted(i for i, r in records.items() if r["status"] == "ok"):
        part = os.path.join(parts_dir, f"{pullback_id}.parquet")
        if os.path.exists(part):
            parts.append(pd.read_parquet(part))
        else:
            print(f"WARNING {pullback_id}: manifest says ok but {part} is missing; recording an error")
            records[pullback_id] = {**records[pullback_id], "status": "error",
                                    "reason": "frames_parts file missing"}
    frames = pd.concat(parts, ignore_index=True) if parts else _empty_frames()
    _check_frames(frames)
    pq.write_table(_frames_table(frames), os.path.join(dst, "frames.parquet"), compression="zstd")

    pullbacks = _merge_pullbacks(planned, records)
    pq.write_table(_pullbacks_table(pullbacks), os.path.join(dst, "pullbacks.parquet"),
                   compression="zstd")

    excluded = [{"pullback_id": r["pullback_id"], "source_image": r["source_image"],
                 "reason": r["reason"]}
                for r in _by_id([r for r in records.values() if r["status"] in ("excluded", "error")])]
    _write_json(os.path.join(dst, "excluded.json"), excluded)
    orientation = _orientation_report(records, store.orientation_min_contrast)
    _write_json(os.path.join(dst, "qc", "orientation.json"), orientation)
    counts_match = _excel_counts_match(frames, planned)
    info = _dataset_info(cfg, planned, records, frames, orientation, counts_match)
    _write_json(os.path.join(dst, "dataset_info.json"), info)
    consolidate(os.path.join(dst, "store.zarr"))

    print(f"frames.parquet: {info['frames']['total']:,} frames "
          f"({info['frames']['manual']:,} manual) of {info['pullbacks']['ok']} pullbacks; "
          f"mean={info['image_mean']:.1f} std={info['image_std']:.1f}")
    print(f"excel_counts_match: {counts_match['match']}/{counts_match['checked']} labelled pullbacks"
          + ("" if counts_match["all_match"] else
             f"; mismatches: {[m['pullback_id'] for m in counts_match['mismatch']]}"))
    print(f"orientation QC: {orientation['checked']} checked, failed: {orientation['failed']}")
    print(f"excluded.json: {len(excluded)} entries")
    for entry in excluded:
        print(f"  {entry['pullback_id']}: {entry['reason']}")


# ---------------------------------------------------------------- worker
_LUT_CACHE: dict[str, dict] = {}


def _worker_init(lut_path: str) -> None:
    """Pool initializer: build the LUT once per worker process."""
    _worker_lut(lut_path)


def _worker_lut(lut_path: str) -> dict:
    lut = _LUT_CACHE.get(lut_path)
    if lut is None:
        lut = _LUT_CACHE[lut_path] = load_lut(lut_path)
    return lut


def _check_sources(job: dict, record: dict):
    """The volume is the truth: probe it, read the label, refuse what cannot line up."""
    meta = probe_image(job["source_image"])
    n_frames = meta.n_frames
    if n_frames < 1:
        raise BuildError(f"volume has {n_frames} frames")
    if (meta.rows, meta.cols) != FRAME_SHAPE:
        raise BuildError(f"frame shape {meta.rows}x{meta.cols} is not "
                         f"{FRAME_SHAPE[0]}x{FRAME_SHAPE[1]}")
    label = None
    if job["has_label"]:
        label = read_label_volume(job["source_label"])
        if label.shape[0] != n_frames:
            raise BuildError(f"label/image frame count mismatch: label has {label.shape[0]} frames, "
                             f"image has {n_frames}")
        if label.shape[1:] != FRAME_SHAPE:
            raise BuildError(f"label frame shape {label.shape[1]}x{label.shape[2]} is not "
                             f"{FRAME_SHAPE[0]}x{FRAME_SHAPE[1]}")
    manual_frames = [int(frame) for frame in job["manual_frames"]]
    if manual_frames and max(manual_frames) >= n_frames:
        raise BuildError(f"manual frame {max(manual_frames)} is beyond the {n_frames} frames "
                         f"of the volume")
    if job["excel_n_frames"] >= 0 and job["excel_n_frames"] != n_frames:
        record["excel_shape_mismatch"] = True
        print(f"WARNING {job['pullback_id']}: Excel shape says {job['excel_n_frames']} frames, "
              f"the volume has {n_frames}; using the volume", flush=True)
    return meta, label, manual_frames


def _ingest(job: dict, record: dict, root) -> None:
    """check -> create group -> stream image -> label -> verify -> stats -> QC."""
    cfg: StoreCfg = job["cfg"]
    pullback_id = job["pullback_id"]
    meta, label, manual_frames = _check_sources(job, record)
    n_frames = meta.n_frames
    z_spacing, z_spacing_source = derive_z_spacing(n_frames, job["z_spacing_table"])
    attrs = {"schema_version": SCHEMA_VERSION, "pullback_id": pullback_id,
             "patient_id": job["patient_id"], "split": job["split"], "n_frames": n_frames,
             "excel_n_frames": job["excel_n_frames"] if job["excel_n_frames"] >= 0 else None,
             "manual_frames": manual_frames, "has_label": job["has_label"],
             "pixel_spacing_mm": _finite(meta.pixel_spacing_mm), "z_spacing_mm": z_spacing,
             "z_spacing_source": z_spacing_source, "source_image": job["source_image"],
             "source_label": job["source_label"] or None, "source_format": meta.source_format,
             "gray_source": "", "offcurve_frac": 0.0}
    group = create_pullback(root, pullback_id, n_frames, job["has_label"], attrs, cfg)

    qc_frames = [] if label is None else _qc_frames(manual_frames, n_frames,
                                                    cfg.qc_frames_per_pullback)
    verify = sorted({0, n_frames // 2, n_frames - 1})
    kept: dict[int, np.ndarray] = {}
    gray_source, offcurve, stats = _stream_image(job, group["image"], n_frames,
                                                 sorted(set(verify) | set(qc_frames)), kept)
    attrs.update(gray_source=gray_source, offcurve_frac=offcurve)
    group.update_attributes({"gray_source": gray_source, "offcurve_frac": offcurve})

    if label is not None:
        for start in range(0, n_frames, cfg.shard_frames):
            write_block(group["label"], start, label[start:start + cfg.shard_frames])
        verify_frames(group["label"], {frame: label[frame] for frame in verify})
    verify_frames(group["image"], {frame: kept[frame] for frame in verify})

    if label is not None:
        counts, bboxes = frame_class_stats(label)
    else:
        counts = np.zeros((n_frames, N_CLASSES), dtype=np.int32)
        bboxes = np.full((n_frames, N_CLASSES, 4), -1, dtype=np.int16)
    stats_frame(pullback_id, job["patient_id"], job["split"], n_frames, manual_frames,
                job["has_label"], z_spacing, counts, bboxes).to_parquet(
        os.path.join(job["parts_dir"], f"{pullback_id}.parquet"), index=False)

    qc = []
    for frame in qc_frames:
        ok, detail = orientation_ok(kept[frame], label[frame], cfg.orientation_min_contrast)
        png = f"{pullback_id}_f{frame:04d}.png"
        save_overlay_png(os.path.join(job["qc_dir"], png), kept[frame], label[frame],
                         f"{pullback_id} frame {frame}")
        qc.append({"frame": frame, "ok": ok, "png": png,
                   **{key: _finite(value) for key, value in detail.items()}})

    record.update(n_frames=n_frames, attrs=attrs, qc=qc,
                  orientation_ok=bool(qc) and all(entry["ok"] for entry in qc),
                  bytes_image=_array_bytes(job["store_path"], pullback_id, "image"),
                  bytes_label=_array_bytes(job["store_path"], pullback_id, "label"), **stats)


def _stream_image(job: dict, array, n_frames: int, keep: list[int],
                  kept: dict[int, np.ndarray]) -> tuple[str, float, dict]:
    """Stream the source into `array` in shard-aligned blocks; collect disc statistics and `keep` frames."""
    cfg: StoreCfg = job["cfg"]
    mask = disc_mask(FRAME_SHAPE[0], tuple(job["disc_center"]), job["disc_radius"])
    stats = {"image_sum": 0.0, "image_sumsq": 0.0, "n_pixels": 0}
    pending: list[np.ndarray] = []
    written = pending_frames = 0
    offcurve_total = 0.0
    gray_source = ""
    for batch, offcurve, source in iter_image_gray(job["source_image"], _worker_lut(job["lut_path"]),
                                                   cfg.read_batch):
        gray_source = source
        offcurve_total += offcurve * len(batch)
        batch[:, ~mask] = 0                                      # zero the vendor overlay / corners
        for frame_pixels in batch[:, mask]:                     # one frame at a time: float64 sums,
            values = frame_pixels.astype(np.float64)            # bounded temporaries
            stats["image_sum"] += float(values.sum())
            stats["image_sumsq"] += float((values * values).sum())
            stats["n_pixels"] += values.size
        for frame in keep:
            if written + pending_frames <= frame < written + pending_frames + len(batch):
                kept[frame] = np.array(batch[frame - written - pending_frames])
        pending.append(batch)
        pending_frames += len(batch)
        if pending_frames >= cfg.shard_frames:
            block = np.concatenate(pending)
            full = len(block) - len(block) % cfg.shard_frames
            write_block(array, written, block[:full])
            written += full
            pending = [block[full:]] if full < len(block) else []
            pending_frames = len(block) - full
    if pending_frames:
        write_block(array, written, np.concatenate(pending))
        written += pending_frames
    if written != n_frames:
        raise BrokenSourceError(f"{job['source_image']}: yielded {written} of {n_frames} frames")
    return gray_source, offcurve_total / max(n_frames, 1), stats


def _qc_frames(manual_frames: list[int], n_frames: int, how_many: int) -> list[int]:
    """First manual frame, then the middle frame, capped at `how_many` (qc_frames_per_pullback).

    The cap applies to that priority order, so `qc_frames_per_pullback=1` keeps the manual frame.
    """
    candidates = ([manual_frames[0]] if manual_frames else []) + [n_frames // 2]
    return sorted(list(dict.fromkeys(candidates))[:max(how_many, 0)])


def _remove_group(root, pullback_id: str) -> None:
    """Drop a partially written group so a failed pullback leaves nothing behind."""
    if root is None:
        return
    try:
        del root[pullback_id]
    except Exception:                      # cleanup of a failed pullback must never raise
        pass


def _array_bytes(store_path: str, pullback_id: str, name: str) -> int:
    total = 0
    for directory, _subdirs, files in os.walk(os.path.join(store_path, pullback_id, name)):
        total += sum(os.path.getsize(os.path.join(directory, f)) for f in files)
    return total


# ---------------------------------------------------------------- planning helpers
def _source_roots(store: StoreCfg) -> tuple[str, ...]:
    return tuple(store.image_roots) + (store.label_root, os.path.dirname(store.excel))


def _frame_estimate(planned: pd.DataFrame) -> int:
    """Frames the sheet claims, plus the sidecar count of the unlabelled rows (0 without a sidecar)."""
    return int(planned[["excel_n_frames", "sidecar_n_frames"]].max(axis=1).clip(lower=0).sum())


def _free_bytes(path: str) -> int:
    """Free bytes on the filesystem that will hold `path` (walking up to the first existing parent)."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    return shutil.disk_usage(probe).free


def _print_plan(planned: pd.DataFrame, dst: str) -> None:
    status = planned["build_status"]
    buildable = planned[status == "ok"]
    frames, build_frames = _frame_estimate(planned), _frame_estimate(buildable)
    unlabelled = planned[planned["split"] == "unlabelled"]
    print(f"planned {len(planned)} pullbacks ({len(unlabelled)} unlabelled): "
          f"{int((status != 'missing_image').sum())} images found, "
          f"{int((status == 'missing_image').sum())} missing_image, "
          f"{int((status == 'missing_label').sum())} missing_label")
    print(f"{frames:,} frames in the sheet + sidecars -> est {frames * BYTES_PER_FRAME / 1e9:.1f}GB "
          f"at {BYTES_PER_FRAME / 1024:.0f}KB/frame "
          f"({int((unlabelled['sidecar_n_frames'] < 0).sum())} unlabelled without sidecar count as 0)")
    print(f"buildable now: {len(buildable)} pullbacks / {build_frames:,} frames "
          f"(est {build_frames * BYTES_PER_FRAME / 1e9:.1f}GB)")
    print(f"free on {dst}: {_free_bytes(dst) / 1e9:.0f}GB")


def _preflight(dst: str, est_bytes: int, store: StoreCfg) -> None:
    free = _free_bytes(dst)
    if free - est_bytes < store.disk_reserve_bytes:
        sys.exit(f"REFUSING: would leave {(free - est_bytes) / 1e9:.0f}GB free on {dst}, "
                 f"reserve is {store.disk_reserve_bytes / 1e9:.0f}GB (store.disk_reserve_bytes)")


def _write_lut(store: StoreCfg, dst: str) -> str:
    """Vendor table unless `store.lut_path` names one; always saved as `<dst>/lut.npz`."""
    lut = load_lut(store.lut_path) if store.lut_path else build_vendor_lut()
    lut_path = os.path.join(dst, "lut.npz")
    save_lut(lut_path, lut)
    print(f"LUT [{lut['source']}]: {lut['n']} triplets -> {lut_path}")
    return lut_path


def _job(row, cfg: Config, store_path: str, parts_dir: str, qc_dir: str, lut_path: str) -> dict:
    """One picklable unit of work (no zarr handles, no arrays)."""
    return {"pullback_id": str(row.pullback_id), "patient_id": str(row.patient_id),
            "split": str(row.split), "source_image": str(row.source_image),
            "source_label": str(row.source_label), "has_label": bool(row.source_label),
            "manual_frames": [int(f) for f in row.manual_frames],
            "excel_n_frames": int(row.excel_n_frames), "cfg": cfg.store,
            "z_spacing_table": cfg.store.z_spacing_table,
            "disc_center": tuple(float(c) for c in cfg.store.center),
            "disc_radius": float(cfg.store.disc_radius), "store_path": store_path,
            "parts_dir": parts_dir, "qc_dir": qc_dir, "lut_path": lut_path}


def _already_built(record: dict | None, parts_dir: str, pullback_id: str) -> bool:
    return (record is not None and record["status"] == "ok"
            and os.path.exists(os.path.join(parts_dir, f"{pullback_id}.parquet")))


def _read_manifest(dst: str) -> dict[str, dict]:
    """{pullback_id: last record} from manifest.jsonl."""
    records: dict[str, dict] = {}
    path = os.path.join(dst, "manifest.jsonl")
    if not os.path.exists(path):
        return records
    with open(path) as manifest:
        for number, line in enumerate(manifest, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING {path}:{number} is not valid JSON; ignoring that record")
                continue
            records[record["pullback_id"]] = record
    return records


def _write_json(path: str, payload) -> None:
    with open(path, "w") as out:
        json.dump(payload, out, indent=1)


def _by_id(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: r["pullback_id"])


# ---------------------------------------------------------------- finalize helpers
def _empty_frames() -> pd.DataFrame:
    """The frames schema with no rows (a store where nothing was built)."""
    return stats_frame("", "", "", 0, [], False, None, np.zeros((0, N_CLASSES), dtype=np.int32),
                       np.full((0, N_CLASSES, 4), -1, dtype=np.int16))


def _check_frames(frames: pd.DataFrame) -> None:
    """Every frame of every ok pullback, sorted by (pullback_id, frame) and contiguous from 0.

    `OCTStore` maps a pullback to one contiguous row span, so this is an invariant, not a nicety.
    """
    ids = frames["pullback_id"].tolist()
    if ids != sorted(ids):
        raise ValueError("frames.parquet rows are not sorted by pullback_id")
    for pullback_id, group in frames.groupby("pullback_id", sort=False):
        expected = np.arange(len(group), dtype=np.int32)
        if not np.array_equal(group["frame"].to_numpy(), expected):
            raise ValueError(f"{pullback_id}: frame rows are not contiguous 0..{len(group) - 1}")
        if int(group["n_frames"].iloc[0]) != len(group):
            raise ValueError(f"{pullback_id}: {len(group)} rows for {int(group['n_frames'].iloc[0])} frames")


def _frames_table(frames: pd.DataFrame) -> pa.Table:
    """Table types; the three id columns are dictionary-encoded like the rest of the store."""
    types = {"frame": pa.int32(), "n_frames": pa.int32(), "is_manual": pa.bool_(),
             "has_label": pa.bool_(), "z_mm": pa.float32()}
    types.update({column: pa.int32() for column in px_columns()})
    types.update({column: pa.int16() for column in bbox_columns()})
    columns = {name: pa.array(frames[name], type=pa.string()).dictionary_encode()
               for name in DICTIONARY_COLUMNS}
    columns.update({name: pa.array(frames[name], type=arrow_type) for name, arrow_type in types.items()})
    return pa.table({name: columns[name] for name in list(DICTIONARY_COLUMNS) + list(types)})


def _merge_pullbacks(planned: pd.DataFrame, records: dict[str, dict]) -> pd.DataFrame:
    """One row per planned pullback, build columns from its last manifest record."""
    out = planned.copy()
    out["has_label"] = out["source_label"].str.len() > 0
    built = {"n_frames": [], "z_spacing_mm": [], "pixel_spacing_mm": [], "gray_source": [],
             "offcurve_frac": [], "build_status": [], "build_reason": [], "orientation_ok": [],
             "source_format": []}
    for row in planned.itertuples():
        record = records.get(row.pullback_id)
        attrs = record["attrs"] if record else {}
        built["n_frames"].append(int(record["n_frames"]) if record else -1)
        built["z_spacing_mm"].append(_float(attrs.get("z_spacing_mm")))
        built["pixel_spacing_mm"].append(_float(attrs.get("pixel_spacing_mm")))
        built["gray_source"].append(attrs.get("gray_source", ""))
        built["offcurve_frac"].append(_float(attrs.get("offcurve_frac")))
        built["source_format"].append(attrs.get("source_format") or row.source_format)
        built["build_status"].append(record["status"] if record else row.build_status)
        built["build_reason"].append(record["reason"] if record else
                                     ("not built" if row.build_status == "ok" else ""))
        built["orientation_ok"].append(bool(record["orientation_ok"]) if record else False)
    for column, values in built.items():
        out[column] = values
    return out


def _pullbacks_table(pullbacks: pd.DataFrame) -> pa.Table:
    """Column order and types."""
    types = {"pullback_id": pa.string(), "patient_id": pa.string(), "split": pa.string(),
             "excel_row": pa.int32(), "excel_id": pa.int32(), "n_frames": pa.int32(),
             "excel_n_frames": pa.int32(), "n_manual": pa.int32(),
             "manual_frames": pa.list_(pa.int32()), "has_label": pa.bool_(),
             "z_spacing_mm": pa.float32(), "pixel_spacing_mm": pa.float32(),
             "source_image": pa.string(), "source_label": pa.string(), "source_format": pa.string(),
             "gray_source": pa.string(), "offcurve_frac": pa.float32(), "build_status": pa.string(),
             "build_reason": pa.string(), "orientation_ok": pa.bool_()}
    types.update({f"xl_frames_{token}": pa.int16() for token in CLASS_TOKENS})
    types.update({"dem_age": pa.float32(), "dem_gender": pa.int8(), "mh_mi": pa.int8(),
                  "mh_dm": pa.int8(), "mh_ht": pa.int8(), "mh_hc": pa.int8(),
                  "lipid_lowering": pa.int8(), "dataset": pa.int8()})
    return pa.table({name: pa.array(pullbacks[name], type=arrow_type)
                     for name, arrow_type in types.items()})


def _orientation_report(records: dict[str, dict], min_contrast: float) -> dict:
    """qc/orientation.json: per-pullback verdicts for every pullback whose label was checked."""
    verdicts = {}
    for pullback_id, record in sorted(records.items()):
        if record["status"] != "ok" or not record["qc"]:
            continue
        verdicts[pullback_id] = {"ok": bool(record["orientation_ok"]),
                                 "frames": [{key: _finite(value) for key, value in entry.items()}
                                            for entry in record["qc"]]}
    failed = [pullback_id for pullback_id, verdict in verdicts.items() if not verdict["ok"]]
    return {"min_contrast": float(min_contrast), "checked": len(verdicts), "failed": failed,
            "pullbacks": verdicts}


def _excel_counts_match(frames: pd.DataFrame, planned: pd.DataFrame) -> dict:
    """Manual-frame class presence vs the sheet's 17 `xl_frames_<token>` counts, per labelled pullback."""
    expected = planned[planned["excel_row"] >= 0].set_index("pullback_id")
    manual = frames[frames["is_manual"] & frames["has_label"]]
    result = {"checked": 0, "match": 0, "all_match": True, "mismatch": []}
    for pullback_id, group in manual.groupby("pullback_id", sort=True):
        if pullback_id not in expected.index:
            continue
        row = expected.loc[pullback_id]
        found = {token: int((group[f"px_{token}"] > 0).sum()) for token in CLASS_TOKENS}
        sheet = {token: int(row[f"xl_frames_{token}"]) for token in CLASS_TOKENS}
        differing = [token for token in CLASS_TOKENS if found[token] != sheet[token]]
        result["checked"] += 1
        if differing:
            result["mismatch"].append({"pullback_id": pullback_id, "tokens": differing,
                                       "excel": [sheet[t] for t in differing],
                                       "found": [found[t] for t in differing]})
        else:
            result["match"] += 1
    result["all_match"] = not result["mismatch"]
    return result


def _dataset_info(cfg: Config, planned: pd.DataFrame, records: dict[str, dict], frames: pd.DataFrame,
                  orientation: dict, counts_match: dict) -> dict:
    """dataset_info.json (+ `pullbacks.excluded` and the `excel_counts_match` report)."""
    store = cfg.store
    ok = {i: r for i, r in records.items() if r["status"] == "ok"}
    labelled = frames[frames["has_label"]]
    manual = labelled[labelled["is_manual"]]
    per_pullback = frames.drop_duplicates("pullback_id")
    statuses = pd.Series([r["status"] for r in records.values()], dtype="str")
    normalise = [i for i, r in ok.items() if r["attrs"]["split"] in ("train", "unlabelled")]
    total = sum(ok[i]["image_sum"] for i in normalise)
    total_sq = sum(ok[i]["image_sumsq"] for i in normalise)
    n_pixels = sum(ok[i]["n_pixels"] for i in normalise)
    mean = total / max(n_pixels, 1)
    variance = total_sq / max(n_pixels, 1) - mean * mean
    counts = frames[px_columns()].to_numpy(dtype=np.int64)
    manual_counts = manual[px_columns()].to_numpy(dtype=np.int64)
    return {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_cfg": dataclasses.asdict(store),
        "z_spacing_table": store.z_spacing_table,
        "lut": _lut_info(store.dst),
        "layout": {"frame_shape": list(FRAME_SHAPE), "chunk_frames": store.chunk_frames,
                   "shard_frames": store.shard_frames,
                   "image_codec": f"blosc-zstd-{store.zstd_level}-noshuffle",
                   "label_codec": f"blosc-zstd-{store.zstd_level}-bitshuffle"},
        "class_names": list(CLASS_NAMES),
        "class_tokens": list(CLASS_TOKENS),
        # planned / missing_image / labelled / unlabelled count planned rows; ok / error / excluded
        # count build attempts (a planned pullback that was never attempted is in none of the three)
        "pullbacks": {"planned": len(planned), "ok": len(ok),
                      "error": int((statuses == "error").sum()),
                      "excluded": int((statuses == "excluded").sum()),
                      "missing_image": int((planned["build_status"] == "missing_image").sum()),
                      "labelled": int((planned["split"] != "unlabelled").sum()),
                      "unlabelled": int((planned["split"] == "unlabelled").sum())},
        "frames": {"total": len(frames), "labelled": len(labelled), "manual": len(manual)},
        "splits": _split_summary(per_pullback, frames),
        "image_mean": float(mean),
        "image_std": float(np.sqrt(max(variance, 0.0))),
        "class_pixels_all": dict(zip(CLASS_TOKENS, counts.sum(0).tolist())),
        "class_pixels_manual": dict(zip(CLASS_TOKENS, manual_counts.sum(0).tolist())),
        "class_frames_all": dict(zip(CLASS_TOKENS, (counts > 0).sum(0).tolist())),
        "class_frames_manual": dict(zip(CLASS_TOKENS, (manual_counts > 0).sum(0).tolist())),
        "qc": {"orientation_checked": orientation["checked"],
               "orientation_failed": orientation["failed"],
               "overlay_pngs": sum(len(r["qc"]) for r in ok.values())},
        "excel_counts_match": counts_match,
        "errors": [{"pullback_id": r["pullback_id"], "status": r["status"], "reason": r["reason"]}
                   for r in _by_id([r for r in records.values() if r["status"] != "ok"])],
        "bytes": {"image": sum(r["bytes_image"] for r in ok.values()),
                  "label": sum(r["bytes_label"] for r in ok.values())},
    }


def _split_summary(per_pullback: pd.DataFrame, frames: pd.DataFrame) -> dict:
    out = {}
    for split in ("train", "val", "test", "unlabelled"):
        pullbacks = per_pullback[per_pullback["split"] == split]
        rows = frames[frames["split"] == split]
        out[split] = {"pullbacks": len(pullbacks),
                      "patients": int(pullbacks["patient_id"].nunique()),
                      "frames": len(rows), "manual": int(rows["is_manual"].sum())}
    return out


def _lut_info(dst: str) -> dict:
    """{source, n} of the colormap the build used, read straight from `<dst>/lut.npz`."""
    path = os.path.join(dst, "lut.npz")
    if not os.path.exists(path):           # --finalize-only on a store built elsewhere
        return {"source": "unknown", "n": 0}
    saved = np.load(path)
    return {"source": str(saved["source"]), "n": int(len(saved["curve"]))}


def _float(value) -> float:
    return float(value) if value is not None else float("nan")


def _finite(value):
    """NaN -> None so the QC json stays strict JSON."""
    return None if isinstance(value, float) and not np.isfinite(value) else value
