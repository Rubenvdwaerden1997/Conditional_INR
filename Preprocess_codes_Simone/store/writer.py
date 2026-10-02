"""zarr v3 store writer: one group per pullback, one-frame chunks, 64-frame shards."""
from __future__ import annotations

import warnings

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from zarr.errors import ZarrUserWarning

from ..config import StoreCfg

FRAME_SHAPE = (704, 704)          # canonical Cartesian geometry
DIMENSION_NAMES = ("frame", "row", "col")


def image_codec(level: int) -> BloscCodec:
    """blosc-zstd, noshuffle."""
    return BloscCodec(cname="zstd", clevel=level, shuffle="noshuffle")


def label_codec(level: int) -> BloscCodec:
    """blosc-zstd, bitshuffle."""
    return BloscCodec(cname="zstd", clevel=level, shuffle="bitshuffle")


def open_root(store_path: str, mode: str) -> zarr.Group:
    """"a": zarr_format=3, use_consolidated=False; "r": use_consolidated=True with fallback to False."""
    if mode == "a":
        return zarr.open_group(store_path, mode="a", zarr_format=3, use_consolidated=False)
    if mode == "r":
        try:
            return zarr.open_group(store_path, mode="r", use_consolidated=True)
        except ValueError:                       # no consolidated metadata (build not finalized yet)
            return zarr.open_group(store_path, mode="r", use_consolidated=False)
    raise ValueError(f"open_root mode must be 'a' or 'r', got {mode!r}")


def create_pullback(root: zarr.Group, pullback_id: str, n_frames: int, has_label: bool, attrs: dict,
                    cfg: StoreCfg) -> zarr.Group:
    """Create `<pullback_id>/image` (+ `/label`) uint8 [n_frames, 704, 704] with overwrite=True and the group attrs."""
    g = root.create_group(pullback_id, overwrite=True)
    g.update_attributes(dict(attrs))
    shape = (n_frames, *FRAME_SHAPE)
    chunks = (cfg.chunk_frames, *FRAME_SHAPE)
    shards = (cfg.shard_frames, *FRAME_SHAPE)
    g.create_array(name="image", shape=shape, dtype="uint8", chunks=chunks, shards=shards,
                   compressors=image_codec(cfg.zstd_level), fill_value=0, dimension_names=DIMENSION_NAMES)
    if has_label:
        g.create_array(name="label", shape=shape, dtype="uint8", chunks=chunks, shards=shards,
                       compressors=label_codec(cfg.zstd_level), fill_value=0, dimension_names=DIMENSION_NAMES)
    return g


def write_block(arr: zarr.Array, start: int, block: np.ndarray) -> None:
    """Shard-aligned write: `start % shard_frames == 0` (no read-modify-write)."""
    shard_frames = arr.shards[0] if arr.shards is not None else arr.chunks[0]
    if start % shard_frames:
        raise ValueError(f"write start {start} is not a multiple of shard_frames {shard_frames}")
    arr[start:start + len(block)] = block


def verify_frames(arr: zarr.Array, kept: dict[int, np.ndarray]) -> None:
    """Re-read the kept frames and assert bit-exact equality."""
    for frame, expected in kept.items():
        got = arr[frame]
        if not np.array_equal(got, expected):
            raise ValueError(f"frame {frame} of array {arr.path!r} differs from what was written")


def consolidate(store_path: str) -> None:
    """zarr.consolidate_metadata(store_path)."""
    with warnings.catch_warnings():
        # zarr warns that consolidated metadata is not part of the v3 spec; we opt into it
        # deliberately and every reader goes through open_root(), which falls back when it is absent.
        warnings.filterwarnings("ignore", category=ZarrUserWarning, message=".*[Cc]onsolidated metadata.*")
        zarr.consolidate_metadata(store_path)
