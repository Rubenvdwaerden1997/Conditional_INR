"""Invert the vendor colormap (sepia LUT) to recover single-channel uint8.

The scanner exports a physically single-channel signal through a ~256-entry
RGB colormap. Two ways to get the curve back:

  vendor     the scanner's own table, transcribed in vendor_lut.py -- exact
             gray indices, no dependence on what this dataset happens to show
  empirical  order the RGB triplets observed in the data by luminance and
             spread gray over [0, 255] -- works for any colormap

Either way every pixel is mapped through a 24-bit lookup table. Pixels whose
triplet is not on the curve (rare: overlays, interpolation) fall back to
Rec.709 luminance projected onto the same curve.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _pack(triplets: np.ndarray) -> np.ndarray:
    t = triplets.astype(np.uint32)
    return (t[:, 0] << 16) | (t[:, 1] << 8) | t[:, 2]


def _assemble(curve: np.ndarray, gray: np.ndarray, source: str) -> dict:
    """Expand a (curve, gray) pair into the dense lookup structures.

    Returns dict with:
      table:  uint8 array of 2^24 gray values (dense 24-bit lookup)
      valid:  bool array of 2^24 (triplet present on the curve)
      n:      number of curve entries
      monotonic_frac: fraction of curve steps where each channel is non-decreasing
      curve_lum/curve_gray: luminance-sorted curve, for the off-curve fallback
    """
    curve = curve.astype(np.uint8)
    gray = gray.astype(np.uint8)
    table = np.zeros(1 << 24, dtype=np.uint8)
    valid = np.zeros(1 << 24, dtype=bool)
    packed = _pack(curve)
    table[packed] = gray
    valid[packed] = True
    lum = curve.astype(np.float64) @ LUMA
    order = np.argsort(lum, kind="stable")
    diffs = np.diff(curve[order].astype(np.int32), axis=0)
    monotonic_frac = float(np.mean((diffs >= 0).all(axis=1))) if len(curve) > 1 else 1.0
    return {"table": table, "valid": valid, "n": len(curve),
            "monotonic_frac": monotonic_frac, "source": source,
            "curve": curve, "gray": gray, "curve_lum": lum[order],
            "curve_gray": gray[order].astype(np.float32)}


def build_lut(triplets: np.ndarray) -> dict:
    """Order observed triplets by luminance -> gray levels spread over [0, 255]."""
    lum = triplets.astype(np.float64) @ LUMA
    order = np.argsort(lum, kind="stable")
    curve = triplets[order]
    # gray proportional to luminance position along the curve: faithful even
    # when only part of the colormap was observed (rank-based would warp it)
    ls = lum[order]
    span = max(ls[-1] - ls[0], 1e-9)
    gray = np.round((ls - ls[0]) / span * 255.0).astype(np.uint8) if len(curve) > 1 \
        else np.zeros(len(curve), np.uint8)
    return _assemble(curve, gray, "empirical")


def build_vendor_lut() -> dict:
    """The scanner's own colormap table (vendor_lut.py), no data needed."""
    from .vendor_lut import VENDOR_CURVE, VENDOR_GRAY
    return _assemble(VENDOR_CURVE, VENDOR_GRAY, "vendor")


def rgb_to_gray(frames: np.ndarray, lut: dict) -> tuple[np.ndarray, float]:
    """Map [..., 3] uint8 RGB through the LUT. Returns (gray, offcurve_fraction).

    Pixels whose triplet was never observed on the curve are projected onto it
    by luminance (interpolated along the curve), so both paths share one scale.
    """
    shape = frames.shape[:-1]
    flat = frames.reshape(-1, 3)
    packed = _pack(flat)
    out = lut["table"][packed]
    ok = lut["valid"][packed]
    off = 1.0 - float(ok.mean())
    if off > 0:
        miss = ~ok
        lum = flat[miss].astype(np.float32) @ LUMA.astype(np.float32)
        proj = np.interp(lum, lut["curve_lum"], lut["curve_gray"])
        out = out.copy() if not out.flags.writeable else out
        out[miss] = np.clip(np.round(proj), 0, 255).astype(np.uint8)
    return out.reshape(shape), off


def save_lut(path: str, lut: dict) -> None:
    np.savez_compressed(path, curve=lut["curve"], gray=lut["gray"],
                        source=lut.get("source", "empirical"))


def load_lut(path: str) -> dict:
    z = np.load(path)
    source = str(z["source"]) if "source" in z else "empirical"
    return _assemble(z["curve"], z["gray"], source)
