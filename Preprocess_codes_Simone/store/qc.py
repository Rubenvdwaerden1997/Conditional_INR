"""Build-time QC: label orientation check and overlay PNGs.

Orientation: the verified convention is `[frame, row, col]` with the label equal to
`nib_array.transpose(2, 1, 0)`. In that orientation the intima is far brighter than the lumen
(97 vs 19 gray on the reference pullback); the row/column transpose of the same label scores
near zero. `orientation_contrast` therefore reports the intima-minus-lumen mean gray for the
label as stored and for its transpose, and the build refuses to trust a pullback whose stored
contrast is neither large nor larger than the transposed one.
"""
from __future__ import annotations

import numpy as np

LUMEN_CLASS = 1
TISSUE_CLASS = 3          # intima: the bright layer that sits directly outside the lumen


def orientation_contrast(gray: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """(tissue - lumen) mean-gray contrast for the label as stored and for label.T."""
    return _contrast(gray, label), _contrast(gray, np.swapaxes(label, -1, -2))


def orientation_ok(gray: np.ndarray, label: np.ndarray, min_contrast: float) -> tuple[bool, dict]:
    """True when stored contrast >= min_contrast and > the transposed contrast; dict holds both numbers."""
    stored, transposed = orientation_contrast(gray, label)
    ok = bool(stored >= min_contrast and stored > transposed)          # NaN (class absent) -> False
    return ok, {"stored": stored, "transposed": transposed, "min_contrast": float(min_contrast),
                "n_tissue": int(np.count_nonzero(label == TISSUE_CLASS)),
                "n_lumen": int(np.count_nonzero(label == LUMEN_CLASS))}


def save_overlay_png(path: str, gray: np.ndarray, label: np.ndarray, title: str) -> None:
    """matplotlib Agg overlay of label on gray."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5.0, 5.0), dpi=110)
    axes.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes.imshow(np.ma.masked_where(label == 0, label), cmap="tab20", vmin=0, vmax=16,
                alpha=0.45, interpolation="nearest")
    axes.set_title(title, fontsize=8)
    axes.set_axis_off()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _contrast(gray: np.ndarray, label: np.ndarray) -> float:
    """Mean gray over the tissue class minus mean gray over the lumen; NaN when either is absent.

    A non-square frame has no comparable transpose, so its transposed contrast is NaN.
    """
    if label.shape != gray.shape:
        return float("nan")
    tissue, lumen = gray[label == TISSUE_CLASS], gray[label == LUMEN_CLASS]
    if tissue.size == 0 or lumen.size == 0:
        return float("nan")
    return float(tissue.mean() - lumen.mean())
