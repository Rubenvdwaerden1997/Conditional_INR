"""Segmentation augmentations in polar space: image and label move together.

`ClipAugment` (pretraining) touches only `sample["image"]`; a segmentation transform must apply
every geometric op to `sample["label"]` with the *same* draw or the supervision silently
desynchronises from the pixels. Arrays are the frame-major numpy uint8 [T, R, THETA] the
`FrameWindowDataset` transform hook hands over, after the geometry and before normalisation.
"""
from __future__ import annotations

import numpy as np

from .transforms import intensity_jitter


def shift_radial(a: np.ndarray, s: int) -> np.ndarray:
    """Shift by `s` px along r (axis 1) with edge clamp; radial position stays meaningful."""
    if s == 0:
        return a
    out = np.empty_like(a)
    if s > 0:
        out[:, s:], out[:, :s] = a[:, :-s], a[:, :1]
    else:
        out[:, :s], out[:, s:] = a[:, -s:], a[:, -1:]
    return out


def scale_radial(a: np.ndarray, factor: float, linear: bool) -> np.ndarray:
    """Resample along r (axis 1) by `factor`, keeping the row count, with edge clamp.

    `factor > 1` stretches the annulus outward -- the outer rows run past the last source row and
    repeat it -- and `factor < 1` pulls it in. `linear` interpolates between the two neighbouring
    source rows (the image); otherwise the nearest source row is taken whole, because averaging
    class ids would invent labels that are not in the map.
    """
    n = a.shape[1]
    src = np.clip(np.arange(n, dtype=np.float32) / float(factor), 0, n - 1)
    if not linear:
        return a[:, np.rint(src).astype(np.int64)]
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo).astype(np.float32)[None, :, None]
    out = a[:, lo].astype(np.float32) * (1.0 - w) + a[:, hi].astype(np.float32) * w
    return np.rint(out).astype(a.dtype)


def gamma_jitter(clip: np.ndarray, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Random gamma on the 0..1 scale, exponent drawn in [1 - gamma, 1 + gamma].

    Unlike brightness/contrast this is non-linear, so it changes the *relative* intensity of the
    dim intima and the bright media the way a different gain setting or a different catheter does.
    """
    if gamma <= 0:
        return clip
    g = float(rng.uniform(1.0 - gamma, 1.0 + gamma))
    return np.rint(np.power(clip.astype(np.float32) / 255.0, g) * 255.0).astype(np.uint8)


def speckle_noise(clip: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative noise `x * (1 + sigma * N(0, 1))`: OCT speckle scales with the signal."""
    if sigma <= 0:
        return clip
    noise = rng.normal(1.0, sigma, size=clip.shape).astype(np.float32)
    return np.clip(clip.astype(np.float32) * noise, 0, 255).astype(np.uint8)


class SegAugment:
    """Theta roll and flip, radial scale and shift (all on image + label), then image-only intensity.

    The theta roll is the important one: theta wraps, so a circular roll is an exact symmetry of the
    data -- and, with the angular positional embedding of `models.vit.sincos_angular`, of the model
    too. It is a free full-rotation augmentation, which matters a great deal on 616 training frames.

    The radial scale is the one geometric degree of freedom the shift does not cover: the catheter
    sits at a different distance from the wall in every pullback, so the whole annulus is stretched
    or compressed rather than only translated. Image and label share one draw, bilinear against
    nearest.

    Gamma and speckle are image-only and run before the brightness/contrast jitter, so the linear
    jitter is the last thing applied and its clipping is the only clipping the pixel sees twice.

    There is deliberately **no time reversal**: the label is the anchor frame at index `T // 2`, and
    reversing an even-length window moves the anchor to `T - 1 - T // 2`, which would silently
    supervise the wrong frame.

    Draws from numpy's global RNG, which `seed_worker` reseeds per DataLoader worker and epoch.
    Eval mode is the identity -- the full theta range is what the model sees at train time too.
    """

    def __init__(self, r_jitter_px: int = 8, r_scale: float = 0.1, flip_p: float = 0.5,
                 gamma: float = 0.3, speckle: float = 0.1, brightness: float = 0.15,
                 contrast: float = 0.15, train: bool = True):
        self.r_jitter_px = int(r_jitter_px)
        self.r_scale = float(r_scale)
        self.flip_p = float(flip_p)
        self.gamma = float(gamma)
        self.speckle = float(speckle)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.train = bool(train)

    @classmethod
    def from_cfg(cls, seg) -> "SegAugment":
        """Build from a `SegCfg`, so the trainer and the loader benchmark cannot drift apart."""
        return cls(r_jitter_px=seg.aug_r_jitter, r_scale=seg.aug_r_scale, flip_p=seg.aug_flip_p,
                   gamma=seg.aug_gamma, speckle=seg.aug_speckle, brightness=seg.aug_brightness,
                   contrast=seg.aug_contrast, train=seg.augment)

    def __call__(self, sample: dict) -> dict:
        if not self.train:
            return sample
        rng = np.random.default_rng(np.random.randint(0, 2**31))
        keys = [k for k in ("image", "label") if k in sample]
        arrs = {k: sample[k] for k in keys}

        shift = int(rng.integers(0, arrs["image"].shape[-1]))
        arrs = {k: np.roll(v, shift, axis=-1) for k, v in arrs.items()}
        if self.r_scale > 0:
            factor = float(rng.uniform(1.0 - self.r_scale, 1.0 + self.r_scale))
            arrs = {k: scale_radial(v, factor, linear=k == "image") for k, v in arrs.items()}
        if self.r_jitter_px > 0:
            s = int(rng.integers(-self.r_jitter_px, self.r_jitter_px + 1))
            arrs = {k: shift_radial(v, s) for k, v in arrs.items()}
        if rng.random() < self.flip_p:
            arrs = {k: v[..., ::-1] for k, v in arrs.items()}

        image = gamma_jitter(arrs["image"], self.gamma, rng)
        image = speckle_noise(image, self.speckle, rng)
        arrs["image"] = intensity_jitter(image, self.brightness, self.contrast, rng)
        for k, v in arrs.items():
            sample[k] = np.ascontiguousarray(v)
        return sample
