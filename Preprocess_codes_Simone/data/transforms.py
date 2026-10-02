"""Clip augmentations in polar space. Clips are numpy uint8 [T, R, THETA]."""
from __future__ import annotations

import numpy as np


def theta_shift_crop(clip: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
    """Random circular shift along theta, then crop a window of `width`.

    This is simultaneously the theta wrap-around handling and the rotation
    augmentation: after the roll, the crop edges are arbitrary angles.
    """
    nt = clip.shape[-1]
    shift = int(rng.integers(0, nt))
    rolled = np.roll(clip, shift, axis=-1)
    return rolled[..., :width]


def r_jitter(clip: np.ndarray, max_px: int, rng: np.random.Generator) -> np.ndarray:
    """Small radial shift with edge clamp (radial position stays meaningful)."""
    if max_px <= 0:
        return clip
    s = int(rng.integers(-max_px, max_px + 1))
    if s == 0:
        return clip
    out = np.empty_like(clip)
    if s > 0:
        out[:, s:] = clip[:, :-s]
        out[:, :s] = clip[:, :1]
    else:
        out[:, :s] = clip[:, -s:]
        out[:, s:] = clip[:, -1:]
    return out


def intensity_jitter(clip: np.ndarray, brightness: float, contrast: float,
                     rng: np.random.Generator) -> np.ndarray:
    if brightness <= 0 and contrast <= 0:
        return clip
    b = rng.uniform(-brightness, brightness) * 255.0
    c = 1.0 + rng.uniform(-contrast, contrast)
    out = clip.astype(np.float32) * c + b
    return np.clip(out, 0, 255).astype(np.uint8)


class ClipAugment:
    """`FrameWindowDataset` transform for V-JEPA clips: theta roll + crop, radial jitter, flip,
    time reverse, intensity jitter on the uint8 `image` [T, R, THETA]; eval mode only crops theta.

    Draws from numpy's global RNG, which `seed_worker` seeds per DataLoader worker and epoch, so
    forked workers do not replay one stream. Normalisation is the dataset's job.
    """

    def __init__(self, theta_crop: int, r_jitter_px: int, flip_p: float,
                 time_reverse_p: float, brightness: float, contrast: float, train: bool = True):
        self.theta_crop = theta_crop
        self.r_jitter_px = r_jitter_px
        self.flip_p = flip_p
        self.time_reverse_p = time_reverse_p
        self.brightness = brightness
        self.contrast = contrast
        self.train = train

    def __call__(self, sample: dict) -> dict:
        clip = sample["image"]
        if self.train:
            rng = np.random.default_rng(np.random.randint(0, 2**31))
            clip = theta_shift_crop(clip, self.theta_crop, rng)
            clip = r_jitter(clip, self.r_jitter_px, rng)
            if rng.random() < self.flip_p:
                clip = clip[..., ::-1]
            if rng.random() < self.time_reverse_p:
                clip = clip[::-1]
            clip = intensity_jitter(clip, self.brightness, self.contrast, rng)
        else:
            clip = clip[..., :self.theta_crop]
        sample["image"] = np.ascontiguousarray(clip)
        return sample
