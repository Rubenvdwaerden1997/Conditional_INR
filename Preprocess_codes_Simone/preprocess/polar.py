"""Cartesian <-> polar resampling for OCT frames.

Polar grid: rows = radius r in [r0, r1) at native 1 px step, columns = angle
theta over [0, 2pi). Bilinear interpolation with precomputed gather indices,
vectorized over whole frame stacks.
"""
from __future__ import annotations

import cv2
import numpy as np


def disc_mask(size: int, center: tuple[float, float], radius: float) -> np.ndarray:
    """Boolean [size, size] mask, True inside the imaging disc.

    The vendor overlays (logo/watermark, scale text, legend) sit in the corners
    outside this disc, so masking with it keeps them out of anything measured
    on whole frames -- the polar resampling already drops them.
    """
    cx, cy = center
    yy, xx = np.ogrid[:size, :size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


class PolarMapper:
    def __init__(self, size: int = 704, r0: int = 32, r1: int = 352,
                 n_theta: int = 704, center: tuple[float, float] = (352.0, 352.0)):
        self.size, self.r0, self.r1, self.n_theta = size, r0, r1, n_theta
        self.center = center
        r = np.arange(r0, r1, dtype=np.float64)
        th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
        cx, cy = center
        x = cx + r[:, None] * np.cos(th)[None, :]
        y = cy + r[:, None] * np.sin(th)[None, :]
        # cv2.remap wants float32 pixel-coordinate maps; ~30x faster than the numpy gather below for the
        # same bilinear result (checked to within 1 grey level), so this is the only path `to_polar` uses.
        self._fw_map_x = x.astype(np.float32)
        self._fw_map_y = y.astype(np.float32)
        self._fw_shape = x.shape
        # inverse map (polar -> cartesian) for QC round-trips, annulus only
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        dx, dy = xx - cx, yy - cy
        rad = np.hypot(dx, dy)
        ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        self.annulus = (rad >= r0) & (rad < r1 - 1)
        rr = np.clip(rad - r0, 0, (r1 - r0) - 1.001)
        tt = ang / (2 * np.pi) * n_theta
        self._bw = self._bilinear_gather_polar(rr, tt, r1 - r0, n_theta)

    @staticmethod
    def _bilinear_gather(x, y, size):
        x0 = np.clip(np.floor(x).astype(np.int32), 0, size - 2)
        y0 = np.clip(np.floor(y).astype(np.int32), 0, size - 2)
        fx = (x - x0).astype(np.float32)
        fy = (y - y0).astype(np.float32)
        w00 = (1 - fx) * (1 - fy)
        w01 = fx * (1 - fy)
        w10 = (1 - fx) * fy
        w11 = fx * fy
        flat = lambda yy, xx: (yy * size + xx).ravel()
        return dict(i00=flat(y0, x0), i01=flat(y0, x0 + 1), i10=flat(y0 + 1, x0),
                    i11=flat(y0 + 1, x0 + 1),
                    w00=w00.ravel(), w01=w01.ravel(), w10=w10.ravel(), w11=w11.ravel(),
                    shape=x.shape)

    @staticmethod
    def _bilinear_gather_polar(rr, tt, nr, nt):
        r0i = np.floor(rr).astype(np.int32)
        t0i = np.floor(tt).astype(np.int32) % nt
        fr = (rr - np.floor(rr)).astype(np.float32)
        ft = (tt - np.floor(tt)).astype(np.float32)
        r1i = np.clip(r0i + 1, 0, nr - 1)
        t1i = (t0i + 1) % nt          # theta wraps
        flat = lambda ri, ti: (ri * nt + ti).ravel()
        return dict(i00=flat(r0i, t0i), i01=flat(r0i, t1i), i10=flat(r1i, t0i),
                    i11=flat(r1i, t1i),
                    w00=((1 - fr) * (1 - ft)).ravel(), w01=((1 - fr) * ft).ravel(),
                    w10=(fr * (1 - ft)).ravel(), w11=(fr * ft).ravel(),
                    shape=rr.shape)

    @staticmethod
    def _apply(g, imgs_flat):
        out = (imgs_flat[:, g["i00"]] * g["w00"] + imgs_flat[:, g["i01"]] * g["w01"] +
               imgs_flat[:, g["i10"]] * g["w10"] + imgs_flat[:, g["i11"]] * g["w11"])
        return out

    def to_polar(self, frames: np.ndarray) -> np.ndarray:
        """[T, H, W] uint8 -> [T, R, THETA] uint8, bilinear (cv2.remap with precomputed maps)."""
        single = frames.ndim == 2
        if single:
            frames = frames[None]
        out = np.stack([cv2.remap(f, self._fw_map_x, self._fw_map_y, cv2.INTER_LINEAR)
                         for f in frames])
        return out[0] if single else out

    def to_cartesian(self, polar: np.ndarray) -> np.ndarray:
        """[T, R, THETA] uint8 -> [T, H, W] uint8 (zeros outside annulus)."""
        single = polar.ndim == 2
        if single:
            polar = polar[None]
        t = polar.shape[0]
        out = self._apply(self._bw, polar.reshape(t, -1).astype(np.float32))
        out = np.clip(np.round(out), 0, 255).astype(np.uint8).reshape(t, self.size, self.size)
        out *= self.annulus[None]
        return out[0] if single else out

    def to_polar_nearest(self, frames: np.ndarray) -> np.ndarray:
        """Integer-dtype polar resampling with a single nearest tap (label-safe: no new values).

        [H, W] -> [R, THETA] and [T, H, W] -> [T, R, THETA]; the dtype and the exact value set of the input
        are preserved, so a class-id label never gains an interpolated value the way `to_polar` would.
        """
        single = frames.ndim == 2
        if single:
            frames = frames[None]
        t = frames.shape[0]
        gather = self._nearest_gather()
        out = frames.reshape(t, -1)[:, gather["idx"]].reshape((t,) + gather["shape"])
        return out[0] if single else out

    def _nearest_gather(self) -> dict:
        """Nearest-tap gather indices, built (and cached) on first use so the bilinear path pays nothing."""
        cache = getattr(self, "_nearest", None)
        if cache is None:
            r = np.arange(self.r0, self.r1, dtype=np.float64)
            th = np.linspace(0.0, 2 * np.pi, self.n_theta, endpoint=False)
            cx, cy = self.center
            x = np.clip(np.rint(cx + r[:, None] * np.cos(th)[None, :]), 0, self.size - 1).astype(np.int64)
            y = np.clip(np.rint(cy + r[:, None] * np.sin(th)[None, :]), 0, self.size - 1).astype(np.int64)
            cache = {"idx": (y * self.size + x).ravel(), "shape": (self.r1 - self.r0, self.n_theta)}
            self._nearest = cache
        return cache

    def to_cartesian_nearest(self, polar: np.ndarray) -> np.ndarray:
        """Integer-dtype polar -> Cartesian with a single nearest tap (label-safe: no new values).

        [R, THETA] -> [H, W] and [T, R, THETA] -> [T, H, W]. The inverse of `to_polar_nearest`, and
        the counterpart of `to_cartesian`, which is bilinear and rounds to uint8 and so cannot carry
        class ids. Pixels outside the annulus stay 0 -- the r < r0 catheter core is not in the polar
        field of view at all, so a metric computed on the result must mask to `self.annulus`.
        """
        single = polar.ndim == 2
        if single:
            polar = polar[None]
        t = polar.shape[0]
        g = self._nearest_back_gather()
        out = np.zeros((t, self.size * self.size), dtype=polar.dtype)
        out[:, g["dst"]] = polar.reshape(t, -1)[:, g["src"]]
        out = out.reshape(t, self.size, self.size)
        return out[0] if single else out

    def _nearest_back_gather(self) -> dict:
        """Annulus pixel -> nearest polar cell, same convention as `cart_to_polar_index`."""
        cache = getattr(self, "_nearest_back", None)
        if cache is None:
            cx, cy = self.center
            yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float64)
            rad = np.hypot(xx - cx, yy - cy)[self.annulus]
            ang = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)[self.annulus]
            row = np.clip(np.rint(rad - self.r0), 0, (self.r1 - self.r0) - 1).astype(np.int64)
            col = np.rint(ang / (2 * np.pi) * self.n_theta).astype(np.int64) % self.n_theta
            cache = {"dst": np.flatnonzero(self.annulus.ravel()), "src": row * self.n_theta + col}
            self._nearest_back = cache
        return cache

    def cartesian_sampling_grid(self) -> np.ndarray:
        """float32 [H, W, 2] of (x, y) in [-1, 1] for `F.grid_sample(..., align_corners=True)`.

        The input to grid_sample is the polar tensor [B, C, R, THETA + 1] with column 0 appended
        after column THETA - 1, so the seam theta = 2pi interpolates back into theta = 0 instead of
        clamping. x indexes that padded theta axis (theta bin i sits at angle 2pi i / n_theta),
        y indexes r as `rad - r0`. Pixels outside the annulus get a clipped radius and hold
        whatever value grid_sample produces; the caller masks them. Built and cached on first use.
        """
        cache = getattr(self, "_sampling_grid", None)
        if cache is None:
            cx, cy = self.center
            yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float64)
            rad = np.hypot(xx - cx, yy - cy)
            ang = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
            theta_idx = ang / (2 * np.pi) * self.n_theta                 # in [0, n_theta]
            r_idx = np.clip(rad - self.r0, 0, (self.r1 - self.r0) - 1)
            x = theta_idx / self.n_theta * 2 - 1
            y = r_idx / ((self.r1 - self.r0) - 1) * 2 - 1
            cache = np.stack([x, y], axis=-1).astype(np.float32)
            self._sampling_grid = cache
        return cache

    def cart_to_polar_index(self, y: float, x: float) -> tuple[int, int]:
        """Cartesian (y, x) -> (row, col) of the polar grid: row = round(r - r0) clipped, col = theta bin mod n_theta."""
        cx, cy = self.center
        rad = float(np.hypot(x - cx, y - cy))
        ang = float(np.mod(np.arctan2(y - cy, x - cx), 2 * np.pi))
        row = int(np.clip(round(rad - self.r0), 0, self.r1 - self.r0 - 1))
        col = int(round(ang / (2 * np.pi) * self.n_theta)) % self.n_theta
        return row, col

    def qc_roundtrip(self, frame: np.ndarray) -> float:
        """PSNR (dB) of cart -> polar -> cart within the annulus."""
        back = self.to_cartesian(self.to_polar(frame))
        a = frame[self.annulus].astype(np.float64)
        b = back[self.annulus].astype(np.float64)
        mse = np.mean((a - b) ** 2)
        return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
