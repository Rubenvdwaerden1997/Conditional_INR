"""
Dataset for IV-OCT pullbacks stored as .npz files with sparse per-frame labels.

Annotated frame indices come from an Excel split file (same format used by the
old training pipeline). All other frames are masked with ignore_index=255 so
the loss function skips them automatically.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import rotate as nd_rotate
from torch.utils.data import Dataset

from config import Config
from model import resize_volume, resize_labels


_DEFAULT_LABEL_MAPPING: Dict[int, int] = {
    0:  0,   # Background
    1:  1,   # Lumen
    2:  2,   # Guidewire
    3:  3,   # Intima
    4:  4,   # Lipid
    5:  5,   # Calcium
    6:  6,   # Media
    7:  1,   # Catheter → Lumen
    8:  7,   # Sidebranch
    9:  8,   # Red thrombus
    10: 8,   # White thrombus → Red thrombus
    11: 3,   # Dissection → Intima
    12: 9,   # Plaque rupture
    13: 10,  # Layered plaque
    14: 11,  # Neovascularization
    15: 3,   # Intraplaque hemorrhage → Intima
    16: 3,   # Intramural hematoma → Intima
}


class OCTPullbackDataset(Dataset):
    """
    Each .npz file must contain:
        Volume_input_image : [1, D, H, W] float32
        Segmentation_image : [D, H, W]   int  (labels present for all frames)

    Only frames listed in entry["annotated_frames"] are used for supervised
    training; all other frames receive ignore_index=255.

    Preferred construction: OCTPullbackDataset.from_excel()
    """

    def __init__(
        self,
        entries: List[Dict],
        cfg: Config,
        n_points: int = 8192,
        mode: str = "train",
        augment: bool = True,
        label_mapping: Optional[Dict[int, int]] = None,
        mapping_activated: bool = True,
    ):
        self.entries           = entries
        self.cfg               = cfg
        self.n_pts             = n_points
        self.mode              = mode
        self.augment           = augment and (mode == "train")
        self.label_mapping     = label_mapping if label_mapping is not None else _DEFAULT_LABEL_MAPPING
        self.mapping_activated = mapping_activated

        self._file_cache: Dict[str, Dict] = {}
        if cfg.preload_data and mode == "train":
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            files = sorted({e["file"] for e in entries})
            n = len(files)
            frac = max(0.0, min(1.0, cfg.preload_frac))
            n_preload = max(1, int(round(n * frac)))
            preload_files = files[:n_preload]
            print(f"Preloading {n_preload}/{n} files into RAM "
                  f"({int(frac * 100)}% upfront, rest lazy-cached during training)...")
            report_every = max(1, n_preload // 20)
            def _load_one(f):
                return f, dict(np.load(f, allow_pickle=False))
            done = 0
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_load_one, f): f for f in preload_files}
                for future in as_completed(futures):
                    f, data = future.result()
                    self._file_cache[f] = data
                    done += 1
                    if done % report_every == 0 or done == n_preload:
                        elapsed = time.time() - t0
                        rate = done / elapsed if elapsed > 0 else 0
                        eta = (n_preload - done) / rate if rate > 0 else float("inf")
                        print(f"  Preloading {done * 100 // n_preload}% ({done}/{n_preload}) "
                              f"— {rate:.1f} files/s, ETA {eta:.0f}s")
            print("Preloading done.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_excel(
        cls,
        folders,
        excel_df: pd.DataFrame,
        cfg: Config,
        set_excel: str = "training",
        **kwargs,
    ) -> "OCTPullbackDataset":
        """Build dataset from folder(s) of .npz files + Excel split file.

        Excel must have columns:
            pullback  — stem of the .npz filename (no extension)
            set       — "training" | "validation" | "testing"
            frames    — comma-separated 1-based frame numbers that are annotated

        In "2D" mode (cfg.training_mode == "2D") each annotated frame becomes
        its own dataset entry.  In "3D" mode the full pullback is one entry and
        only annotated frames contribute to the loss.
        """
        if isinstance(folders, (str, Path)):
            folders = [folders]

        df = excel_df[excel_df["set"].str.lower() == set_excel.lower()]

        pullback_to_frames: Dict[str, List[int]] = {}
        for _, row in df.iterrows():
            pid    = str(row["pullback"])
            frames = [
                int(f.strip()) - 1          # 1-based → 0-based
                for f in str(row["frames"]).split(",")
                if f.strip().isdigit()
            ]
            pullback_to_frames.setdefault(pid, []).extend(frames)

        mode_2d = cfg.training_mode == "2D"
        entries = []
        for folder in folders:
            folder = Path(folder)
            if not folder.is_dir():
                raise ValueError(f"Data root not found: {folder}")
            for npz in sorted(folder.glob("*.npz")):
                pid = npz.stem.replace("_circ_gray", "")
                if pid not in pullback_to_frames:
                    continue
                if "Segmentation_image" not in np.load(npz, allow_pickle=False).files:
                    print(f"[{set_excel}] Skipping {npz.name} — no Segmentation_image key")
                    continue
                if mode_2d:
                    frames_folder = Path(cfg.frames_folder) if cfg.frames_folder else None
                    for z in pullback_to_frames[pid]:
                        frame_file   = frames_folder / f"{pid}_frame{z:04d}.npz" if frames_folder else None
                        use_prebuilt = frame_file is not None and frame_file.exists()
                        entries.append({
                            "file":      str(frame_file) if use_prebuilt else str(npz),
                            "pullback":  pid,
                            "frame_idx": None if use_prebuilt else z,
                            "prebuilt":  use_prebuilt,
                        })
                else:
                    patches_folder_3d = Path(cfg.patches_folder_3d) if cfg.patches_folder_3d else None
                    if patches_folder_3d:
                        # Prebuilt mode: one entry per annotated frame (faster IO)
                        for z in pullback_to_frames[pid]:
                            patch_file = patches_folder_3d / f"{pid}_frame{z:04d}_patch3d.npz"
                            if patch_file.exists():
                                entries.append({
                                    "file":        str(patch_file),
                                    "pullback":    pid,
                                    "prebuilt_3d": True,
                                })
                            # silently skip frames whose patch file hasn't been built yet
                    else:
                        # On-the-fly mode: one entry per pullback (slow, loads full volume)
                        entries.append({
                            "file":             str(npz),
                            "pullback":         pid,
                            "annotated_frames": pullback_to_frames[pid],
                        })

        unit = "frames" if (mode_2d or cfg.patches_folder_3d) else "pullbacks"
        print(f"[{set_excel}] {len(entries)} {unit} found.")
        return cls(entries, cfg, **kwargs)

    @classmethod
    def from_single_volume(
        cls,
        npz_path: str,
        annotated_frames: List[int],
        cfg: Config,
        mode: str = "train",
        **kwargs,
    ) -> "OCTPullbackDataset":
        """Build a dataset from one .npz for unconditional INR overfitting.

        Train set is repeated cfg.unconditional_n_repeats times so the DataLoader
        produces that many batches per epoch, each with freshly sampled coordinates.
        The file is preloaded into the in-memory cache at construction time.
        """
        npz_path = str(npz_path)
        entry = {
            "file":             npz_path,
            "pullback":         Path(npz_path).stem,
            "unconditional":    True,
            "annotated_frames": annotated_frames,
        }
        n       = cfg.unconditional_n_repeats if mode == "train" else 1
        entries = [entry] * n
        ds      = cls(entries, cfg, mode=mode, augment=(mode == "train"), **kwargs)
        # Preload file once
        if npz_path not in ds._file_cache:
            ds._file_cache[npz_path] = dict(np.load(npz_path, allow_pickle=False))

        # Precompute labeled coords once — avoids rebuilding a [D,H,W] array every step
        data = ds._file_cache[npz_path]
        vol  = data["Volume_input_image"]
        if vol.ndim == 4: vol = vol[0]
        segm = data["Segmentation_image"]
        if segm.ndim == 4: segm = segm[0]
        segm = segm.astype(np.int64)
        if kwargs.get("mapping_activated", True) and kwargs.get("label_mapping") is not None:
            segm = ds._remap(segm, kwargs["label_mapping"])
        D, H, W = vol.shape
        if cfg.resize_to:
            H = W = cfg.resize_to   # vol itself is unused beyond .shape — only labels need resizing

        labels = np.full((D, H, W), cfg.ignore_index, dtype=np.int64)
        for z in annotated_frames:
            if 0 <= z < D:
                label_frame = segm[z]
                if cfg.resize_to:
                    label_frame = resize_labels(torch.from_numpy(label_frame), cfg.resize_to).numpy()
                labels[z] = label_frame
        labeled_mask = labels != cfg.ignore_index
        labeled_zyx  = np.argwhere(labeled_mask).astype(np.int32)   # [M, 3]
        label_vals   = labels[labeled_zyx[:, 0], labeled_zyx[:, 1], labeled_zyx[:, 2]].astype(np.int32)

        ds._unc_labeled_zyx  = labeled_zyx   # [M, 3] (z, y, x)
        ds._unc_label_vals   = label_vals    # [M]
        ds._unc_volume_shape = (D, H, W)
        return ds

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        if entry.get("unconditional"):
            return self._get_unconditional(entry)

        if self.cfg.training_mode == "2D":
            frame, label_frame = self._load_2d(entry)
            if self.augment:
                frame, label_frame = self._augment_2d(frame, label_frame)
            coords, point_labels = self._sample_coords_2d(frame if frame.ndim == 2 else frame[frame.shape[0] // 2], label_frame)
            frame_t = torch.from_numpy(frame).float()
            if frame_t.ndim == 2:
                frame_t = frame_t.unsqueeze(0)   # [1, H, W] for non-prebuilt single frame
            return (
                frame_t,                                                    # [C, H, W]
                torch.from_numpy(coords).float(),                           # [N, 2]
                torch.from_numpy(point_labels).long(),                      # [N]
                torch.from_numpy(label_frame.copy()).long(),                # [H, W] full GT for dense supervision
            )

        volume, labels = self._load_3d_patch(entry)
        if self.augment:
            volume, labels = self._augment(volume, labels)
        coords, point_labels = self._sample_coords(volume, labels)
        return (
            torch.from_numpy(volume).unsqueeze(0).float(),                 # [1, patch_z, H, W]
            torch.from_numpy(coords).float(),                               # [N, 3]
            torch.from_numpy(point_labels).long(),                          # [N]
            torch.from_numpy(labels).long(),                               # [patch_z, H, W] dense GT (255 for unannotated frames)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_unconditional(self, entry: Dict):
        """Sample coords+labels from the precomputed labeled-pixel arrays.

        No per-call allocation of the full [D,H,W] volume — all heavy work is
        done once at from_single_volume() construction time.
        """
        D, H, W     = self._unc_volume_shape
        labeled_zyx = self._unc_labeled_zyx   # [M, 3]  int32
        label_vals  = self._unc_label_vals    # [M]     int32

        # Sample integer positions into labeled_zyx rather than the coords themselves,
        # so we can look up both coordinates and labels cheaply after sampling.
        positions = np.arange(len(labeled_zyx), dtype=np.int64)

        if self.mode == "train":
            _sample_fn = (
                self._stratified_sample_with_bg_floor
                if self.cfg.background_floor_frac > 0
                else self._stratified_sample
            )
            chosen_pos   = _sample_fn(positions, label_vals, self.n_pts)
            coords_zyx   = labeled_zyx[chosen_pos]
            point_labels = label_vals[chosen_pos]
        else:
            max_pts = self.n_pts * 4
            if len(labeled_zyx) > max_pts:
                stride     = max(1, len(labeled_zyx) // max_pts)
                idx        = np.arange(0, len(labeled_zyx), stride)[:max_pts]
            else:
                idx        = np.arange(len(labeled_zyx))
            coords_zyx   = labeled_zyx[idx]
            point_labels = label_vals[idx]

        coords = coords_zyx[:, [2, 1, 0]].astype(np.float32)  # (z,y,x) → (x,y,z)
        return (
            torch.tensor([D, H, W], dtype=torch.long),
            torch.from_numpy(coords).float(),
            torch.from_numpy(point_labels.astype(np.int64)).long(),
            torch.empty(0, dtype=torch.long),
        )

    def _load(self, entry: Dict):
        """Load .npz, normalise volume, mask non-annotated frames."""
        data = np.load(entry["file"])

        vol = data["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]                        # [1, D, H, W] → [D, H, W]
        volume = vol.astype(np.float32)         # already z-score normalised upstream

        segm = data["Segmentation_image"]
        if segm.ndim == 4:
            segm = segm[0]                      # [1, D, H, W] → [D, H, W]
        segm = segm.astype(np.int64)

        if self.mapping_activated and self.label_mapping:
            segm = self._remap(segm, self.label_mapping)

        labels = np.full_like(segm, self.cfg.ignore_index)
        for z in entry["annotated_frames"]:
            if 0 <= z < labels.shape[0]:
                labels[z] = segm[z]

        return volume, labels

    def _load_3d_patch(self, entry: Dict):
        """Load a z-patch of patch_z frames centered on one annotated frame.

        Prebuilt mode (entry["prebuilt_3d"] == True):
            Loads a small pre-extracted patch file — fast, labels already remapped.

        On-the-fly mode (fallback):
            Loads the full pullback volume and slices a patch at runtime.
            Training: annotated frame chosen randomly each call.
            Validation: middle annotated frame (deterministic).
        """
        if entry.get("prebuilt_3d"):
            data = self._file_cache.get(entry["file"]) or dict(np.load(entry["file"], allow_pickle=False))
            buffer         = data["patch"].astype(np.float32)    # [2*patch_z, H, W]
            z_local_buffer = int(data["z_local"])                # primary annotated frame in buffer
            if "labels" in data:
                labels_buf = data["labels"].astype(np.int64)     # [2*patch_z, H, W] — 255 for unannotated
            else:
                # old format: single label [H,W] at z_local; fill rest with ignore_index
                labels_buf = np.full(buffer.shape, self.cfg.ignore_index, dtype=np.int64)
                labels_buf[z_local_buffer] = data["label"].astype(np.int64)
            pz             = self.cfg.patch_z
            buf_size       = buffer.shape[0]                     # == 2*patch_z

            if self.mode == "train":
                # Random sub-window: primary annotated frame lands anywhere in [0, pz-1]
                s_min = max(0, z_local_buffer - (pz - 1))
                s_max = min(buf_size - pz, z_local_buffer)
                s     = int(np.random.randint(s_min, s_max + 1))
            else:
                # Fixed sub-window centred on the primary annotated frame (deterministic)
                s = int(np.clip(z_local_buffer - pz // 2, 0, buf_size - pz))

            patch_vol    = buffer[s : s + pz]
            patch_labels = labels_buf[s : s + pz]               # all annotated frames in window included
            patch_vol, patch_labels = self._resize_xy_labels(patch_vol, patch_labels)
            return patch_vol, patch_labels

        # --- on-the-fly fallback ---
        data = self._file_cache.get(entry["file"]) or dict(np.load(entry["file"], allow_pickle=False))

        vol = data["Volume_input_image"]
        if vol.ndim == 4:
            vol = vol[0]

        segm = data["Segmentation_image"]
        if segm.ndim == 4:
            segm = segm[0]
        segm = segm.astype(np.int64)

        if self.mapping_activated and self.label_mapping:
            segm = self._remap(segm, self.label_mapping)

        D, H, W = vol.shape
        ann = [z for z in entry["annotated_frames"] if 0 <= z < D]
        if not ann:
            ann = [D // 2]

        z_center = (
            int(np.random.choice(ann))
            if self.mode == "train"
            else ann[len(ann) // 2]
        )

        pz      = self.cfg.patch_z
        half    = pz // 2
        z_start = int(np.clip(z_center - half, 0, max(0, D - pz)))
        z_end   = z_start + pz

        patch_vol = vol[z_start:z_end].astype(np.float32)

        if patch_vol.shape[0] < pz:
            pad       = pz - patch_vol.shape[0]
            patch_vol = np.pad(patch_vol, ((0, pad), (0, 0), (0, 0)), mode="edge")

        _, H, W      = patch_vol.shape
        patch_labels = np.full((pz, H, W), self.cfg.ignore_index, dtype=np.int64)
        for z_ann in ann:
            if z_start <= z_ann < z_end:
                patch_labels[z_ann - z_start] = segm[z_ann]
        patch_vol, patch_labels = self._resize_xy_labels(patch_vol, patch_labels)
        return patch_vol, patch_labels

    def _remap(self, seg: np.ndarray, merge_map: Dict[int, int]) -> np.ndarray:
        lut_size = int(max(int(seg.max()), max(merge_map.keys()))) + 1
        lut = np.arange(lut_size, dtype=np.int64)
        for old, new in merge_map.items():
            lut[old] = new
        return lut[seg]

    def _resize_xy_labels(self, volume: np.ndarray, labels: np.ndarray):
        """Resize volume [Z, H, W] and labels [Z, H, W] together.

        Labels use nearest-neighbour so class boundaries stay crisp.
        No-op when cfg.resize_to == 0.
        """
        size = self.cfg.resize_to
        if not size:
            return volume, labels
        volume = resize_volume(torch.from_numpy(volume), size).numpy()
        labels = resize_labels(torch.from_numpy(labels), size).numpy()
        return volume, labels

    def _resize_xy(self, volume: np.ndarray, label_frame: np.ndarray):
        """Resize the (H, W) dims of a volume/frame and its single label frame.

        No-op when cfg.resize_to == 0. Bilinear for the image, nearest-neighbour
        for labels so class boundaries stay crisp instead of being blended.
        """
        size = self.cfg.resize_to
        if not size:
            return volume, label_frame
        volume      = resize_volume(torch.from_numpy(volume), size).numpy()
        label_frame = resize_labels(torch.from_numpy(label_frame), size).numpy()
        return volume, label_frame

    def _augment(self, volume: np.ndarray, labels: np.ndarray):
        """Safe OCT augmentations — no z-flips (pullback direction must stay intact)."""
        if np.random.rand() < 0.5:
            volume = volume[:, :, ::-1].copy()
            labels = labels[:, :, ::-1].copy()

        # Random in-plane (x/y) rotation — identical angle for all z-frames
        angle  = np.random.uniform(0.0, 360.0)
        volume = nd_rotate(volume, angle, axes=(1, 2), reshape=False,
                           order=1, mode="constant", cval=0.0)
        labels = nd_rotate(labels, angle, axes=(1, 2), reshape=False,
                           order=0, mode="constant", cval=self.cfg.ignore_index)

        gain   = np.random.uniform(0.8, 1.2)
        volume = volume * gain
        return volume, labels

    def _stratified_sample(self, indices: np.ndarray, label_vals: np.ndarray,
                           n: int) -> np.ndarray:
        """Sample n indices equally distributed across unique classes in label_vals."""
        classes   = np.unique(label_vals)
        n_per_cls = max(1, n // len(classes))
        chosen    = []
        for c in classes:
            pool = indices[label_vals == c]
            chosen.append(pool[np.random.choice(len(pool), size=n_per_cls, replace=True)])
        result = np.concatenate(chosen, axis=0)
        # Trim to exactly n (rounding may overshoot by at most n_classes-1)
        return result[:n]

    def _stratified_sample_with_bg_floor(
        self,
        indices: np.ndarray,
        label_vals: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Stratified sample with a guaranteed minimum for background (class 0).

        Reserves background_floor_frac * n_pts points for class 0, then
        distributes the remainder equally across all other present classes.
        Falls back to plain stratified if no background pixels are present.
        """
        n_bg_floor = int(self.cfg.background_floor_frac * self.n_pts)
        bg_mask    = label_vals == 0
        bg_idx     = indices[bg_mask]

        if n_bg_floor == 0 or len(bg_idx) == 0:
            return self._stratified_sample(indices, label_vals, n)

        n_bg    = min(n_bg_floor, len(bg_idx))
        n_other = max(0, n - n_bg)

        bg_chosen = bg_idx[np.random.choice(len(bg_idx), size=n_bg, replace=True)]

        other_idx  = indices[~bg_mask]
        other_vals = label_vals[~bg_mask]
        if n_other > 0 and len(other_idx) > 0:
            other_chosen = self._stratified_sample(other_idx, other_vals, n_other)
            return np.concatenate([bg_chosen, other_chosen], axis=0)
        return bg_chosen[:n]

    def _sample_coords(self, volume: np.ndarray, labels: np.ndarray):
        """Sample N query coordinates, biased 70 % toward labeled pixels."""
        D, H, W = volume.shape

        labeled_mask = labels != self.cfg.ignore_index
        labeled_zyx  = np.argwhere(labeled_mask)   # [M, 3] in (z, y, x)

        if self.mode == "train":
            n_labeled = int(self.n_pts * 0.7)
        else:
            # Validation: use ALL labeled pixels for a deterministic, low-variance metric
            n_labeled = len(labeled_zyx)

        if len(labeled_zyx) == 0:
            coords = np.random.randint(
                [0, 0, 0], [D, H, W], size=(self.n_pts, 3)
            ).astype(np.float32)
            point_labels = np.full(self.n_pts, self.cfg.ignore_index, dtype=np.int64)
            return coords[:, [2, 1, 0]], point_labels   # → (x, y, z)

        if self.mode == "train":
            label_vals = labels[labeled_zyx[:, 0], labeled_zyx[:, 1], labeled_zyx[:, 2]]
            if self.cfg.sampling_strategy == "stratified":
                _sample_fn = (
                    self._stratified_sample_with_bg_floor
                    if self.cfg.background_floor_frac > 0
                    else self._stratified_sample
                )
                chosen = _sample_fn(labeled_zyx, label_vals,
                                    min(n_labeled, len(labeled_zyx)))
            else:
                chosen = labeled_zyx[
                    np.random.choice(len(labeled_zyx),
                                     size=min(n_labeled, len(labeled_zyx)),
                                     replace=True)
                ]
            n_random  = self.n_pts - len(chosen)
            ann_z     = int(np.where(labeled_mask.any(axis=(1, 2)))[0][0])
            random_yx = np.random.randint([0, 0], [H, W], size=(n_random, 2))
            random_zyx = np.column_stack([
                np.full(n_random, ann_z, dtype=np.int64),
                random_yx,
            ])
            random_labels = labels[random_zyx[:, 0], random_zyx[:, 1], random_zyx[:, 2]]
            all_zyx    = np.concatenate([chosen, random_zyx], axis=0)
            all_labels = np.concatenate([labels[chosen[:, 0], chosen[:, 1], chosen[:, 2]],
                                         random_labels])
        else:
            # Deterministic stride-based subsample: same pixels every epoch, no randomness.
            # Cap at 4× training budget to stay within GPU memory.
            max_pts = self.n_pts * 4
            if len(labeled_zyx) > max_pts:
                stride = max(1, len(labeled_zyx) // max_pts)
                all_zyx = labeled_zyx[::stride][:max_pts]
            else:
                all_zyx = labeled_zyx
            all_labels = labels[all_zyx[:, 0], all_zyx[:, 1], all_zyx[:, 2]]

        coords = all_zyx[:, [2, 1, 0]].astype(np.float32)   # (z,y,x) → (x,y,z)
        if self.mode == "train" and self.cfg.coord_jitter_max > 0:
            jitter_xy = np.random.uniform(
                -self.cfg.coord_jitter_max, self.cfg.coord_jitter_max, size=(coords.shape[0], 2)
            ).astype(np.float32)
            coords[:, :2] += jitter_xy   # x, y only -- z stays exactly on the labeled frame
        return coords, all_labels.astype(np.int64)

    # ------------------------------------------------------------------
    # 2D helpers
    # ------------------------------------------------------------------

    def _load_2d(self, entry: Dict):
        """Load a single annotated frame (or pre-built context stack) from .npz."""
        data = self._file_cache.get(entry["file"]) or dict(np.load(entry["file"]))

        if entry.get("prebuilt"):
            # Pre-split file: frame is [C, H, W], label is [H, W]
            frame       = data["frame"].astype(np.float32)
            label_frame = data["label"].astype(np.int64)
        else:
            vol = data["Volume_input_image"]
            if vol.ndim == 4:
                vol = vol[0]
            segm = data["Segmentation_image"]
            if segm.ndim == 4:
                segm = segm[0]
            segm = segm.astype(np.int64)
            z           = entry["frame_idx"]
            frame       = vol[z].astype(np.float32)   # [H, W]
            label_frame = segm[z]

        if self.mapping_activated and self.label_mapping:
            label_frame = self._remap(label_frame, self.label_mapping)
        frame, label_frame = self._resize_xy(frame, label_frame)
        return frame, label_frame

    def _augment_2d(self, frame: np.ndarray, label_frame: np.ndarray):
        """Horizontal flip + random rotation + brightness jitter for a single 2-D frame."""
        if np.random.rand() < 0.5:
            # frame is [H, W] or [C, H, W] — flip the W axis in both cases
            frame       = np.flip(frame, axis=-1).copy()
            label_frame = label_frame[:, ::-1].copy()

        # Random in-plane (x/y) rotation — OCT frames are rotationally symmetric
        angle = np.random.uniform(0.0, 360.0)
        rot_axes = (1, 2) if frame.ndim == 3 else None   # [C,H,W] vs [H,W]
        frame = (nd_rotate(frame, angle, axes=rot_axes, reshape=False,
                           order=1, mode="constant", cval=0.0)
                 if rot_axes is not None else
                 nd_rotate(frame, angle, reshape=False,
                           order=1, mode="constant", cval=0.0))
        label_frame = nd_rotate(label_frame, angle, reshape=False,
                                order=0, mode="constant", cval=self.cfg.ignore_index)

        gain  = np.random.uniform(0.8, 1.2)
        frame = frame * gain
        return frame, label_frame

    def _sample_coords_2d(self, frame: np.ndarray, label_frame: np.ndarray):
        """Sample N query 2-D coordinates, biased 70 % toward labeled pixels."""
        H, W = frame.shape

        labeled_mask = label_frame != self.cfg.ignore_index
        labeled_yx   = np.argwhere(labeled_mask)   # [M, 2] in (y, x)

        if self.mode == "train":
            n_labeled = int(self.n_pts * 0.7)
        else:
            n_labeled = len(labeled_yx)

        if len(labeled_yx) == 0:
            coords = np.random.randint([0, 0], [H, W], size=(self.n_pts, 2)).astype(np.float32)
            point_labels = np.full(self.n_pts, self.cfg.ignore_index, dtype=np.int64)
            return coords[:, [1, 0]], point_labels   # → (x, y)

        if self.mode == "train":
            label_vals = label_frame[labeled_yx[:, 0], labeled_yx[:, 1]]
            if self.cfg.sampling_strategy == "stratified":
                _sample_fn = (
                    self._stratified_sample_with_bg_floor
                    if self.cfg.background_floor_frac > 0
                    else self._stratified_sample
                )
                chosen = _sample_fn(labeled_yx, label_vals,
                                    min(n_labeled, len(labeled_yx)))
            else:
                chosen = labeled_yx[
                    np.random.choice(len(labeled_yx),
                                     size=min(n_labeled, len(labeled_yx)),
                                     replace=True)
                ]
            n_random      = self.n_pts - len(chosen)
            random_yx     = np.random.randint([0, 0], [H, W], size=(n_random, 2))
            random_labels = label_frame[random_yx[:, 0], random_yx[:, 1]]
            all_yx     = np.concatenate([chosen, random_yx], axis=0)
            all_labels = np.concatenate([label_frame[chosen[:, 0], chosen[:, 1]],
                                         random_labels])
        else:
            max_pts = self.n_pts * 4
            if len(labeled_yx) > max_pts:
                stride = max(1, len(labeled_yx) // max_pts)
                all_yx = labeled_yx[::stride][:max_pts]
            else:
                all_yx = labeled_yx
            all_labels = label_frame[all_yx[:, 0], all_yx[:, 1]]

        coords = all_yx[:, [1, 0]].astype(np.float32)   # (y, x) → (x, y)
        return coords, all_labels.astype(np.int64)
