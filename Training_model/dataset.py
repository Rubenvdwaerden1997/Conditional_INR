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

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

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
            torch.empty(0, dtype=torch.long),                              # dense supervision not used in 3D
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            data            = np.load(entry["file"], allow_pickle=False)
            buffer          = data["patch"].astype(np.float32)   # [2*patch_z, H, W]
            label_frame     = data["label"].astype(np.int64)     # [H, W]  already remapped
            z_local_buffer  = int(data["z_local"])               # annotated frame in buffer
            pz              = self.cfg.patch_z
            buf_size        = buffer.shape[0]                    # == 2*patch_z

            if self.mode == "train":
                # Random sub-window: annotated frame lands anywhere in [0, pz-1]
                s_min = max(0, z_local_buffer - (pz - 1))
                s_max = min(buf_size - pz, z_local_buffer)
                s     = int(np.random.randint(s_min, s_max + 1))
            else:
                # Fixed centre sub-window for deterministic validation
                s = (buf_size - pz) // 2

            patch_vol   = buffer[s : s + pz]
            z_local     = z_local_buffer - s
            _, H, W     = patch_vol.shape
            patch_labels = np.full((pz, H, W), self.cfg.ignore_index, dtype=np.int64)
            patch_labels[z_local] = label_frame
            return patch_vol, patch_labels

        # --- on-the-fly fallback ---
        data = np.load(entry["file"], allow_pickle=False)

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

        patch_labels = np.full((pz, H, W), self.cfg.ignore_index, dtype=np.int64)
        z_local      = z_center - z_start
        patch_labels[z_local] = segm[z_center]

        return patch_vol, patch_labels

    def _remap(self, seg: np.ndarray, merge_map: Dict[int, int]) -> np.ndarray:
        lut_size = int(max(int(seg.max()), max(merge_map.keys()))) + 1
        lut = np.arange(lut_size, dtype=np.int64)
        for old, new in merge_map.items():
            lut[old] = new
        return lut[seg]

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
            chosen.append(pool[np.random.choice(len(pool), size=min(n_per_cls, len(pool)),
                                                replace=True)])
        result = np.concatenate(chosen, axis=0)
        # Trim to exactly n (rounding may overshoot by at most n_classes-1)
        return result[:n]

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
                chosen = self._stratified_sample(labeled_zyx, label_vals,
                                                 min(n_labeled, len(labeled_zyx)))
            else:
                chosen = labeled_zyx[
                    np.random.choice(len(labeled_zyx),
                                     size=min(n_labeled, len(labeled_zyx)),
                                     replace=True)
                ]
            n_random      = self.n_pts - len(chosen)
            random_zyx    = np.random.randint([0, 0, 0], [D, H, W], size=(n_random, 3))
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
        return coords, all_labels.astype(np.int64)

    # ------------------------------------------------------------------
    # 2D helpers
    # ------------------------------------------------------------------

    def _load_2d(self, entry: Dict):
        """Load a single annotated frame (or pre-built context stack) from .npz."""
        data = np.load(entry["file"])

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
                chosen = self._stratified_sample(labeled_yx, label_vals,
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
