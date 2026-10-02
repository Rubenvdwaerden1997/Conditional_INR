"""Streamed readers for raw pullbacks: nii.gz (RGB24 / uint8 / other), multi-frame DICOM, npz.

Orientation convention: every reader yields `[frame, row, col]`; a label volume equals
`np.asarray(nib.load(p).dataobj).transpose(2, 1, 0)`.

Dispatch:
  .nii(.gz)  datatype 128, vox_offset 352, no scaling -> `read_niigz_frames` + `rgb_to_gray` per batch
             datatype 2, same stream                  -> gray_source `uint8_direct`
             any other header                         -> nibabel full load, uint8 view [cols,rows,Z,3]
                                                         -> transpose(2,1,0,3)
  .dcm       SimpleITK; 3 components -> `rgb_to_gray` per batch, 1 -> `uint8_direct`;
             pixel spacing from tag 0028|0030, frame count checked against tag 0028|0008
  .npz       key "image" [Z,rows,cols] gray or [Z,rows,cols,3] RGB, optional "pixel_spacing_mm"

The LUT is built once per process by the build and passed in; readers never build one.
"""
from __future__ import annotations

import gzip
import zlib
from dataclasses import dataclass
from typing import Iterator

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from ..preprocess.colormap import rgb_to_gray

DT_RGB24, DT_UINT8 = 128, 2
NIFTI_SUFFIXES = (".nii.gz", ".nii")
TAG_N_FRAMES = "0028|0008"
TAG_PIXEL_SPACING = "0028|0030"
_READ_ERRORS = (OSError, EOFError, zlib.error, RuntimeError)


@dataclass
class VolumeMeta:
    n_frames: int
    rows: int
    cols: int
    source_format: str            # niigz-rgb24 | niigz-uint8 | niigz-nibabel | dcm-rgb | dcm-gray | npz
    pixel_spacing_mm: float | None


class BrokenSourceError(Exception):
    """Unreadable / truncated source; the build EXCLUDES the pullback and records the reason."""


def read_niigz_frames(path: str, cols: int, rows: int, frames: int, datatype: int,
                      batch: int = 32) -> Iterator[np.ndarray]:
    """Yield uint8 frame batches [B, rows, cols(, 3)] streamed from a .nii.gz (vox_offset 352, no scaling)."""
    ch = 3 if datatype == DT_RGB24 else 1
    frame_bytes = rows * cols * ch
    with gzip.open(path, "rb") as f:
        if len(f.read(352)) != 352:
            raise IOError("truncated nifti header")
        for start in range(0, frames, batch):
            n = min(batch, frames - start)
            buf = f.read(frame_bytes * n)
            if len(buf) != frame_bytes * n:
                raise IOError(f"truncated pixel data at frame {start}")
            arr = np.frombuffer(buf, dtype=np.uint8)
            shape = (n, rows, cols, 3) if ch == 3 else (n, rows, cols)
            yield arr.reshape(shape)


def derive_z_spacing(frames: int, table: dict) -> tuple[float | None, str]:
    """(z spacing mm, "table") for a frame count within 2 of a table key, else (None, "unknown")."""
    for n, (_mm, spacing) in table.items():
        if abs(frames - int(n)) <= 2:
            return spacing, "table"
    return None, "unknown"


def nifti_raw_header(path: str) -> nib.Nifti1Header:
    """Parse the first 348 bytes of the (gzip) stream; never depends on a JSON sidecar."""
    try:
        with open(path, "rb") as raw:
            gzipped = raw.read(2) == b"\x1f\x8b"
        opener = gzip.open if gzipped else open
        with opener(path, "rb") as stream:
            block = stream.read(348)
    except _READ_ERRORS as ex:
        raise BrokenSourceError(f"{path}: cannot read nifti header ({type(ex).__name__}: {ex})") from ex
    if len(block) != 348:
        raise BrokenSourceError(f"{path}: truncated nifti header ({len(block)} of 348 bytes)")
    header = nib.Nifti1Header(binaryblock=block, endianness=None, check=False)
    if int(header["sizeof_hdr"]) != 348 or bytes(header["magic"]).rstrip(b"\x00") not in (b"n+1", b"ni1"):
        raise BrokenSourceError(f"{path}: not a NIfTI-1 header")
    return header


def probe_image(path: str) -> VolumeMeta:
    """Header-only probe: nifti raw header / SimpleITK ReadImageInformation / npz. Raises BrokenSourceError.

    (`.npz` is the exception: it has no header to read, so the array is loaded.)
    """
    if path.endswith(NIFTI_SUFFIXES):
        return _probe_nifti(path)
    if path.endswith(".dcm"):
        return _probe_dicom(path)
    if path.endswith(".npz"):
        return _probe_npz(path)
    raise BrokenSourceError(f"{path}: unsupported image format")


def iter_image_gray(path: str, lut: dict, batch: int = 32) -> Iterator[tuple[np.ndarray, float, str]]:
    """Yield (gray uint8 [B, rows, cols], offcurve_frac, gray_source) batches.

    gray_source: vendor_lut | file_lut | uint8_direct. Raises BrokenSourceError on any read failure,
    on a gzip/pixel-data short read, when a DICOM yields fewer frames than its NumberOfFrames tag,
    or when a `uint8_direct` volume carries values outside the 0..255 palette scale.
    """
    meta = probe_image(path)
    if meta.source_format in ("niigz-rgb24", "niigz-uint8"):
        yield from _iter_niigz_stream(path, meta, lut, batch)
    elif meta.source_format == "niigz-nibabel":
        yield from _iter_volume(_load_nibabel_volume(path), lut, batch)
    elif meta.source_format.startswith("dcm"):
        yield from _iter_volume(_load_dicom_volume(path), lut, batch)
    else:
        yield from _iter_volume(_load_npz_volume(path), lut, batch)


def read_label_volume(path: str) -> np.ndarray:
    """uint8 [Z, rows, cols] == np.asarray(dataobj).transpose(2, 1, 0); asserts max <= 16."""
    array = np.asarray(nib.load(path).dataobj)
    if array.ndim != 3:
        raise ValueError(f"{path}: label volume has shape {array.shape}, expected 3 dimensions")
    top = int(array.max()) if array.size else 0
    if top > 16:
        raise ValueError(f"{path}: label value {top} exceeds the 17-class range 0..16")
    return array.transpose(2, 1, 0).astype(np.uint8, order="C")


def uint8_direct_check(frames: np.ndarray) -> None:
    """Raise BrokenSourceError unless values fit the 0..255 palette scale.

    `_gray_batch` casts a gray source to uint8, which would store anything else mod 256 (4096
    becomes 0), so the source is refused: the build excludes that pullback with this reason
    instead of storing silently wrong gray levels.
    """
    array = np.asarray(frames)
    if array.size == 0:
        return
    low, high = int(array.min()), int(array.max())
    if low < 0 or high > 255:
        raise BrokenSourceError(
            f"uint8_direct source has values [{low}, {high}] outside the 0..255 palette scale; "
            f"its gray levels do not share the vendor LUT scale")


# ---------------------------------------------------------------- probes
def _probe_nifti(path: str) -> VolumeMeta:
    header = nifti_raw_header(path)
    shape = header.get_data_shape()
    if len(shape) < 3:
        raise BrokenSourceError(f"{path}: nifti shape {shape} is not a frame stack")
    cols, rows, n_frames = int(shape[0]), int(shape[1]), int(shape[2])
    datatype, offset = int(header["datatype"]), float(header["vox_offset"])
    slope, inter = float(header["scl_slope"]), float(header["scl_inter"])
    scaled = not (np.isnan(slope) or slope in (0.0, 1.0)) or (not np.isnan(inter) and inter != 0.0)
    streamable = offset == 352.0 and not scaled
    if datatype == DT_RGB24 and streamable:
        source_format = "niigz-rgb24"
    elif datatype == DT_UINT8 and streamable:
        source_format = "niigz-uint8"
    else:
        source_format = "niigz-nibabel"
    spacing = float(header["pixdim"][1])
    return VolumeMeta(n_frames, rows, cols, source_format, spacing or None)


def _dicom_reader(path: str) -> sitk.ImageFileReader:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    try:
        reader.ReadImageInformation()
    except _READ_ERRORS as ex:
        raise BrokenSourceError(f"{path}: cannot read DICOM header ({type(ex).__name__}: {ex})") from ex
    return reader


def _probe_dicom(path: str) -> VolumeMeta:
    reader = _dicom_reader(path)
    size = reader.GetSize()
    if len(size) < 3:
        raise BrokenSourceError(f"{path}: DICOM size {size} is not a frame stack")
    components = reader.GetNumberOfComponents()
    if components not in (1, 3):
        raise BrokenSourceError(f"{path}: DICOM has {components} components, expected 1 or 3")
    spacing = None
    if reader.HasMetaDataKey(TAG_PIXEL_SPACING):
        try:
            spacing = float(reader.GetMetaData(TAG_PIXEL_SPACING).strip().split("\\")[0])
        except ValueError:
            spacing = None
    source_format = "dcm-rgb" if components == 3 else "dcm-gray"
    return VolumeMeta(int(size[2]), int(size[1]), int(size[0]), source_format, spacing)


def _declared_dicom_frames(path: str) -> int:
    reader = _dicom_reader(path)
    if reader.HasMetaDataKey(TAG_N_FRAMES):
        try:
            return int(reader.GetMetaData(TAG_N_FRAMES).strip())
        except ValueError:
            pass
    return int(reader.GetSize()[2])


def _probe_npz(path: str) -> VolumeMeta:
    volume, spacing = _read_npz(path)
    rows, cols = volume.shape[1], volume.shape[2]
    return VolumeMeta(int(volume.shape[0]), int(rows), int(cols), "npz", spacing)


# ---------------------------------------------------------------- volume loading
def _read_npz(path: str) -> tuple[np.ndarray, float | None]:
    try:
        with np.load(path) as data:
            if "image" not in data:
                raise BrokenSourceError(f"{path}: npz has no 'image' key (keys: {list(data)})")
            volume = data["image"]
            spacing = float(data["pixel_spacing_mm"]) if "pixel_spacing_mm" in data else None
    except (OSError, EOFError, ValueError, zlib.error) as ex:
        raise BrokenSourceError(f"{path}: cannot read npz ({type(ex).__name__}: {ex})") from ex
    if volume.ndim not in (3, 4) or (volume.ndim == 4 and volume.shape[3] != 3):
        raise BrokenSourceError(f"{path}: npz 'image' has shape {volume.shape}, "
                                f"expected [Z,rows,cols] or [Z,rows,cols,3]")
    return volume, spacing


def _load_npz_volume(path: str) -> np.ndarray:
    return _read_npz(path)[0]


def _load_nibabel_volume(path: str) -> np.ndarray:
    """Full nibabel load for nifti headers the raw stream cannot be trusted for."""
    try:
        array = np.asarray(nib.load(path).dataobj)
    except _READ_ERRORS as ex:
        raise BrokenSourceError(f"{path}: cannot read nifti ({type(ex).__name__}: {ex})") from ex
    if array.dtype.fields is not None:                     # RGB24: (cols, rows, Z) of (R,G,B)
        shape = array.shape
        array = np.ascontiguousarray(array).view(np.uint8).reshape(shape + (3,))
        return array.transpose(2, 1, 0, 3)
    return array.transpose(2, 1, 0)


def _load_dicom_volume(path: str) -> np.ndarray:
    declared = _declared_dicom_frames(path)
    try:
        image = sitk.ReadImage(path)
    except _READ_ERRORS as ex:
        raise BrokenSourceError(f"{path}: cannot read DICOM ({type(ex).__name__}: {ex})") from ex
    volume = sitk.GetArrayFromImage(image)                 # [Z, rows, cols(, 3)]
    if volume.shape[0] < declared:
        raise BrokenSourceError(f"{path}: DICOM yielded {volume.shape[0]} frames, "
                                f"NumberOfFrames ({TAG_N_FRAMES}) says {declared}")
    return volume


# ---------------------------------------------------------------- gray streams
def _gray_source(lut: dict) -> str:
    return "vendor_lut" if lut.get("source") == "vendor" else "file_lut"


def _gray_batch(block: np.ndarray, lut: dict) -> tuple[np.ndarray, float, str]:
    """One (gray uint8, offcurve_frac, gray_source) batch from RGB [B,rows,cols,3] or gray [B,rows,cols]."""
    if block.ndim == 4:
        gray, offcurve = rgb_to_gray(block, lut)
        return gray, offcurve, _gray_source(lut)
    return np.ascontiguousarray(block, dtype=np.uint8), 0.0, "uint8_direct"


def _iter_niigz_stream(path: str, meta: VolumeMeta, lut: dict,
                       batch: int) -> Iterator[tuple[np.ndarray, float, str]]:
    """The fast path: raw frames straight out of the gzip stream, no nibabel.

    No `uint8_direct_check` here: datatype 2 is uint8 in the file, so its values are 0..255
    by construction. Only the nibabel / dicom / npz paths can carry another dtype.
    """
    datatype = DT_RGB24 if meta.source_format == "niigz-rgb24" else DT_UINT8
    try:
        for block in read_niigz_frames(path, meta.cols, meta.rows, meta.n_frames, datatype, batch):
            yield _gray_batch(block, lut)
    except _READ_ERRORS as ex:
        raise BrokenSourceError(f"{path}: {type(ex).__name__}: {ex}") from ex


def _iter_volume(volume: np.ndarray, lut: dict, batch: int) -> Iterator[tuple[np.ndarray, float, str]]:
    """Batch an in-memory volume [Z, rows, cols(, 3)] into the same (gray, offcurve, source) stream.

    The whole volume is already in memory, so the palette-scale check covers every frame before
    the first batch is yielded (a per-batch check would miss an out-of-range value that only
    appears in a later batch, after wrong gray levels had already been written).
    """
    if volume.ndim == 3:
        uint8_direct_check(volume)
    for start in range(0, volume.shape[0], batch):
        yield _gray_batch(volume[start:start + batch], lut)
