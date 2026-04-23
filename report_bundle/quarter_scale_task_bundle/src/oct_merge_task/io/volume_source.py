from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import tifffile


Index3 = Tuple[int, int, int]


@dataclass
class VolumeSource:
    path: Path
    array: np.ndarray

    @property
    def shape(self) -> Index3:
        return tuple(int(v) for v in self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.array.dtype)

    def read_region(
        self,
        origin: Sequence[int],
        shape: Sequence[int],
        stride: Sequence[int] = (1, 1, 1),
    ) -> np.ndarray:
        origin3 = _as_index3(origin, "origin")
        shape3 = _as_index3(shape, "shape")
        stride3 = _as_index3(stride, "stride")
        slices = []
        for dim, (start, size, step) in enumerate(zip(origin3, shape3, stride3)):
            if step <= 0:
                raise ValueError("stride values must be positive.")
            if start < 0 or size < 0 or start + size > self.shape[dim]:
                raise ValueError(
                    f"Region {origin3} + {shape3} is outside source shape {self.shape}."
                )
            slices.append(slice(start, start + size, step))
        return np.asarray(self.array[slices[0], slices[1], slices[2]])


def open_volume_source(
    path: str | Path,
    shape: Optional[Index3] = None,
    dtype: Optional[str] = None,
) -> VolumeSource:
    volume_path = Path(path)
    suffix = volume_path.suffix.lower()

    if suffix == ".npy":
        array = np.load(volume_path, mmap_mode="r")
        _require_3d(array.shape)
        return VolumeSource(path=volume_path, array=array)

    if suffix == ".raw":
        if shape is None or dtype is None:
            raise ValueError("RAW input requires both shape and dtype.")
        dtype_obj = np.dtype(dtype)
        expected_bytes = int(np.prod(shape)) * dtype_obj.itemsize
        actual_bytes = volume_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"RAW file size mismatch: expected {expected_bytes} bytes, got {actual_bytes}."
            )
        array = np.memmap(volume_path, dtype=dtype_obj, mode="r", shape=shape)
        return VolumeSource(path=volume_path, array=array)

    if suffix in {".tif", ".tiff"}:
        array = tifffile.memmap(volume_path)
        _require_3d(array.shape)
        return VolumeSource(path=volume_path, array=array)

    raise ValueError(f"Unsupported volume format: {volume_path.suffix}")


def _as_index3(value: Sequence[int], name: str) -> Index3:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three values.")
    return tuple(int(v) for v in value)


def _require_3d(shape: Sequence[int]) -> None:
    if len(shape) != 3:
        raise ValueError("OCT volume source expects a 3D array.")
