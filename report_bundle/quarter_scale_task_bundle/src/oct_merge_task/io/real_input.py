from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile


def load_volume_file(
    path: str | Path,
    shape: Optional[Tuple[int, int, int]] = None,
    dtype: Optional[str] = None,
    mmap_mode: Optional[str] = None,
) -> np.ndarray:
    volume_path = Path(path)
    suffix = volume_path.suffix.lower()

    if suffix == ".npy":
        if mmap_mode is not None:
            return np.load(volume_path, mmap_mode=mmap_mode)
        return np.load(volume_path).astype(np.float32)

    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(volume_path).astype(np.float32)

    if suffix == ".raw":
        if shape is None or dtype is None:
            raise ValueError("RAW input requires both shape and dtype.")
        array = np.fromfile(volume_path, dtype=np.dtype(dtype))
        expected = int(np.prod(shape))
        if array.size != expected:
            raise ValueError(f"RAW file size mismatch: expected {expected} voxels, got {array.size}.")
        return array.reshape(shape).astype(np.float32)

    raise ValueError(f"Unsupported volume format: {volume_path.suffix}")
