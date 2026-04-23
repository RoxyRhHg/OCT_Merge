from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PyramidLevel:
    level: int
    factor: int
    shape: tuple[int, int, int]
    path: Path


@dataclass
class VolumeStore:
    name: str
    root_dir: Path
    _base_array: np.ndarray
    _level_arrays: dict[int, np.ndarray]
    _level_records: dict[int, PyramidLevel]

    @classmethod
    def from_array(cls, name: str, array: np.ndarray, root_dir: str | Path) -> "VolumeStore":
        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(array)
        if arr.ndim != 3:
            raise ValueError("VolumeStore expects a 3D array.")
        return cls(name=name, root_dir=root, _base_array=arr.astype(np.float32, copy=True), _level_arrays={}, _level_records={})

    def build_pyramid(self, level_factors: tuple[int, ...]) -> list[PyramidLevel]:
        levels = []
        for level, factor in enumerate(level_factors):
            level_array = self._base_array if factor == 1 else self._downsample(self._base_array, factor)
            record = PyramidLevel(level=level, factor=factor, shape=tuple(int(v) for v in level_array.shape), path=self.root_dir / f"level_{level}.npy")
            self._level_arrays[level] = level_array
            self._level_records[level] = record
            levels.append(record)
        return levels

    def get_level_array(self, level: int) -> np.ndarray:
        return self._level_arrays[level]

    def get_level_record(self, level: int) -> PyramidLevel:
        return self._level_records[level]

    @staticmethod
    def _downsample(array: np.ndarray, factor: int) -> np.ndarray:
        cropped_shape = tuple((dim // factor) * factor for dim in array.shape)
        cropped = array[: cropped_shape[0], : cropped_shape[1], : cropped_shape[2]]
        reshaped = cropped.reshape(
            cropped_shape[0] // factor,
            factor,
            cropped_shape[1] // factor,
            factor,
            cropped_shape[2] // factor,
            factor,
        )
        return reshaped.mean(axis=(1, 3, 5)).astype(np.float32)
