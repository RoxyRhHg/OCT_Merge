from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HalfScaleCaseConfig:
    volume_shape: tuple[int, int, int] = (1500, 750, 1000)
    overlap_voxels: int = 150
    preview_step: int = 8
    dtype: str = "uint16"
    block_depth: int = 64
    rotation_deg: float = -4.0
    local_shift: int = 2
