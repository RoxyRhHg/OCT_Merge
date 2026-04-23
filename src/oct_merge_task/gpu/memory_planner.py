from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Index3 = Tuple[int, int, int]


@dataclass(frozen=True)
class MemoryBudget:
    max_gpu_bytes: int
    bytes_per_voxel: int = 4
    num_live_volumes: float = 2.0
    temp_buffer_factor: float = 1.0
    min_slab_depth: int = 16

    def estimate_bytes_for_shape(self, shape: Index3) -> int:
        voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
        multiplier = float(self.num_live_volumes) * float(self.temp_buffer_factor)
        return int(voxels * self.bytes_per_voxel * multiplier)

    def max_slab_shape(self, full_shape: Index3) -> Index3:
        plane_voxels = int(full_shape[1]) * int(full_shape[2])
        bytes_per_plane = plane_voxels * self.bytes_per_voxel * float(self.num_live_volumes) * float(self.temp_buffer_factor)
        if bytes_per_plane <= 0:
            raise ValueError("bytes_per_plane must be positive.")
        max_depth = int(self.max_gpu_bytes // bytes_per_plane)
        slab_depth = min(int(full_shape[0]), max(1, max(self.min_slab_depth, max_depth)))
        while slab_depth > 1 and self.estimate_bytes_for_shape((slab_depth, full_shape[1], full_shape[2])) > self.max_gpu_bytes:
            slab_depth -= 1
        return (slab_depth, int(full_shape[1]), int(full_shape[2]))
