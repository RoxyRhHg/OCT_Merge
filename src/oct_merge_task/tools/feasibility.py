from __future__ import annotations

from typing import Tuple

from oct_merge_task.gpu.memory_planner import MemoryBudget


Index3 = Tuple[int, int, int]


def assess_single_gpu_feasibility(
    shape_a: Index3,
    shape_b: Index3,
    gpu_budget_gb: float = 24.0,
    bytes_per_voxel: int = 4,
) -> dict:
    budget_bytes = int(gpu_budget_gb * (1024**3))
    full_pair_bytes = int(shape_a[0] * shape_a[1] * shape_a[2] * bytes_per_voxel) + int(
        shape_b[0] * shape_b[1] * shape_b[2] * bytes_per_voxel
    )
    planner = MemoryBudget(
        max_gpu_bytes=budget_bytes,
        bytes_per_voxel=bytes_per_voxel,
        num_live_volumes=2.0,
        temp_buffer_factor=1.5,
    )
    reference_shape = shape_a if shape_a[0] >= shape_b[0] else shape_b
    slab_shape = planner.max_slab_shape(reference_shape)
    slab_bytes = planner.estimate_bytes_for_shape(slab_shape)
    return {
        "gpu_budget_gb": float(gpu_budget_gb),
        "bytes_per_voxel": int(bytes_per_voxel),
        "shape_a": list(shape_a),
        "shape_b": list(shape_b),
        "full_float32_pair_bytes": int(full_pair_bytes),
        "full_float32_pair_gb": float(full_pair_bytes / (1024**3)),
        "fits_full_float32_pair": full_pair_bytes <= budget_bytes,
        "streaming": {
            "planned_slab_shape": list(slab_shape),
            "planned_slab_bytes": int(slab_bytes),
            "planned_slab_gb": float(slab_bytes / (1024**3)),
            "fits_budget": slab_bytes <= budget_bytes,
            "remaining_budget_gb": float((budget_bytes - slab_bytes) / (1024**3)),
        },
    }
