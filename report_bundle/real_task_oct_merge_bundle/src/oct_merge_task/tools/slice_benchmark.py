from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from oct_merge_task.fusion.brick_store import DiskBackedBrickStore


def benchmark_brick_store_slices(brick_store_dir: str | Path, axis: int = 2, passes: int = 1) -> dict:
    store = DiskBackedBrickStore(brick_store_dir)
    layout = store.read_layout()
    num_slices = int(layout["output_shape"][axis])
    tasks_by_slice = _index_tasks_by_slice(layout, axis)
    per_slice_ms = []
    disk_reads = 0
    for _ in range(int(passes)):
        for index in range(num_slices):
            start = perf_counter()
            _ = _read_slice_from_bricks(store, layout, tasks_by_slice[index], axis, index)
            disk_reads += len(tasks_by_slice[index])
            per_slice_ms.append((perf_counter() - start) * 1000.0)
    mean_slice_ms = float(np.mean(per_slice_ms))
    return {
        "axis": axis,
        "passes": passes,
        "num_slices": num_slices,
        "mean_slice_ms": mean_slice_ms,
        "estimated_fps": 1000.0 / mean_slice_ms if mean_slice_ms > 0.0 else float("inf"),
        "cache_info": {"cache_size": 0, "resident_bricks": 0, "disk_reads": disk_reads},
    }


def _index_tasks_by_slice(layout: dict, axis: int) -> list[list[dict]]:
    tasks_by_slice = [[] for _ in range(int(layout["output_shape"][axis]))]
    for task in layout["tasks"]:
        origin = task["origin"]
        shape = task["shape"]
        for index in range(origin[axis], origin[axis] + shape[axis]):
            tasks_by_slice[index].append(task)
    return tasks_by_slice


def _read_slice_from_bricks(
    store: DiskBackedBrickStore,
    layout: dict,
    tasks: list[dict],
    axis: int,
    index: int,
) -> np.ndarray:
    output_shape = layout["output_shape"]
    plane_axes = [dim for dim in range(3) if dim != axis]
    plane = np.zeros((output_shape[plane_axes[0]], output_shape[plane_axes[1]]), dtype=np.float32)
    for task in tasks:
        brick_id = tuple(int(v) for v in task["brick_id"])
        origin = task["origin"]
        shape = task["shape"]
        brick = store.read_brick(brick_id)
        local_index = index - origin[axis]
        brick_slices = [slice(None), slice(None), slice(None)]
        brick_slices[axis] = local_index
        plane_slices = (
            slice(origin[plane_axes[0]], origin[plane_axes[0]] + shape[plane_axes[0]]),
            slice(origin[plane_axes[1]], origin[plane_axes[1]] + shape[plane_axes[1]]),
        )
        plane[plane_slices] = brick[tuple(brick_slices)]
    return plane
