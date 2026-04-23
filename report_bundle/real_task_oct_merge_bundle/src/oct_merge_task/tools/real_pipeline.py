from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from oct_merge_task.fusion.brick_store import DiskBackedBrickStore
from oct_merge_task.gpu.memory_planner import MemoryBudget
from oct_merge_task.io.volume_source import VolumeSource, open_volume_source
from oct_merge_task.registration.gpu_global_registrar import GPUGlobalRegistrar
from oct_merge_task.registration.similarity import normalized_cross_correlation
from oct_merge_task.tools.slice_benchmark import benchmark_brick_store_slices


Index3 = Tuple[int, int, int]


def estimate_axis_overlap(
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    overlap_fraction_range: Tuple[float, float] = (0.05, 0.20),
    axis: int = 0,
) -> dict:
    if axis != 0:
        raise ValueError("Only axis=0 overlap estimation is currently supported.")
    a = np.asarray(volume_a, dtype=np.float32)
    b = np.asarray(volume_b, dtype=np.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Overlap estimation expects 3D volumes.")
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("A/B volumes must match on non-stitched axes.")

    min_fraction, max_fraction = overlap_fraction_range
    if min_fraction <= 0.0 or max_fraction <= 0.0 or min_fraction > max_fraction:
        raise ValueError("overlap_fraction_range must be positive and ordered.")

    min_overlap = max(1, int(round(min(a.shape[0], b.shape[0]) * min_fraction)))
    max_overlap = min(a.shape[0], b.shape[0], int(round(min(a.shape[0], b.shape[0]) * max_fraction)))
    best = {"overlap_voxels": min_overlap, "tx": a.shape[0] - min_overlap, "score": float("-inf")}
    for overlap in range(min_overlap, max_overlap + 1):
        a_view = a[a.shape[0] - overlap : a.shape[0]]
        b_view = b[:overlap]
        score = normalized_cross_correlation(a_view, b_view)
        if score > best["score"]:
            best = {"overlap_voxels": int(overlap), "tx": int(a.shape[0] - overlap), "score": float(score)}
    return best


def run_real_data_pipeline(
    path_a: str | Path,
    path_b: str | Path,
    output_dir: str | Path,
    shape_a: Optional[Index3] = None,
    shape_b: Optional[Index3] = None,
    dtype_a: Optional[str] = None,
    dtype_b: Optional[str] = None,
    brick_size: Index3 = (128, 128, 128),
    overlap_fraction_range: Tuple[float, float] = (0.05, 0.20),
    preview_stride: Index3 = (8, 8, 8),
    max_gpu_bytes: int = 40 * (1024**3),
    registration_device: str = "cpu",
    fusion_device: str = "cpu",
) -> dict:
    source_a = open_volume_source(path_a, shape=shape_a, dtype=dtype_a)
    source_b = open_volume_source(path_b, shape=shape_b, dtype=dtype_b)
    _validate_sources(source_a, source_b)
    memory_budget = MemoryBudget(
        max_gpu_bytes=max_gpu_bytes,
        bytes_per_voxel=max(source_a.dtype.itemsize, source_b.dtype.itemsize, 4),
        num_live_volumes=2.0,
        temp_buffer_factor=1.5,
    )
    slab_shape = memory_budget.max_slab_shape(source_a.shape)

    effective_preview_stride = _effective_preview_stride(source_a, source_b, preview_stride)
    preview_a = _read_preview(source_a, effective_preview_stride)
    preview_b = _read_preview(source_b, effective_preview_stride)
    preview_estimate = estimate_axis_overlap(
        preview_a,
        preview_b,
        overlap_fraction_range=overlap_fraction_range,
        axis=0,
    )
    estimated_overlap = max(1, int(round(preview_estimate["overlap_voxels"] * effective_preview_stride[0])))
    estimated_overlap = min(estimated_overlap, source_a.shape[0], source_b.shape[0])
    registration = GPUGlobalRegistrar(device=registration_device).estimate_translation(
        preview_a,
        preview_b,
        overlap_voxels=max(1, preview_estimate["overlap_voxels"]),
    )
    transform = {
        "tx": int(source_a.shape[0] - estimated_overlap),
        "ty": 0,
        "tz": 0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "score": float(registration["score"]),
    }

    output_path = Path(output_dir)
    brick_store_dir = output_path / "stitched_bricks"
    stitcher = StreamingBrickStitcher(brick_size=brick_size, device=fusion_device)
    brick_result = stitcher.stitch_to_bricks(
        source_a=source_a,
        source_b=source_b,
        tx=int(transform["tx"]),
        output_dir=brick_store_dir,
    )
    benchmark = benchmark_brick_store_slices(brick_store_dir=brick_store_dir, axis=2, passes=1)
    summary = {
        "shape_a": list(source_a.shape),
        "shape_b": list(source_b.shape),
        "dtype_a": str(source_a.dtype),
        "dtype_b": str(source_b.dtype),
        "estimated_overlap_voxels": int(estimated_overlap),
        "estimated_overlap_fraction": float(estimated_overlap / source_a.shape[0]),
        "overlap_fraction_range": list(overlap_fraction_range),
        "estimated_transform": transform,
        "stitched_shape": brick_result["layout"]["output_shape"],
        "brick_count": brick_result["brick_count"],
        "brick_size": list(brick_size),
        "memory_policy": {
            "loads_full_volume": False,
            "source_access": "memmap",
            "fusion": "streaming_bricks",
            "preview_stride": list(effective_preview_stride),
        },
        "memory_budget": {
            "max_gpu_bytes": int(memory_budget.max_gpu_bytes),
            "planned_slab_shape": list(slab_shape),
            "planned_slab_bytes": int(memory_budget.estimate_bytes_for_shape(slab_shape)),
        },
        "registration": {
            "mode": registration["mode"],
            "axis": registration["axis"],
            "preview_overlap_voxels": int(preview_estimate["overlap_voxels"]),
            "preview_score": float(preview_estimate["score"]),
        },
        "fusion": {
            "mode": stitcher.fusion_mode,
            "axis": 0,
        },
        "benchmark": benchmark,
        "output_dir": str(output_path),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "real_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


class StreamingBrickStitcher:
    def __init__(self, brick_size: Index3 = (128, 128, 128), device: str = "cpu") -> None:
        self.brick_size = tuple(int(v) for v in brick_size)
        self.device = device
        self.fusion_mode = "numpy-fallback"

    def stitch_to_bricks(
        self,
        source_a: VolumeSource,
        source_b: VolumeSource,
        tx: int,
        output_dir: str | Path,
    ) -> dict:
        output_shape = (
            max(source_a.shape[0], tx + source_b.shape[0]),
            max(source_a.shape[1], source_b.shape[1]),
            max(source_a.shape[2], source_b.shape[2]),
        )
        tasks = _build_tasks(output_shape, self.brick_size)
        store = DiskBackedBrickStore(output_dir)
        store.write_layout({"output_shape": list(output_shape), "brick_size": list(self.brick_size), "tasks": tasks})
        for task in tasks:
            brick = self._render_brick(source_a, source_b, tx, task)
            store.write_brick(tuple(task["brick_id"]), brick)
        return {"layout": store.read_layout(), "brick_count": len(tasks)}

    def _render_brick(
        self,
        source_a: VolumeSource,
        source_b: VolumeSource,
        tx: int,
        task: dict,
    ) -> np.ndarray:
        origin = tuple(int(v) for v in task["origin"])
        shape = tuple(int(v) for v in task["shape"])
        accum = np.zeros(shape, dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)

        region_a, mask_a = _read_source_region(source_a, origin, shape, source_offset=(0, 0, 0))
        region_b, mask_b = _read_source_region(source_b, origin, shape, source_offset=(tx, 0, 0))
        return self._fuse_regions(region_a, mask_a, region_b, mask_b)

    def _fuse_regions(
        self,
        region_a: np.ndarray,
        mask_a: np.ndarray,
        region_b: np.ndarray,
        mask_b: np.ndarray,
    ) -> np.ndarray:
        try:
            import torch

            target_device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
            tensor_a = torch.as_tensor(region_a, dtype=torch.float32, device=target_device)
            tensor_b = torch.as_tensor(region_b, dtype=torch.float32, device=target_device)
            tensor_mask_a = torch.as_tensor(mask_a, dtype=torch.float32, device=target_device)
            tensor_mask_b = torch.as_tensor(mask_b, dtype=torch.float32, device=target_device)
            accum = tensor_a * tensor_mask_a + tensor_b * tensor_mask_b
            weight = tensor_mask_a + tensor_mask_b
            brick = torch.where(weight > 0.0, accum / torch.clamp(weight, min=1.0), torch.zeros_like(accum))
            self.fusion_mode = "torch-cuda" if target_device == "cuda" else "torch-cpu"
            return brick.detach().cpu().numpy().astype(np.float32)
        except Exception:
            accum = region_a * mask_a.astype(np.float32) + region_b * mask_b.astype(np.float32)
            weight = mask_a.astype(np.float32) + mask_b.astype(np.float32)
            brick = np.zeros(region_a.shape, dtype=np.float32)
            valid = weight > 0.0
            brick[valid] = accum[valid] / weight[valid]
            self.fusion_mode = "numpy-fallback"
            return brick


def _validate_sources(source_a: VolumeSource, source_b: VolumeSource) -> None:
    if len(source_a.shape) != 3 or len(source_b.shape) != 3:
        raise ValueError("A/B inputs must both be 3D volumes.")
    if source_a.shape[1:] != source_b.shape[1:]:
        raise ValueError("A/B inputs must match on y/z dimensions for the streaming pipeline.")


def _read_preview(source: VolumeSource, stride: Index3) -> np.ndarray:
    preview_shape = tuple((dim + step - 1) // step for dim, step in zip(source.shape, stride))
    shape = tuple((size - 1) * step + 1 for size, step in zip(preview_shape, stride))
    return source.read_region((0, 0, 0), shape, stride=stride).astype(np.float32)


def _effective_preview_stride(
    source_a: VolumeSource,
    source_b: VolumeSource,
    requested_stride: Index3,
) -> Index3:
    effective = []
    for dim, requested in enumerate(requested_stride):
        min_dim = min(source_a.shape[dim], source_b.shape[dim])
        cap = max(1, min_dim // 16)
        effective.append(max(1, min(int(requested), cap)))
    return tuple(effective)


def _build_tasks(output_shape: Index3, brick_size: Index3) -> list[dict]:
    tasks = []
    nx = int(np.ceil(output_shape[0] / brick_size[0]))
    ny = int(np.ceil(output_shape[1] / brick_size[1]))
    nz = int(np.ceil(output_shape[2] / brick_size[2]))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = (ix * brick_size[0], iy * brick_size[1], iz * brick_size[2])
                shape = (
                    min(brick_size[0], output_shape[0] - origin[0]),
                    min(brick_size[1], output_shape[1] - origin[1]),
                    min(brick_size[2], output_shape[2] - origin[2]),
                )
                tasks.append({"brick_id": [ix, iy, iz], "origin": list(origin), "shape": list(shape)})
    return tasks


def _read_source_region(
    source: VolumeSource,
    output_origin: Index3,
    output_shape: Index3,
    source_offset: Index3,
) -> tuple[np.ndarray, np.ndarray]:
    region = np.zeros(output_shape, dtype=np.float32)
    mask = np.zeros(output_shape, dtype=bool)
    output_start = np.array(output_origin, dtype=np.int64)
    output_stop = output_start + np.array(output_shape, dtype=np.int64)
    source_start = np.array(source_offset, dtype=np.int64)
    source_stop = source_start + np.array(source.shape, dtype=np.int64)

    inter_start = np.maximum(output_start, source_start)
    inter_stop = np.minimum(output_stop, source_stop)
    if np.any(inter_stop <= inter_start):
        return region, mask

    out_local_start = inter_start - output_start
    out_local_stop = inter_stop - output_start
    src_local_start = inter_start - source_start
    region_shape = inter_stop - inter_start
    data = source.read_region(
        tuple(int(v) for v in src_local_start),
        tuple(int(v) for v in region_shape),
    ).astype(np.float32)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(out_local_start, out_local_stop))
    region[slices] = data
    mask[slices] = True
    return region, mask
