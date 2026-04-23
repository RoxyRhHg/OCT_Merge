from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from oct_merge_task.fusion.brick_stitch import BrickStitcher
from oct_merge_task.fusion.simple_stitch import SimpleStitcher
from oct_merge_task.io.volume_store import VolumeStore
from oct_merge_task.registration.global_registrar import GlobalRegistrar
from oct_merge_task.registration.local_refiner import LocalRefiner
from oct_merge_task.tools.slice_benchmark import benchmark_brick_store_slices
from oct_merge_task.types import HalfScaleCaseConfig


class QuarterScaleCaseConfig(HalfScaleCaseConfig):
    def __init__(self) -> None:
        super().__init__(volume_shape=(750, 375, 500), overlap_voxels=75, preview_step=8, dtype="uint16", block_depth=64)


def generate_half_scale_case(output_dir: str | Path, config: HalfScaleCaseConfig | None = None) -> dict:
    config = config or HalfScaleCaseConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_a_path = output_path / "volume_a.npy"
    full_b_path = output_path / "volume_b.npy"
    _generate_full_resolution_memmaps(full_a_path, full_b_path, config)

    preview_shape = tuple(max(2, dim // config.preview_step) for dim in config.volume_shape)
    preview_a, preview_b = _generate_preview_pair(config)
    np.save(output_path / "preview_volume_a.npy", preview_a)
    np.save(output_path / "preview_volume_b.npy", preview_b)

    summary = {
        "volume_shape": list(config.volume_shape),
        "shape_a": list(config.volume_shape),
        "shape_b": list(config.volume_shape),
        "overlap_voxels": config.overlap_voxels,
        "overlap_ratio": float(config.overlap_voxels / config.volume_shape[0]),
        "preview_step": config.preview_step,
        "preview_shape": list(preview_shape),
        "preview_overlap_voxels": max(1, config.overlap_voxels // config.preview_step),
        "preview_overlap_ratio": float(max(1, config.overlap_voxels // config.preview_step) / preview_shape[0]),
        "world_shape": [config.volume_shape[0] * 2 - config.overlap_voxels, config.volume_shape[1], config.volume_shape[2]],
        "preview_world_shape": [preview_shape[0] * 2 - max(1, config.overlap_voxels // config.preview_step), preview_shape[1], preview_shape[2]],
        "same_size_pair": True,
        "rotation_deg": config.rotation_deg,
        "local_shift": config.local_shift,
        "signal_components": 6,
        "full_volume_a_path": str(full_a_path),
        "full_volume_b_path": str(full_b_path),
        "preview_volume_a_path": str(output_path / "preview_volume_a.npy"),
        "preview_volume_b_path": str(output_path / "preview_volume_b.npy"),
    }
    (output_path / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_preview_pipeline_on_half_scale_case(
    case_dir: str | Path,
    output_dir: str | Path,
    config: HalfScaleCaseConfig | None = None,
) -> dict:
    config = config or HalfScaleCaseConfig()
    case_path = Path(case_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    preview_run_dir = output_path / "preview_run"
    brick_store_dir = preview_run_dir / "stitched_bricks"

    preview_a = np.load(case_path / "preview_volume_a.npy").astype(np.float32)
    preview_b = np.load(case_path / "preview_volume_b.npy").astype(np.float32)
    preview_shape = list(preview_a.shape)
    preview_overlap = max(1, config.overlap_voxels // config.preview_step)

    level_factors = _auto_level_factors(preview_a.shape, preview_b.shape)
    store_a = VolumeStore.from_array("a", preview_a, preview_run_dir / "store_a")
    store_b = VolumeStore.from_array("b", preview_b, preview_run_dir / "store_b")
    store_a.build_pyramid(level_factors)
    store_b.build_pyramid(level_factors)

    registrar = GlobalRegistrar(search_radius=(2, 2, 1))
    estimated_transform = registrar.estimate_multiscale(
        store_a=store_a,
        store_b=store_b,
        levels=tuple(range(len(level_factors) - 1, -1, -1)),
        axis=0,
        overlap_voxels=preview_overlap,
    )
    if estimated_transform["tx"] <= 0.0:
        estimated_transform["tx"] = float(preview_overlap)
    local_field = LocalRefiner().fit(preview_a, preview_b, estimated_transform)
    stitched_result = SimpleStitcher().stitch(preview_a, preview_b, estimated_transform, local_field=local_field)
    stitched_volume = stitched_result["volume"]
    brick_result = BrickStitcher(brick_size=(8, 8, 8)).stitch_to_bricks(stitched_volume, brick_store_dir)

    preview_summary = {
        "input_shape": preview_shape,
        "stitched_shape": list(stitched_volume.shape),
        "estimated_transform": estimated_transform,
        "brick_count": brick_result["brick_count"],
    }
    benchmark_summary = benchmark_brick_store_slices(brick_store_dir=brick_store_dir, axis=2, passes=1)
    summary = {
        "preview_shape": preview_shape,
        "preview_run": preview_summary,
        "benchmark": benchmark_summary,
        "output_dir": str(output_path),
    }
    (output_path / "task_simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _generate_full_resolution_memmaps(path_a: Path, path_b: Path, config: HalfScaleCaseConfig) -> None:
    shape = tuple(config.volume_shape)
    dtype = np.uint16 if config.dtype == "uint16" else np.float32
    world_shape = (shape[0] * 2 - int(config.overlap_voxels), shape[1], shape[2])
    world = np.zeros(world_shape, dtype=np.float32)
    volume_a = open_memmap(path_a, mode="w+", dtype=dtype, shape=shape)
    volume_b = open_memmap(path_b, mode="w+", dtype=dtype, shape=shape)
    volume_a[:] = 0
    volume_b[:] = 0

    _add_layered_background(world, config, target_dtype="float32", shape_override=world_shape)
    _add_box(world, (world_shape[0] // 10, shape[1] // 8, shape[2] // 10), (world_shape[0] // 4, shape[1] // 3, shape[2] // 5), 0.75)
    _add_box(world, (world_shape[0] // 3, shape[1] // 2, shape[2] // 3), (world_shape[0] // 2, int(shape[1] * 0.78), shape[2] // 2), 1.0)
    _add_box(world, (int(world_shape[0] * 0.55), shape[1] // 5, int(shape[2] * 0.55)), (int(world_shape[0] * 0.78), int(shape[1] * 0.38), int(shape[2] * 0.82)), 0.62)
    _add_tube(world, x_start=world_shape[0] // 4, x_end=int(world_shape[0] * 0.85), y_center=shape[1] // 3, z_center=int(shape[2] * 0.65), radius=max(2, shape[1] // 20), value=0.9)
    _add_sphere_local(world, center=(world_shape[0] // 2, int(shape[1] * 0.7), int(shape[2] * 0.28)), radius=max(3, shape[1] // 12), value=0.82)
    world = _apply_attenuation_and_noise(world)

    start_b = shape[0] - int(config.overlap_voxels)
    volume_a[:] = _to_dtype(world[:shape[0]], config.dtype)
    volume_b[:] = _to_dtype(world[start_b:start_b + shape[0]], config.dtype)
    volume_a.flush()
    volume_b.flush()


def _generate_preview_pair(config: HalfScaleCaseConfig) -> tuple[np.ndarray, np.ndarray]:
    preview_shape = tuple(max(2, dim // config.preview_step) for dim in config.volume_shape)
    preview_overlap = max(1, config.overlap_voxels // config.preview_step)
    world_shape = (preview_shape[0] * 2 - preview_overlap, preview_shape[1], preview_shape[2])
    start_b = preview_shape[0] - preview_overlap
    world = np.zeros(world_shape, dtype=np.float32)

    _add_layered_background(world, config, target_dtype="float32", shape_override=world_shape)
    _add_box(world, (world_shape[0] // 8, preview_shape[1] // 6, preview_shape[2] // 6), (max(2, world_shape[0] // 3), max(2, preview_shape[1] // 2), max(2, preview_shape[2] // 3)), 0.8)
    _add_box(world, (world_shape[0] // 3, preview_shape[1] // 2, preview_shape[2] // 3), (max(2, world_shape[0] // 2), max(2, int(preview_shape[1] * 0.78)), max(2, preview_shape[2] // 2)), 1.0)
    _add_box(world, (max(1, int(world_shape[0] * 0.58)), max(1, preview_shape[1] // 5), max(1, int(preview_shape[2] * 0.55))), (max(2, int(world_shape[0] * 0.82)), max(2, int(preview_shape[1] * 0.38)), max(2, int(preview_shape[2] * 0.82))), 0.65)
    _add_tube(world, x_start=world_shape[0] // 4, x_end=max(world_shape[0] // 4 + 1, int(world_shape[0] * 0.85)), y_center=max(1, preview_shape[1] // 3), z_center=max(1, int(preview_shape[2] * 0.65)), radius=max(1, preview_shape[1] // 20), value=0.9)
    _add_sphere_local(world, center=(world_shape[0] // 2, max(1, int(preview_shape[1] * 0.7)), max(1, int(preview_shape[2] * 0.28))), radius=max(1, preview_shape[1] // 8), value=0.78)
    _add_slanted_band(world, value=0.55)
    world = _apply_attenuation_and_noise(world)

    volume_a = world[:preview_shape[0]].copy()
    volume_b = world[start_b:start_b + preview_shape[0]].copy()
    volume_b = _rotate_volume_z_local(volume_b, angle_deg=config.rotation_deg)
    volume_b = _apply_local_distortion_preview(volume_b, shift=config.local_shift)
    return volume_a, volume_b


def _add_box(volume: np.ndarray, start: tuple[int, int, int], end: tuple[int, int, int], value) -> None:
    volume[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = value


def _add_layered_background(volume: np.ndarray, config: HalfScaleCaseConfig, target_dtype: str | None = None, shape_override: tuple[int, int, int] | None = None) -> None:
    shape = shape_override or volume.shape
    dtype_name = target_dtype or config.dtype
    z = np.linspace(0.05, 0.3, shape[2], dtype=np.float32)
    layer = np.tile(z[None, None, :], (shape[0], shape[1], 1))
    if dtype_name == "uint16":
        layer = _to_dtype(layer, "uint16")
    volume[:] = np.maximum(volume, layer)


def _add_tube(volume: np.ndarray, x_start: int, x_end: int, y_center: int, z_center: int, radius: int, value) -> None:
    for x in range(x_start, x_end):
        y0 = max(0, y_center - radius)
        y1 = min(volume.shape[1], y_center + radius + 1)
        z0 = max(0, z_center - radius)
        z1 = min(volume.shape[2], z_center + radius + 1)
        yy, zz = np.meshgrid(np.arange(y0, y1), np.arange(z0, z1), indexing="ij")
        mask = (yy - y_center) ** 2 + (zz - z_center) ** 2 <= radius ** 2
        region = volume[x, y0:y1, z0:z1]
        region[mask] = np.maximum(region[mask], value)
        volume[x, y0:y1, z0:z1] = region


def _add_sphere_local(volume: np.ndarray, center: tuple[int, int, int], radius: int, value) -> None:
    x0 = max(0, center[0] - radius)
    x1 = min(volume.shape[0], center[0] + radius + 1)
    y0 = max(0, center[1] - radius)
    y1 = min(volume.shape[1], center[1] + radius + 1)
    z0 = max(0, center[2] - radius)
    z1 = min(volume.shape[2], center[2] + radius + 1)
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(x0, x1),
        np.arange(y0, y1),
        np.arange(z0, z1),
        indexing="ij",
    )
    distance = np.sqrt((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 + (grid_z - center[2]) ** 2)
    region = volume[x0:x1, y0:y1, z0:z1]
    region[distance <= radius] = np.maximum(region[distance <= radius], value)
    volume[x0:x1, y0:y1, z0:z1] = region


def _rotate_volume_z_local(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    angle = np.deg2rad(angle_deg)
    rotation = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 0.0],
            [np.sin(angle), np.cos(angle), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    grid = F.affine_grid(rotation, size=tensor.shape, align_corners=True)
    rotated = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rotated.squeeze(0).squeeze(0).numpy()


def _apply_local_distortion_preview(volume: np.ndarray, shift: int) -> np.ndarray:
    distorted = volume.copy()
    x_slice = slice(max(1, volume.shape[0] // 4), min(volume.shape[0], volume.shape[0] // 4 + volume.shape[0] // 2))
    y_slice = slice(max(1, volume.shape[1] // 3), volume.shape[1])
    if shift <= 0:
        return distorted
    source = distorted[x_slice, y_slice, :-shift].copy()
    distorted[x_slice, y_slice, shift:] = source
    return distorted


def _apply_attenuation_and_noise(volume: np.ndarray) -> np.ndarray:
    z = np.linspace(1.0, 0.55, volume.shape[2], dtype=np.float32)
    attenuated = volume * z[None, None, :]
    rng = np.random.default_rng(17)
    speckle = rng.normal(1.0, 0.08, size=volume.shape).astype(np.float32)
    noisy = attenuated * speckle
    return np.clip(noisy, 0.0, 1.0)


def _add_slanted_band(volume: np.ndarray, value: float) -> None:
    for x in range(volume.shape[0]):
        y0 = min(volume.shape[1] - 1, max(0, volume.shape[1] // 5 + x // 6))
        y1 = min(volume.shape[1], y0 + max(1, volume.shape[1] // 8))
        z0 = max(0, volume.shape[2] // 4 - x // 12)
        z1 = min(volume.shape[2], z0 + max(1, volume.shape[2] // 10))
        volume[x, y0:y1, z0:z1] = np.maximum(volume[x, y0:y1, z0:z1], value)


def _dtype_value(dtype_name: str, normalized_value: float):
    if dtype_name == "uint16":
        return np.uint16(int(np.clip(normalized_value, 0.0, 1.0) * 65535.0))
    return float(normalized_value)


def _to_dtype(volume: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "uint16":
        return np.clip(volume * 65535.0, 0.0, 65535.0).astype(np.uint16)
    if dtype_name == "float32":
        return volume.astype(np.float32)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _auto_level_factors(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    min_dim = int(min(min(shape_a), min(shape_b)))
    factors = [1]
    for candidate in (2, 4, 8):
        if min_dim // candidate >= 2:
            factors.append(candidate)
    return tuple(factors)
