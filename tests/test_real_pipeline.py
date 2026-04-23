from pathlib import Path

import numpy as np

from oct_merge_task.tools.real_pipeline import estimate_axis_overlap, run_real_data_pipeline


def test_estimate_axis_overlap_finds_unknown_overlap_near_ten_percent() -> None:
    rng = np.random.default_rng(42)
    world = rng.normal(size=(38, 8, 6)).astype(np.float32)
    volume_a = world[:20]
    volume_b = world[16:36]

    result = estimate_axis_overlap(
        volume_a,
        volume_b,
        overlap_fraction_range=(0.05, 0.25),
        axis=0,
    )

    assert result["overlap_voxels"] == 4
    assert result["tx"] == 16
    assert result["score"] > 0.99


def test_run_real_data_pipeline_streams_real_inputs_to_bricks(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    world = rng.integers(0, 4096, size=(42, 12, 8), dtype=np.uint16)
    volume_a = world[:24]
    volume_b = world[21:42]
    path_a = tmp_path / "volume_a.npy"
    path_b = tmp_path / "volume_b.npy"
    np.save(path_a, volume_a)
    np.save(path_b, volume_b)

    summary = run_real_data_pipeline(
        path_a=path_a,
        path_b=path_b,
        output_dir=tmp_path / "run",
        brick_size=(7, 6, 5),
        overlap_fraction_range=(0.05, 0.20),
    )

    assert summary["estimated_overlap_voxels"] == 3
    assert summary["estimated_transform"]["tx"] == 21
    assert summary["stitched_shape"] == [42, 12, 8]
    assert summary["brick_count"] > 0
    assert summary["memory_policy"]["loads_full_volume"] is False
    assert (tmp_path / "run" / "stitched_bricks" / "layout.json").exists()
    assert summary["benchmark"]["cache_info"]["disk_reads"] > 0
