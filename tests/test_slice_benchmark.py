from pathlib import Path

import numpy as np

from oct_merge_task.fusion.brick_stitch import BrickStitcher
from oct_merge_task.tools.slice_benchmark import benchmark_brick_store_slices


def test_slice_benchmark_reads_real_bricks_and_reports_metrics(tmp_path: Path) -> None:
    volume = np.zeros((20, 14, 12), dtype=np.float32)
    volume[5:16, 4:10, 3:9] = 1.0
    result = BrickStitcher(brick_size=(7, 6, 5)).stitch_to_bricks(volume, tmp_path / "stitched_bricks")

    metrics = benchmark_brick_store_slices(tmp_path / "stitched_bricks", axis=2, passes=1)

    assert result["brick_count"] > 0
    assert metrics["mean_slice_ms"] > 0.0
    assert metrics["estimated_fps"] > 0.0
