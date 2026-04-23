from pathlib import Path

from oct_merge_task.tools.benchmark_sweep import run_benchmark_sweep
from oct_merge_task.tools.half_scale_task import HalfScaleCaseConfig, generate_half_scale_case, run_preview_pipeline_on_half_scale_case


def test_benchmark_sweep_runs_multiple_configurations(tmp_path: Path) -> None:
    config = HalfScaleCaseConfig(
        volume_shape=(32, 20, 16),
        overlap_voxels=4,
        preview_step=4,
        dtype="uint16",
        block_depth=8,
    )
    generate_half_scale_case(tmp_path / "case", config=config)
    preview_summary = run_preview_pipeline_on_half_scale_case(
        case_dir=tmp_path / "case",
        output_dir=tmp_path / "preview_run",
        config=config,
    )

    report = run_benchmark_sweep(
        brick_store_dir=Path(preview_summary["output_dir"]) / "preview_run" / "stitched_bricks",
        output_path=tmp_path / "benchmark_sweep.json",
        configurations=[
            {"axis": 2, "passes": 1},
            {"axis": 1, "passes": 1},
        ],
    )

    assert len(report["runs"]) == 2
    assert report["runs"][0]["estimated_fps"] > 0.0
    assert (tmp_path / "benchmark_sweep.json").exists()
