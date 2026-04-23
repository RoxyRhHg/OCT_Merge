from pathlib import Path

from oct_merge_task.tools.half_scale_task import (
    HalfScaleCaseConfig,
    QuarterScaleCaseConfig,
    generate_half_scale_case,
    run_preview_pipeline_on_half_scale_case,
)


def test_generate_half_scale_case_writes_full_and_preview_outputs(tmp_path: Path) -> None:
    config = HalfScaleCaseConfig(
        volume_shape=(32, 20, 16),
        overlap_voxels=4,
        preview_step=4,
        dtype="uint16",
        block_depth=8,
    )

    summary = generate_half_scale_case(tmp_path / "case", config=config)

    assert summary["volume_shape"] == [32, 20, 16]
    assert summary["preview_shape"] == [8, 5, 4]
    assert summary["world_shape"][0] == 32 * 2 - 4
    assert summary["same_size_pair"] is True
    assert summary["shape_a"] == [32, 20, 16]
    assert summary["shape_b"] == [32, 20, 16]
    assert abs(summary["overlap_ratio"] - 0.125) < 1e-6
    assert summary["rotation_deg"] == -4.0
    assert summary["local_shift"] == 2
    assert (tmp_path / "case" / "volume_a.npy").exists()
    assert (tmp_path / "case" / "volume_b.npy").exists()
    assert (tmp_path / "case" / "preview_volume_a.npy").exists()
    assert (tmp_path / "case" / "preview_volume_b.npy").exists()
    assert summary["signal_components"] >= 4


def test_preview_pipeline_runs_from_existing_case(tmp_path: Path) -> None:
    config = HalfScaleCaseConfig(
        volume_shape=(32, 20, 16),
        overlap_voxels=4,
        preview_step=4,
        dtype="uint16",
        block_depth=8,
    )
    generate_half_scale_case(tmp_path / "case", config=config)

    summary = run_preview_pipeline_on_half_scale_case(
        case_dir=tmp_path / "case",
        output_dir=tmp_path / "preview_run",
        config=config,
    )

    assert summary["preview_shape"] == [8, 5, 4]
    assert summary["preview_run"]["stitched_shape"][0] >= 8
    assert summary["preview_run"]["estimated_transform"]["tx"] > 0.0
    assert "benchmark" in summary
    assert summary["benchmark"]["estimated_fps"] > 0.0
    assert (tmp_path / "preview_run" / "task_simulation_summary.json").exists()


def test_quarter_scale_config_matches_one_quarter_task_shape() -> None:
    config = QuarterScaleCaseConfig()

    assert config.volume_shape == (750, 375, 500)
    assert config.overlap_voxels == 75
    assert config.rotation_deg == -4.0


def test_preview_pair_uses_same_shape_and_overlap_crop_logic(tmp_path: Path) -> None:
    config = HalfScaleCaseConfig(
        volume_shape=(40, 24, 20),
        overlap_voxels=8,
        preview_step=4,
        dtype="uint16",
        block_depth=8,
    )
    summary = generate_half_scale_case(tmp_path / "case", config=config)

    assert summary["preview_shape"] == [10, 6, 5]
    assert summary["preview_world_shape"][0] == 10 * 2 - 2
    assert summary["preview_overlap_voxels"] == 2
    assert abs(summary["preview_overlap_ratio"] - 0.2) < 1e-6


def test_preview_generation_contains_richer_structure_than_single_block(tmp_path: Path) -> None:
    config = HalfScaleCaseConfig(
        volume_shape=(64, 32, 24),
        overlap_voxels=8,
        preview_step=4,
        dtype="uint16",
        block_depth=8,
    )
    generate_half_scale_case(tmp_path / "case", config=config)

    import numpy as np

    preview = np.load(tmp_path / "case" / "preview_volume_a.npy")
    nonzero_per_slice = [(preview[i] > 0).sum() for i in range(preview.shape[0])]
    mean_per_slice = [float(preview[i].mean()) for i in range(preview.shape[0])]

    assert max(nonzero_per_slice) > 0
    assert float(preview.max()) > float(preview.min())
    assert max(mean_per_slice) > min(mean_per_slice)


def test_quarter_scale_overlap_ratio_matches_original_task_fraction() -> None:
    config = QuarterScaleCaseConfig()

    assert config.volume_shape == (750, 375, 500)
    assert config.overlap_voxels == 75
    assert abs(config.overlap_voxels / config.volume_shape[0] - 0.10) < 1e-9
