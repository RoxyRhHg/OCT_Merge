from pathlib import Path

from oct_merge_task.tools.half_scale_task import HalfScaleCaseConfig, generate_half_scale_case, run_preview_pipeline_on_half_scale_case
from oct_merge_task.web.payload import build_preview_web_payload


def test_build_preview_web_payload_writes_browser_ready_json(tmp_path: Path) -> None:
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

    payload = build_preview_web_payload(
        preview_run_dir=Path(preview_summary["output_dir"]),
        output_dir=tmp_path / "web_payload",
    )

    assert payload["stitched_shape"][0] >= config.volume_shape[0] // config.preview_step
    assert "transform" in payload
    assert "slices" in payload
    assert "point_clouds" in payload
    assert "volume_a" in payload["point_clouds"]
    assert "volume_b" in payload["point_clouds"]
    assert "stitched" in payload["point_clouds"]
    assert payload["same_size_pair"] is True
    assert abs(payload["overlap_ratio"] - 0.125) < 1e-6
    assert payload["point_clouds"]["volume_a"]["count"] > 0
    assert len(payload["point_clouds"]["volume_a"]["legend"]) >= 4
    stitched_labels = {item["label"] for item in payload["point_clouds"]["stitched"]["legend"]}
    assert "A-only" in stitched_labels
    assert "B-only" in stitched_labels
    assert "Overlap" in stitched_labels
    assert (tmp_path / "web_payload" / "payload.json").exists()
