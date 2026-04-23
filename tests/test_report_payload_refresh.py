from pathlib import Path

import numpy as np

from oct_merge_task.bundle.report_bundle import build_report_payload_from_smoke_run
from oct_merge_task.tools.real_pipeline import run_real_data_pipeline


def test_build_report_payload_from_smoke_run_writes_payload_json(tmp_path: Path) -> None:
    smoke_data = tmp_path / "smoke_data"
    smoke_data.mkdir()
    rng = np.random.default_rng(5)
    world = rng.integers(0, 4096, size=(42, 12, 8), dtype=np.uint16)
    np.save(smoke_data / "volume_a.npy", world[:24])
    np.save(smoke_data / "volume_b.npy", world[21:42])

    smoke_run = tmp_path / "smoke_run"
    run_real_data_pipeline(
        path_a=smoke_data / "volume_a.npy",
        path_b=smoke_data / "volume_b.npy",
        output_dir=smoke_run,
        brick_size=(7, 6, 5),
        overlap_fraction_range=(0.05, 0.20),
    )

    payload = build_report_payload_from_smoke_run(smoke_run)

    assert "summary" in payload
    assert "point_clouds" in payload
    assert "slices" in payload
    assert len(payload["slices"]) == payload["summary"]["stitched_shape"][2]
    assert (smoke_run.parent / "report" / "payload.json").exists() is False
