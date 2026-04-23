from __future__ import annotations

import json
from pathlib import Path

from oct_merge_task.tools.slice_benchmark import benchmark_brick_store_slices


def run_benchmark_sweep(
    brick_store_dir: str | Path,
    output_path: str | Path,
    configurations: list[dict],
) -> dict:
    brick_store_dir = Path(brick_store_dir)
    output_path = Path(output_path)
    runs = []

    for config in configurations:
        axis = int(config.get("axis", 2))
        passes = int(config.get("passes", 1))
        metrics = benchmark_brick_store_slices(brick_store_dir=brick_store_dir, axis=axis, passes=passes)
        metrics["configuration"] = {"axis": axis, "passes": passes}
        runs.append(metrics)

    report = {"brick_store_dir": str(brick_store_dir), "runs": runs}
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
