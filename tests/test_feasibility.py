from oct_merge_task.gpu.memory_planner import MemoryBudget
from oct_merge_task.tools.feasibility import assess_single_gpu_feasibility


def test_assess_single_gpu_feasibility_reports_4090_budget_fields() -> None:
    report = assess_single_gpu_feasibility(
        shape_a=(3000, 1500, 2000),
        shape_b=(3000, 1500, 2000),
        gpu_budget_gb=48.0,
        bytes_per_voxel=4,
    )

    assert report["gpu_budget_gb"] == 48.0
    assert report["fits_full_float32_pair"] is False
    assert report["full_float32_pair_gb"] > 48.0
    assert report["streaming"]["planned_slab_shape"][1:] == [1500, 2000]
    assert report["streaming"]["planned_slab_bytes"] <= int(48.0 * (1024**3))
    assert report["streaming"]["planned_slab_gb"] < 48.0
