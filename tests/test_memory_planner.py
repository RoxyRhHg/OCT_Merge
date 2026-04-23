from oct_merge_task.gpu.memory_planner import MemoryBudget


def test_memory_budget_plans_safe_slab_shape_under_vram_limit() -> None:
    planner = MemoryBudget(max_gpu_bytes=40 * (1024**3), bytes_per_voxel=4, num_live_volumes=2, temp_buffer_factor=1.5)

    slab_shape = planner.max_slab_shape((3000, 1500, 2000))

    assert slab_shape[1:] == (1500, 2000)
    assert slab_shape[0] > 0
    assert planner.estimate_bytes_for_shape(slab_shape) <= planner.max_gpu_bytes
