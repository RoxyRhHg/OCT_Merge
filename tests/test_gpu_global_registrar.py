import numpy as np

from oct_merge_task.registration.gpu_global_registrar import GPUGlobalRegistrar


def test_gpu_global_registrar_estimates_overlap_translation_on_cpu_fallback() -> None:
    rng = np.random.default_rng(123)
    world = rng.normal(size=(40, 10, 8)).astype(np.float32)
    volume_a = world[:24]
    volume_b = world[21:40]

    registrar = GPUGlobalRegistrar(device="cpu")
    result = registrar.estimate_translation(volume_a, volume_b, overlap_voxels=3)

    assert result["tx"] == 21
    assert abs(result["ty"]) <= 1
    assert abs(result["tz"]) <= 1
    assert result["score"] > 0.99


def test_gpu_global_registrar_reports_torch_fft_mode_when_requested() -> None:
    rng = np.random.default_rng(321)
    world = rng.normal(size=(36, 8, 6)).astype(np.float32)
    volume_a = world[:20]
    volume_b = world[17:36]

    registrar = GPUGlobalRegistrar(device="cuda")
    result = registrar.estimate_translation(volume_a, volume_b, overlap_voxels=3)

    assert result["tx"] == 17
    assert result["mode"] in {"torch-fft", "cpu-fallback"}
