import numpy as np

from oct_merge_task.registration.local_refiner import LocalRefiner
from oct_merge_task.registration.similarity import normalized_cross_correlation


def test_local_refiner_improves_local_alignment_score() -> None:
    reference = np.zeros((24, 16, 16), dtype=np.float32)
    moving = np.zeros_like(reference)
    reference[6:18, 4:10, 4:12] = 1.0
    moving[6:18, 4:10, 3:11] = 1.0

    pre = normalized_cross_correlation(reference, moving)
    field = LocalRefiner().fit(reference, moving, {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0})
    coords = np.zeros((moving.shape[0], moving.shape[1], moving.shape[2], 3), dtype=np.float32)
    coords[..., 0] = np.arange(moving.shape[0])[:, None, None]
    coords[..., 1] = np.arange(moving.shape[1])[None, :, None]
    coords[..., 2] = np.arange(moving.shape[2])[None, None, :]
    displacement = field.sample(coords)
    roi_shift = float(displacement[8:16, 5:9, 4:12, 2].mean())

    assert pre < 1.0
    assert abs(roi_shift) > 0.1
