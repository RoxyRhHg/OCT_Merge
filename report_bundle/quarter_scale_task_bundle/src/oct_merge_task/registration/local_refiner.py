from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oct_merge_task.registration.similarity import normalized_cross_correlation


@dataclass(frozen=True)
class LocalDisplacementField:
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    displacements: np.ndarray

    def sample(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        grid_shape = self.displacements.shape[:3]
        origin = np.asarray(self.origin, dtype=np.float32)
        spacing = np.asarray(self.spacing, dtype=np.float32)
        local = (coords - origin) / spacing

        result = np.zeros(coords.shape, dtype=np.float32)
        valid = np.ones(coords.shape[:-1], dtype=bool)
        for dim in range(3):
            valid &= local[..., dim] >= 0.0
            valid &= local[..., dim] <= float(grid_shape[dim] - 1)
        if not np.any(valid):
            return result

        valid_local = local[valid]
        i0 = np.floor(valid_local).astype(np.int32)
        i1 = np.minimum(i0 + 1, np.array(grid_shape, dtype=np.int32) - 1)
        frac = valid_local - i0.astype(np.float32)

        c000 = self.displacements[i0[:, 0], i0[:, 1], i0[:, 2]]
        c100 = self.displacements[i1[:, 0], i0[:, 1], i0[:, 2]]
        c010 = self.displacements[i0[:, 0], i1[:, 1], i0[:, 2]]
        c001 = self.displacements[i0[:, 0], i0[:, 1], i1[:, 2]]
        c110 = self.displacements[i1[:, 0], i1[:, 1], i0[:, 2]]
        c101 = self.displacements[i1[:, 0], i0[:, 1], i1[:, 2]]
        c011 = self.displacements[i0[:, 0], i1[:, 1], i1[:, 2]]
        c111 = self.displacements[i1[:, 0], i1[:, 1], i1[:, 2]]

        fx = frac[:, 0:1]
        fy = frac[:, 1:2]
        fz = frac[:, 2:3]
        c00 = c000 * (1.0 - fx) + c100 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        sampled = c0 * (1.0 - fz) + c1 * fz
        result[valid] = sampled
        return result


class LocalRefiner:
    def __init__(
        self,
        control_point_spacing: tuple[int, int, int] = (8, 8, 8),
        patch_radius: tuple[int, int, int] = (2, 2, 2),
        search_radius: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        self.control_point_spacing = control_point_spacing
        self.patch_radius = patch_radius
        self.search_radius = search_radius

    def fit(self, reference_volume: np.ndarray, moving_volume: np.ndarray, global_transform: dict) -> LocalDisplacementField:
        del global_transform
        reference = np.asarray(reference_volume, dtype=np.float32)
        moving = np.asarray(moving_volume, dtype=np.float32)
        control_axes = []
        for dim in range(3):
            points = list(range(0, reference.shape[dim], self.control_point_spacing[dim]))
            if not points or points[-1] != reference.shape[dim] - 1:
                points.append(reference.shape[dim] - 1)
            control_axes.append(points)

        displacements = np.zeros((len(control_axes[0]), len(control_axes[1]), len(control_axes[2]), 3), dtype=np.float32)
        for ix, x in enumerate(control_axes[0]):
            for iy, y in enumerate(control_axes[1]):
                for iz, z in enumerate(control_axes[2]):
                    center = np.array([x, y, z], dtype=np.int32)
                    displacements[ix, iy, iz] = -self._best_patch_shift(reference, moving, center).astype(np.float32)

        return LocalDisplacementField(
            origin=(float(control_axes[0][0]), float(control_axes[1][0]), float(control_axes[2][0])),
            spacing=tuple(float(v) for v in self.control_point_spacing),
            displacements=displacements,
        )

    def _best_patch_shift(self, reference: np.ndarray, moving: np.ndarray, center: np.ndarray) -> np.ndarray:
        ref_patch = self._extract_patch(reference, center, np.zeros(3, dtype=np.int32))
        if ref_patch is None or float(ref_patch.max()) == 0.0:
            return np.zeros(3, dtype=np.int32)

        best_score = float("-inf")
        best_delta = np.zeros(3, dtype=np.int32)
        for dx in range(-self.search_radius[0], self.search_radius[0] + 1):
            for dy in range(-self.search_radius[1], self.search_radius[1] + 1):
                for dz in range(-self.search_radius[2], self.search_radius[2] + 1):
                    delta = np.array([dx, dy, dz], dtype=np.int32)
                    mov_patch = self._extract_patch(moving, center, delta)
                    if mov_patch is None:
                        continue
                    score = normalized_cross_correlation(ref_patch, mov_patch)
                    if score > best_score:
                        best_score = score
                        best_delta = delta
        return best_delta

    def _extract_patch(self, volume: np.ndarray, center: np.ndarray, delta: np.ndarray) -> np.ndarray | None:
        slices = []
        for dim in range(3):
            start = int(center[dim] + delta[dim] - self.patch_radius[dim])
            end = int(center[dim] + delta[dim] + self.patch_radius[dim] + 1)
            if start < 0 or end > volume.shape[dim]:
                return None
            slices.append(slice(start, end))
        return volume[slices[0], slices[1], slices[2]]
