from __future__ import annotations

import numpy as np

from oct_merge_task.io.volume_store import VolumeStore


class GlobalRegistrar:
    def __init__(self, search_radius: tuple[int, int, int] = (2, 2, 1)) -> None:
        self.search_radius = search_radius

    def estimate_multiscale(
        self,
        store_a: VolumeStore,
        store_b: VolumeStore,
        levels: tuple[int, ...],
        axis: int,
        overlap_voxels: int,
    ) -> dict:
        best_shift_full_res = (0, 0, 0)
        for level in levels:
            record = store_a.get_level_record(level)
            factor = record.factor
            volume_a = store_a.get_level_array(level)
            volume_b = store_b.get_level_array(level)
            overlap_level = max(1, overlap_voxels // factor)
            shift = self._search_translation(volume_a, volume_b, overlap_level)
            best_shift_full_res = tuple(int(v * factor) for v in shift)
        return {
            "tx": float(best_shift_full_res[0]),
            "ty": float(best_shift_full_res[1]),
            "tz": float(best_shift_full_res[2]),
            "rx": 0.0,
            "ry": 0.0,
            "rz": 0.0,
        }

    def _search_translation(self, volume_a: np.ndarray, volume_b: np.ndarray, overlap_voxels: int) -> tuple[int, int, int]:
        best_score = float("-inf")
        best_shift = (0, 0, 0)
        axis_candidates = range(-self.search_radius[0], volume_a.shape[0] - overlap_voxels + self.search_radius[0] + 1)
        for tx in axis_candidates:
            for ty in range(-self.search_radius[1], self.search_radius[1] + 1):
                for tz in range(-self.search_radius[2], self.search_radius[2] + 1):
                    score = self._score(volume_a, volume_b, (tx, ty, tz), overlap_voxels)
                    if score > best_score:
                        best_score = score
                        best_shift = (tx, ty, tz)
        return best_shift

    def _score(self, a: np.ndarray, b: np.ndarray, shift: tuple[int, int, int], overlap_voxels: int) -> float:
        a_ranges = []
        b_ranges = []
        overlap_shape = []
        for dim, delta in enumerate(shift):
            if delta >= 0:
                a_start, a_stop = delta, a.shape[dim]
                b_start, b_stop = 0, b.shape[dim] - delta
            else:
                a_start, a_stop = 0, a.shape[dim] + delta
                b_start, b_stop = -delta, b.shape[dim]
            size = min(a_stop - a_start, b_stop - b_start)
            if size <= 0:
                return float("-inf")
            overlap_shape.append(size)
            a_ranges.append((a_start, a_start + size))
            b_ranges.append((b_start, b_start + size))
        if overlap_shape[0] < overlap_voxels:
            return float("-inf")
        a_view = a[a_ranges[0][0]:a_ranges[0][1], a_ranges[1][0]:a_ranges[1][1], a_ranges[2][0]:a_ranges[2][1]]
        b_view = b[b_ranges[0][0]:b_ranges[0][1], b_ranges[1][0]:b_ranges[1][1], b_ranges[2][0]:b_ranges[2][1]]
        a_centered = a_view - float(a_view.mean())
        b_centered = b_view - float(b_view.mean())
        denom = float(np.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum()))
        if denom == 0.0:
            return float("-inf")
        return float((a_centered * b_centered).sum() / denom)
