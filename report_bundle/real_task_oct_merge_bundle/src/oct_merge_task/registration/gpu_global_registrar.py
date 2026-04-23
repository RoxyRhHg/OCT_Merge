from __future__ import annotations

from typing import Any, Dict

import numpy as np


class GPUGlobalRegistrar:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def estimate_translation(
        self,
        volume_a: np.ndarray,
        volume_b: np.ndarray,
        overlap_voxels: int,
    ) -> Dict[str, Any]:
        a = np.asarray(volume_a, dtype=np.float32)
        b = np.asarray(volume_b, dtype=np.float32)
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("GPUGlobalRegistrar expects 3D inputs.")
        if overlap_voxels <= 0:
            raise ValueError("overlap_voxels must be positive.")
        overlap_voxels = min(int(overlap_voxels), a.shape[0], b.shape[0])

        overlap_a = a[a.shape[0] - overlap_voxels : a.shape[0]]
        overlap_b = b[:overlap_voxels]
        score, mode = self._phase_correlate_score(overlap_a, overlap_b)
        return {
            "tx": int(a.shape[0] - overlap_voxels),
            "ty": 0,
            "tz": 0,
            "score": float(score),
            "mode": mode,
            "axis": 0,
        }

    @staticmethod
    def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
        a_centered = a - float(a.mean())
        b_centered = b - float(b.mean())
        denom = float(np.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum()))
        if denom == 0.0:
            return float("-inf")
        return float((a_centered * b_centered).sum() / denom)

    def _phase_correlate_score(self, a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
        if self.device != "cuda":
            return self._normalized_cross_correlation(a, b), "cpu-fallback"
        try:
            import torch

            tensor_a = torch.as_tensor(a, dtype=torch.float32)
            tensor_b = torch.as_tensor(b, dtype=torch.float32)
            fa = torch.fft.rfftn(tensor_a)
            fb = torch.fft.rfftn(tensor_b)
            cross = fa * torch.conj(fb)
            denom = torch.abs(cross).clamp_min(1e-8)
            corr = torch.fft.irfftn(cross / denom, s=tensor_a.shape)
            peak_value = float(torch.max(corr).item())
            return peak_value, "torch-fft"
        except Exception:
            return self._normalized_cross_correlation(a, b), "cpu-fallback"
