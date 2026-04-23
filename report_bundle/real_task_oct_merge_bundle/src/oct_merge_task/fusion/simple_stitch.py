from __future__ import annotations

import numpy as np


class SimpleStitcher:
    def stitch(self, volume_a: np.ndarray, volume_b: np.ndarray, transform: dict, local_field=None) -> dict:
        tx = int(round(transform["tx"]))
        output_shape = (
            volume_a.shape[0] + max(0, volume_b.shape[0] - tx),
            max(volume_a.shape[1], volume_b.shape[1]),
            max(volume_a.shape[2], volume_b.shape[2]),
        )
        stitched = np.zeros(output_shape, dtype=np.float32)
        stitched[: volume_a.shape[0], : volume_a.shape[1], : volume_a.shape[2]] = volume_a
        b_start = max(0, tx)
        b_end = min(output_shape[0], tx + volume_b.shape[0])
        if b_end > b_start:
            target = stitched[b_start:b_end, : volume_b.shape[1], : volume_b.shape[2]]
            source = volume_b[: b_end - b_start].copy()
            if local_field is not None:
                source = self._apply_preview_local_field(source, b_start, b_end, local_field)
            mask = target > 0
            target[~mask] = source[~mask]
            target[mask] = 0.5 * (target[mask] + source[mask])
            stitched[b_start:b_end, : volume_b.shape[1], : volume_b.shape[2]] = target
        return {"volume": stitched}

    @staticmethod
    def _apply_preview_local_field(source: np.ndarray, b_start: int, b_end: int, local_field) -> np.ndarray:
        # Preview-only approximation: estimate one average z displacement per slice.
        adjusted = source.copy()
        for z in range(adjusted.shape[2]):
            coords = np.zeros((adjusted.shape[0], adjusted.shape[1], 3), dtype=np.float32)
            coords[..., 0] = np.arange(b_start, b_end)[:, None]
            coords[..., 1] = np.arange(adjusted.shape[1])[None, :]
            coords[..., 2] = z
            displacement = local_field.sample(coords)
            z_shift = int(round(float(displacement[..., 2].mean())))
            if z_shift != 0:
                adjusted[:, :, z] = np.roll(adjusted[:, :, z], shift=z_shift, axis=1)
        return adjusted
