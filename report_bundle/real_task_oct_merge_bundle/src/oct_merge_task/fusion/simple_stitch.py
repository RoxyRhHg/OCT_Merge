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
                for z in range(source.shape[2]):
                    coords = np.zeros((source.shape[0], source.shape[1], 3), dtype=np.float32)
                    coords[..., 0] = np.arange(b_start, b_end)[:, None]
                    coords[..., 1] = np.arange(source.shape[1])[None, :]
                    coords[..., 2] = z
                    displacement = local_field.sample(coords)
                    z_shift = int(round(float(displacement[..., 2].mean())))
                    source[:, :, z] = np.roll(source[:, :, z], shift=z_shift, axis=1)
            mask = target > 0
            target[~mask] = source[~mask]
            target[mask] = 0.5 * (target[mask] + source[mask])
            stitched[b_start:b_end, : volume_b.shape[1], : volume_b.shape[2]] = target
        return {"volume": stitched}
