from __future__ import annotations

import numpy as np

from oct_merge_task.fusion.brick_store import DiskBackedBrickStore


class BrickStitcher:
    def __init__(self, brick_size: tuple[int, int, int] = (8, 8, 8)) -> None:
        self.brick_size = brick_size

    def stitch_to_bricks(self, stitched_volume: np.ndarray, output_dir: str | None = None) -> dict:
        shape = stitched_volume.shape
        tasks = []
        bricks = {}
        nx = int(np.ceil(shape[0] / self.brick_size[0]))
        ny = int(np.ceil(shape[1] / self.brick_size[1]))
        nz = int(np.ceil(shape[2] / self.brick_size[2]))
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x0, y0, z0 = ix * self.brick_size[0], iy * self.brick_size[1], iz * self.brick_size[2]
                    sx = min(self.brick_size[0], shape[0] - x0)
                    sy = min(self.brick_size[1], shape[1] - y0)
                    sz = min(self.brick_size[2], shape[2] - z0)
                    tasks.append({"brick_id": [ix, iy, iz], "origin": [x0, y0, z0], "shape": [sx, sy, sz]})
                    bricks[(ix, iy, iz)] = stitched_volume[x0:x0+sx, y0:y0+sy, z0:z0+sz]
        layout = {"output_shape": list(shape), "brick_size": list(self.brick_size), "tasks": tasks}
        if output_dir is not None:
            DiskBackedBrickStore(output_dir).write(layout, bricks)
        return {"layout": layout, "bricks": bricks, "brick_count": len(bricks)}
