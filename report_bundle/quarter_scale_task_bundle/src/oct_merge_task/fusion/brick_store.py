from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DiskBackedBrickStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.bricks_dir = self.root_dir / "bricks"
        self.bricks_dir.mkdir(parents=True, exist_ok=True)
        self.layout_path = self.root_dir / "layout.json"

    def write(self, layout: dict, bricks: dict[tuple[int, int, int], np.ndarray]) -> None:
        self.write_layout(layout)
        for brick_id, brick in bricks.items():
            self.write_brick(brick_id, brick)

    def write_layout(self, layout: dict) -> None:
        self.layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    def write_brick(self, brick_id: tuple[int, int, int], brick: np.ndarray) -> None:
        np.save(self.bricks_dir / self.brick_filename(brick_id), brick.astype(np.float32))

    def read_brick(self, brick_id: tuple[int, int, int]) -> np.ndarray:
        return np.load(self.bricks_dir / self.brick_filename(brick_id))

    def read_layout(self) -> dict:
        return json.loads(self.layout_path.read_text(encoding="utf-8"))

    @staticmethod
    def brick_filename(brick_id: tuple[int, int, int]) -> str:
        return f"brick_{brick_id[0]}_{brick_id[1]}_{brick_id[2]}.npy"
