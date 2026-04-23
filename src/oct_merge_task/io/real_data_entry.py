from __future__ import annotations

from pathlib import Path


def discover_real_data_pair(real_data_dir: str | Path) -> dict:
    real_data_dir = Path(real_data_dir)
    candidates = [
        ("npy", real_data_dir / "volume_a.npy", real_data_dir / "volume_b.npy"),
        ("tiff", real_data_dir / "volume_a.tiff", real_data_dir / "volume_b.tiff"),
        ("tiff", real_data_dir / "volume_a.tif", real_data_dir / "volume_b.tif"),
        ("raw", real_data_dir / "volume_a.raw", real_data_dir / "volume_b.raw"),
    ]
    for fmt, path_a, path_b in candidates:
        if path_a.exists() and path_b.exists():
            return {"format": fmt, "volume_a": path_a, "volume_b": path_b}
    raise FileNotFoundError(f"No supported A/B pair was found in {real_data_dir}")
