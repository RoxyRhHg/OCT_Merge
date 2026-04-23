from pathlib import Path

import numpy as np
import tifffile

from oct_merge_task.io.real_data_entry import discover_real_data_pair


def test_discover_real_data_pair_finds_npy_pair(tmp_path: Path) -> None:
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    np.save(real_data / "volume_a.npy", np.zeros((8, 6, 4), dtype=np.float32))
    np.save(real_data / "volume_b.npy", np.zeros((8, 6, 4), dtype=np.float32))

    pair = discover_real_data_pair(real_data)

    assert pair["format"] == "npy"
    assert pair["volume_a"].name == "volume_a.npy"
    assert pair["volume_b"].name == "volume_b.npy"


def test_discover_real_data_pair_finds_tiff_pair(tmp_path: Path) -> None:
    real_data = tmp_path / "real_data"
    real_data.mkdir()
    tifffile.imwrite(real_data / "volume_a.tiff", np.zeros((8, 6, 4), dtype=np.uint16))
    tifffile.imwrite(real_data / "volume_b.tiff", np.zeros((8, 6, 4), dtype=np.uint16))

    pair = discover_real_data_pair(real_data)

    assert pair["format"] == "tiff"
    assert pair["volume_a"].name == "volume_a.tiff"
    assert pair["volume_b"].name == "volume_b.tiff"
