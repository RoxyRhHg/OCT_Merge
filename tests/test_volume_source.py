from pathlib import Path

import numpy as np

from oct_merge_task.io.volume_source import open_volume_source


def test_open_npy_volume_source_uses_memmap_and_reads_regions(tmp_path: Path) -> None:
    volume = np.arange(18 * 10 * 6, dtype=np.uint16).reshape(18, 10, 6)
    path = tmp_path / "volume.npy"
    np.save(path, volume)

    source = open_volume_source(path)

    assert source.shape == volume.shape
    assert source.dtype == np.dtype("uint16")
    assert isinstance(source.array, np.memmap)
    np.testing.assert_array_equal(source.read_region((4, 2, 1), (5, 4, 3)), volume[4:9, 2:6, 1:4])


def test_open_raw_volume_source_uses_memmap_and_validates_shape(tmp_path: Path) -> None:
    volume = np.arange(12 * 8 * 5, dtype=np.uint16).reshape(12, 8, 5)
    path = tmp_path / "volume.raw"
    volume.tofile(path)

    source = open_volume_source(path, shape=(12, 8, 5), dtype="uint16")

    assert source.shape == volume.shape
    assert source.dtype == np.dtype("uint16")
    assert isinstance(source.array, np.memmap)
    np.testing.assert_array_equal(source.read_region((3, 1, 2), (4, 5, 2)), volume[3:7, 1:6, 2:4])
