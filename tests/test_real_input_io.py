from pathlib import Path

import numpy as np
import tifffile

from oct_merge_task.io.real_input import load_volume_file


def test_load_volume_file_reads_npy_volume(tmp_path: Path) -> None:
    volume = np.arange(8 * 6 * 4, dtype=np.float32).reshape(8, 6, 4)
    path = tmp_path / "vol.npy"
    np.save(path, volume)

    loaded = load_volume_file(path)

    np.testing.assert_allclose(loaded, volume)


def test_load_volume_file_can_memmap_npy_without_float32_copy(tmp_path: Path) -> None:
    volume = np.arange(8 * 6 * 4, dtype=np.uint16).reshape(8, 6, 4)
    path = tmp_path / "vol.npy"
    np.save(path, volume)

    loaded = load_volume_file(path, mmap_mode="r")

    assert isinstance(loaded, np.memmap)
    assert loaded.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(loaded[2:4], volume[2:4])


def test_load_volume_file_reads_tiff_stack(tmp_path: Path) -> None:
    volume = np.arange(10 * 8 * 6, dtype=np.uint16).reshape(10, 8, 6)
    path = tmp_path / "vol.tiff"
    tifffile.imwrite(path, volume)

    loaded = load_volume_file(path)

    np.testing.assert_array_equal(loaded, volume.astype(np.float32))


def test_load_volume_file_reads_raw_with_shape_and_dtype(tmp_path: Path) -> None:
    volume = np.arange(12 * 5 * 4, dtype=np.uint16).reshape(12, 5, 4)
    path = tmp_path / "vol.raw"
    volume.tofile(path)

    loaded = load_volume_file(path, shape=(12, 5, 4), dtype="uint16")

    np.testing.assert_array_equal(loaded, volume.astype(np.float32))
