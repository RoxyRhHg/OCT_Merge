from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.io.real_data_entry import discover_real_data_pair  # noqa: E402
from oct_merge_task.io.real_input import load_volume_file  # noqa: E402


def main() -> int:
    real_data_dir = PROJECT_ROOT / "real_data"
    pair = discover_real_data_pair(real_data_dir)
    volume_a = load_volume_file(pair["volume_a"])
    volume_b = load_volume_file(pair["volume_b"])
    info = {
        "format": pair["format"],
        "volume_a_path": str(pair["volume_a"]),
        "volume_b_path": str(pair["volume_b"]),
        "shape_a": list(volume_a.shape),
        "shape_b": list(volume_b.shape),
        "dtype_a": str(volume_a.dtype),
        "dtype_b": str(volume_b.dtype),
        "min_a": float(volume_a.min()),
        "max_a": float(volume_a.max()),
        "min_b": float(volume_b.min()),
        "max_b": float(volume_b.max()),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
