from pathlib import Path
import argparse
import json
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.io.volume_source import open_volume_source  # noqa: E402


def _parse_shape(value: Optional[str]):
    if value is None:
        return None
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be x,y,z")
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read an OCT volume file and print basic shape information.")
    parser.add_argument("--input", required=True, help="Path to .npy / .tiff / .raw volume")
    parser.add_argument("--shape", default=None, help="Required for .raw, format: x,y,z")
    parser.add_argument("--dtype", default=None, help="Required for .raw, for example uint16")
    args = parser.parse_args()

    volume = open_volume_source(args.input, shape=_parse_shape(args.shape), dtype=args.dtype)
    info = {
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "access": "memmap",
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
