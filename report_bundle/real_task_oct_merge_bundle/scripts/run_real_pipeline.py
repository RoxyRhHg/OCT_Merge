from pathlib import Path
import argparse
import json
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.tools.real_pipeline import run_real_data_pipeline  # noqa: E402


def _parse_shape(value: Optional[str]):
    if value is None:
        return None
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be x,y,z")
    return tuple(parts)


def _parse_triplet(value: str):
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("value must be x,y,z")
    return tuple(parts)


def _parse_range(value: str):
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise ValueError("range must be min,max")
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run memory-aware OCT stitching on real A/B volumes.")
    parser.add_argument("--volume-a", required=True, help="Path to A volume: .npy / .tiff / .raw")
    parser.add_argument("--volume-b", required=True, help="Path to B volume: .npy / .tiff / .raw")
    parser.add_argument("--output-dir", required=True, help="Output directory for stitched bricks and summary")
    parser.add_argument("--shape-a", default=None, help="Required for raw A, format: x,y,z")
    parser.add_argument("--shape-b", default=None, help="Required for raw B, format: x,y,z")
    parser.add_argument("--dtype-a", default=None, help="Required for raw A, for example uint16")
    parser.add_argument("--dtype-b", default=None, help="Required for raw B, for example uint16")
    parser.add_argument("--brick-size", default="128,128,128", help="Output brick size, format: x,y,z")
    parser.add_argument("--overlap-range", default="0.05,0.20", help="Expected overlap fraction range, format: min,max")
    parser.add_argument("--preview-stride", default="8,8,8", help="Preview sampling stride, format: x,y,z")
    args = parser.parse_args()

    summary = run_real_data_pipeline(
        path_a=args.volume_a,
        path_b=args.volume_b,
        output_dir=args.output_dir,
        shape_a=_parse_shape(args.shape_a),
        shape_b=_parse_shape(args.shape_b),
        dtype_a=args.dtype_a,
        dtype_b=args.dtype_b,
        brick_size=_parse_triplet(args.brick_size),
        overlap_fraction_range=_parse_range(args.overlap_range),
        preview_stride=_parse_triplet(args.preview_stride),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
