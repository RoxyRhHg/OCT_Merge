from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.tools.feasibility import assess_single_gpu_feasibility  # noqa: E402


def _parse_shape(value: str):
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be x,y,z")
    return tuple(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the OCT pipeline structure is feasible on a single RTX 4090 style budget.")
    parser.add_argument("--shape-a", required=True, help="Shape of volume A, format x,y,z")
    parser.add_argument("--shape-b", required=True, help="Shape of volume B, format x,y,z")
    parser.add_argument("--gpu-budget-gb", type=float, default=48.0, help="GPU memory budget in GB. Default follows the task assumption of 48GB.")
    parser.add_argument("--bytes-per-voxel", type=int, default=4, help="Bytes per voxel. Use 4 for float32, 2 for uint16/float16 style planning.")
    args = parser.parse_args()

    report = assess_single_gpu_feasibility(
        shape_a=_parse_shape(args.shape_a),
        shape_b=_parse_shape(args.shape_b),
        gpu_budget_gb=args.gpu_budget_gb,
        bytes_per_voxel=args.bytes_per_voxel,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
