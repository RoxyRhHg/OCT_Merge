from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.tools.benchmark_sweep import run_benchmark_sweep  # noqa: E402


def _parse_config(value: str) -> dict:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    config = {}
    for part in parts:
        key, raw = part.split("=")
        config[key.strip()] = int(raw.strip())
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a benchmark sweep for stitched brick slices.")
    parser.add_argument("--brick-store-dir", required=True, help="Path to stitched_bricks directory")
    parser.add_argument("--output", required=True, help="JSON file to write sweep results")
    parser.add_argument("--config", action="append", required=True, help="Configuration like axis=2,passes=1")
    args = parser.parse_args()

    report = run_benchmark_sweep(
        brick_store_dir=Path(args.brick_store_dir),
        output_path=Path(args.output),
        configurations=[_parse_config(v) for v in args.config],
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
