from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.tools.half_scale_task import HalfScaleCaseConfig, generate_half_scale_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a half-scale OCT task case.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the case")
    parser.add_argument("--preview-step", type=int, default=8, help="Preview downsampling factor")
    args = parser.parse_args()

    config = HalfScaleCaseConfig(preview_step=args.preview_step)
    summary = generate_half_scale_case(Path(args.output_dir), config=config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
