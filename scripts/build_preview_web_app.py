from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.web.app import write_web_app  # noqa: E402
from oct_merge_task.web.payload import build_preview_web_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build browser-viewable payload and static app for a preview run.")
    parser.add_argument("--preview-run-dir", required=True, help="Directory containing task_simulation_summary.json")
    parser.add_argument("--output-dir", required=True, help="Directory for the web app")
    args = parser.parse_args()

    payload = build_preview_web_payload(args.preview_run_dir, args.output_dir)
    app = write_web_app(args.output_dir, payload=payload)
    print(json.dumps({"payload": payload, "app": app}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
