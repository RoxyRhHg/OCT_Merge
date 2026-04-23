from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.bundle.report_bundle import create_report_bundle  # noqa: E402


def main() -> int:
    source_run_dir = PROJECT_ROOT / "artifacts" / "quarter_scale_web_v3"
    bundle_dir = PROJECT_ROOT / "report_bundle" / "quarter_scale_task_bundle"
    manifest = create_report_bundle(
        project_root=PROJECT_ROOT,
        bundle_dir=bundle_dir,
        source_run_dir=source_run_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
