from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oct_merge_task.bundle.report_bundle import build_report_payload_from_smoke_run, render_report_index_html  # noqa: E402


def main() -> int:
    bundle_dir = PROJECT_ROOT / "report_bundle" / "real_task_oct_merge_bundle"
    smoke_run_dir = bundle_dir / "smoke_run"
    report_dir = bundle_dir / "report"
    payload = build_report_payload_from_smoke_run(smoke_run_dir)
    (report_dir / "payload.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "index.html").write_text(render_report_index_html(payload), encoding="utf-8")
    print(json.dumps({"payload_path": str(report_dir / "payload.json"), "index_path": str(report_dir / "index.html")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
