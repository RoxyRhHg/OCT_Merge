from pathlib import Path

from oct_merge_task.bundle.report_bundle import create_real_task_report_bundle, create_report_bundle
from oct_merge_task.tools.half_scale_task import QuarterScaleCaseConfig, generate_half_scale_case, run_preview_pipeline_on_half_scale_case
from oct_merge_task.web.payload import build_preview_web_payload
from oct_merge_task.web.app import write_web_app


def test_report_bundle_contains_docs_scripts_and_web_assets(tmp_path: Path) -> None:
    project_root = Path(r"E:\Coding\OCT Merge Task")
    run_root = tmp_path / "quarter_scale_run"
    case_dir = run_root / "case"
    preview_dir = run_root / "preview"
    web_dir = run_root / "web"

    config = QuarterScaleCaseConfig()
    generate_half_scale_case(case_dir, config)
    run_preview_pipeline_on_half_scale_case(case_dir, preview_dir, config)
    payload = build_preview_web_payload(preview_dir, web_dir)
    write_web_app(web_dir, payload=payload)

    bundle_dir = tmp_path / "report_bundle"
    manifest = create_report_bundle(
        project_root=project_root,
        bundle_dir=bundle_dir,
        source_run_dir=run_root,
    )

    assert manifest["bundle_dir"] == str(bundle_dir)
    assert (bundle_dir / "README_使用说明.txt").exists()
    assert not (bundle_dir / "docs").exists()
    assert (bundle_dir / "web" / "index.html").exists()
    assert (bundle_dir / "web" / "payload.json").exists()
    assert (bundle_dir / "scripts" / "read_input_volume.py").exists()
    assert (bundle_dir / "scripts" / "open_web_bundle.bat").exists()


def test_real_task_report_bundle_is_self_contained_for_reporting(tmp_path: Path) -> None:
    project_root = Path(r"E:\Coding\OCT Merge Task")
    bundle_dir = tmp_path / "real_task_bundle"

    manifest = create_real_task_report_bundle(
        project_root=project_root,
        bundle_dir=bundle_dir,
        include_smoke_data=True,
    )

    assert manifest["bundle_type"] == "real_task_oct_merge"
    assert (bundle_dir / "README.txt").exists()
    assert not (bundle_dir / "docs").exists()
    assert (bundle_dir / "report" / "index.html").exists()
    assert (bundle_dir / "scripts" / "run_real_pipeline.py").exists()
    assert (bundle_dir / "scripts" / "open_report.bat").exists()
    assert (bundle_dir / "scripts" / "run_smoke_test.bat").exists()
    assert (bundle_dir / "real_data" / "README.txt").exists()
    assert (bundle_dir / "smoke_data" / "volume_a.npy").exists()
    assert (bundle_dir / "src" / "oct_merge_task" / "tools" / "real_pipeline.py").exists()
    assert "overlap 不是固定先验" in (bundle_dir / "README.txt").read_text(encoding="utf-8")
    assert "48GB" in (bundle_dir / "report" / "index.html").read_text(encoding="utf-8")
    assert manifest["entrypoints"]["open_report"] == "scripts/open_report.bat"
