from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from oct_merge_task.fusion.brick_store import DiskBackedBrickStore
from oct_merge_task.tools.real_pipeline import run_real_data_pipeline


def create_report_bundle(
    project_root: str | Path,
    bundle_dir: str | Path,
    source_run_dir: str | Path,
) -> dict:
    project_root = Path(project_root)
    bundle_dir = Path(bundle_dir)
    source_run_dir = Path(source_run_dir)

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    web_dir = bundle_dir / "web"
    scripts_dir = bundle_dir / "scripts"
    src_dir = bundle_dir / "src"
    real_data_dir = bundle_dir / "real_data"
    web_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    real_data_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(project_root / "src" / "oct_merge_task", src_dir / "oct_merge_task", dirs_exist_ok=True)
    shutil.copytree(source_run_dir / "web", web_dir, dirs_exist_ok=True)
    shutil.copytree(source_run_dir / "preview", bundle_dir / "preview_run", dirs_exist_ok=True)
    shutil.copytree(source_run_dir / "case", bundle_dir / "case", dirs_exist_ok=True)
    shutil.copy2(project_root / "scripts" / "read_input_volume.py", scripts_dir / "read_input_volume.py")
    shutil.copy2(project_root / "scripts" / "display_real_data.py", scripts_dir / "display_real_data.py")
    shutil.copy2(project_root / "real_data" / "README.txt", real_data_dir / "README.txt")

    _write_text(bundle_dir / "README_使用说明.txt", _readme_text(), encoding="utf-8-sig")
    _write_text(scripts_dir / "open_web_bundle.bat", _open_web_bat_text())

    manifest = {
        "bundle_dir": str(bundle_dir),
        "web_dir": str(web_dir),
        "scripts_dir": str(scripts_dir),
    }
    _write_text(bundle_dir / "bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def create_real_task_report_bundle(
    project_root: str | Path,
    bundle_dir: str | Path,
    include_smoke_data: bool = True,
) -> dict:
    project_root = Path(project_root)
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    src_dir = bundle_dir / "src"
    scripts_dir = bundle_dir / "scripts"
    report_dir = bundle_dir / "report"
    real_data_dir = bundle_dir / "real_data"
    for path in (src_dir, scripts_dir, report_dir, real_data_dir):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(project_root / "src" / "oct_merge_task", src_dir / "oct_merge_task", dirs_exist_ok=True)
    for script_name in ("read_input_volume.py", "display_real_data.py", "run_real_pipeline.py"):
        shutil.copy2(project_root / "scripts" / script_name, scripts_dir / script_name)

    _write_text(bundle_dir / "README.txt", _minimal_real_task_readme_text())
    _write_text(report_dir / "index.html", render_report_index_html())
    _write_text(real_data_dir / "README.txt", _real_data_readme_text())
    _write_text(scripts_dir / "open_report.bat", _open_report_bat_text())
    _write_text(scripts_dir / "run_smoke_test.bat", _run_smoke_test_bat_text())

    smoke_data_dir = bundle_dir / "smoke_data"
    report_payload = None
    if include_smoke_data:
        _write_smoke_data(smoke_data_dir)
        run_real_data_pipeline(
            path_a=smoke_data_dir / "volume_a.npy",
            path_b=smoke_data_dir / "volume_b.npy",
            output_dir=bundle_dir / "smoke_run",
            brick_size=(7, 6, 5),
            overlap_fraction_range=(0.05, 0.20),
        )
        report_payload = _build_smoke_report_payload(bundle_dir / "smoke_run")
        _write_text(report_dir / "payload.json", json.dumps(report_payload, indent=2, ensure_ascii=False))

    _write_text(report_dir / "index.html", render_report_index_html(report_payload))

    manifest = {
        "bundle_type": "real_task_oct_merge",
        "bundle_dir": str(bundle_dir),
        "src_dir": str(src_dir),
        "scripts_dir": str(scripts_dir),
        "report_dir": str(report_dir),
        "real_data_dir": str(real_data_dir),
        "includes_smoke_data": bool(include_smoke_data),
        "entrypoints": {
            "inspect_volume": "scripts/read_input_volume.py",
            "run_real_pipeline": "scripts/run_real_pipeline.py",
            "smoke_test": "scripts/run_smoke_test.bat",
            "open_report": "scripts/open_report.bat",
        },
    }
    _write_text(bundle_dir / "bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def _write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def _write_smoke_data(smoke_data_dir: Path) -> None:
    smoke_data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260423)
    world = rng.integers(0, 4096, size=(42, 12, 8), dtype=np.uint16)
    np.save(smoke_data_dir / "volume_a.npy", world[:24])
    np.save(smoke_data_dir / "volume_b.npy", world[21:42])


def _build_smoke_report_payload(smoke_run_dir: Path) -> dict:
    summary = json.loads((smoke_run_dir / "real_pipeline_summary.json").read_text(encoding="utf-8"))
    store = DiskBackedBrickStore(smoke_run_dir / "stitched_bricks")
    layout = store.read_layout()
    smoke_data_dir = smoke_run_dir.parent / "smoke_data"
    volume_a = np.load(smoke_data_dir / "volume_a.npy").astype(np.float32)
    volume_b = np.load(smoke_data_dir / "volume_b.npy").astype(np.float32)
    slices = []
    for z_index in range(int(layout["output_shape"][2])):
        plane = _read_axial_slice(store, layout, z_index)
        slices.append(_encode_plane(plane))
    return {
        "summary": summary,
        "slices": slices,
        "point_clouds": {
            "volume_a": _build_volume_point_cloud(volume_a, "Volume A", "#51d6ff"),
            "volume_b": _build_volume_point_cloud(volume_b, "Volume B", "#ff9a4d"),
            "stitched": _build_stitched_point_cloud(store, layout, volume_a, volume_b, int(summary["estimated_transform"]["tx"])),
        },
    }


def build_report_payload_from_smoke_run(smoke_run_dir: Path) -> dict:
    return _build_smoke_report_payload(Path(smoke_run_dir))


def _read_axial_slice(store: DiskBackedBrickStore, layout: dict, z_index: int) -> np.ndarray:
    output_shape = layout["output_shape"]
    plane = np.zeros((output_shape[0], output_shape[1]), dtype=np.float32)
    for task in layout["tasks"]:
        origin = task["origin"]
        shape = task["shape"]
        if not (origin[2] <= z_index < origin[2] + shape[2]):
            continue
        brick_id = tuple(int(v) for v in task["brick_id"])
        brick = store.read_brick(brick_id)
        local_index = z_index - origin[2]
        plane[
            origin[0] : origin[0] + shape[0],
            origin[1] : origin[1] + shape[1],
        ] = brick[:, :, local_index]
    return plane


def _encode_plane(plane: np.ndarray) -> dict:
    normalized = np.asarray(plane, dtype=np.float32)
    normalized = normalized - float(normalized.min())
    max_value = float(normalized.max())
    if max_value > 0.0:
        normalized = normalized / max_value
    image = (normalized * 255.0).astype(np.uint8)
    return {
        "shape": list(image.shape),
        "data": image.flatten().tolist(),
    }


def _build_volume_point_cloud(volume: np.ndarray, name: str, base_color: str) -> dict:
    coords = np.argwhere(volume >= np.percentile(volume, 82))
    values = volume[coords[:, 0], coords[:, 1], coords[:, 2]] if len(coords) else np.array([], dtype=np.float32)
    labels = []
    legend = [
        {"label": "Low Response", "color": "#2a6f97"},
        {"label": "Medium Response", "color": base_color},
        {"label": "High Response", "color": "#ffe66d"},
    ]
    for value in values:
        if value < np.percentile(values, 35) if len(values) else 0.0:
            labels.append("Low Response")
        elif value < np.percentile(values, 70) if len(values) else 0.0:
            labels.append("Medium Response")
        else:
            labels.append("High Response")
    if len(coords) > 12000:
        order = np.argsort(values)[-12000:]
        coords = coords[order]
        values = values[order]
        labels = [labels[i] for i in order]
    return {
        "name": name,
        "legend": legend,
        "count": int(len(coords)),
        "points": coords.tolist(),
        "values": values.tolist(),
        "labels": labels,
        "shape": list(volume.shape),
        "color": base_color,
    }


def _build_stitched_point_cloud(
    store: DiskBackedBrickStore,
    layout: dict,
    volume_a: np.ndarray,
    volume_b: np.ndarray,
    tx: int,
) -> dict:
    points = []
    values = []
    labels = []
    threshold = 0.35
    for task in layout["tasks"]:
        brick_id = tuple(int(v) for v in task["brick_id"])
        brick = store.read_brick(brick_id)
        coords = np.argwhere(brick >= threshold * max(1.0, float(brick.max())))
        origin = np.array(task["origin"], dtype=np.int32)
        for coord in coords:
            point = coord + origin
            points.append(point.tolist())
            values.append(float(brick[tuple(coord)]))
            in_a = point[0] < volume_a.shape[0] and point[1] < volume_a.shape[1] and point[2] < volume_a.shape[2]
            in_b = tx <= point[0] < tx + volume_b.shape[0] and point[1] < volume_b.shape[1] and point[2] < volume_b.shape[2]
            if in_a and in_b:
                labels.append("Overlap")
            elif in_a:
                labels.append("A-only")
            else:
                labels.append("B-only")
    if len(points) > 12000:
        order = np.argsort(values)[-12000:]
        points = [points[i] for i in order]
        values = [values[i] for i in order]
        labels = [labels[i] for i in order]
    return {
        "name": "Stitched",
        "legend": [
            {"label": "A-only", "color": "#51d6ff"},
            {"label": "B-only", "color": "#ff9a4d"},
            {"label": "Overlap", "color": "#f6ff7e"},
        ],
        "count": len(points),
        "points": points,
        "values": values,
        "labels": labels,
        "shape": layout["output_shape"],
        "color": "#7bf0b2",
    }


def _real_task_readme_text() -> str:
    return """# OCT 真实任务汇报包

这个文件夹用于汇报和复现实验，不依赖原工程目录即可运行核心真实数据 pipeline。

## 已覆盖的真实工程问题

1. overlap 不是固定先验。当前只给默认范围 5% 到 20%，算法在低分辨 preview 上估计实际 overlap。
2. 输入体数据不默认整块转 float32。`.npy` 和 `.raw` 使用 memmap，`.tif/.tiff` 使用 tifffile memmap。
3. 拼接输出按 brick 流式写盘，不创建完整 stitched float32 体。
4. benchmark 会真实读取 stitched bricks 并组装切片，报告 disk_reads、mean_slice_ms、estimated_fps。
5. 当前第一阶段以 axis=0 平移配准为主，已经补上显存规划器和 GPU-ready 全局配准接口，并保留 CPU fallback。

## 快速运行

1. 可先运行 `scripts\\run_smoke_test.bat` 验证包能独立工作。
2. 将真实数据放入 `real_data\\`，参考 `real_data\\README.txt`。
3. 使用 `python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.npy --volume-b real_data\\volume_b.npy --output-dir real_run` 运行真实数据拼接。

## 文件夹说明

- `src\\oct_merge_task\\`: 核心源码。
- `scripts\\`: 检查输入、运行真实 pipeline、烟测脚本。
- `docs\\`: 汇报摘要和运行命令。
- `real_data\\`: 放真实 A/B 体数据。
- `smoke_data\\`: 小型自测数据，便于演示。
"""


def _minimal_real_task_readme_text() -> str:
    return """OCT 真实任务汇报包

这个文件夹用于直接演示真实任务主流程。

包含内容：
- src\\ : 核心源码
- real_data\\ : 放真实 A/B 数据
- scripts\\ : 读取输入、运行真实 pipeline、烟测、打开汇报页
- report\\ : 交互式汇报页
- smoke_data\\ : 自带小型演示输入
- smoke_run\\ : 可直接展示的运行结果

当前算法实现思路：
1. 输入层：
   真实数据通过 memmap 方式读取，不默认整块转 float32，避免两个大体数据一次性占满内存或显存。
2. overlap 估计：
   overlap 不当作固定先验，默认在 5% 到 20% 范围内搜索，用低分辨 preview 上的相关性估计实际重叠长度。
3. 显存规划：
   已加入 MemoryPlanner，根据 48GB 预算规划 slab 形状，后续 GPU 主干会按 slab/brick 方式推进，而不是整块送入显存。
4. 全局配准：
   当前第一阶段以 axis=0 平移配准为主，已经补上基于 torch FFT 的 CUDA 全局配准接口，并保留 CPU fallback。
5. 主干融合：
   拼接输出按 brick 流式写盘，避免创建完整 stitched float32 体；当前融合主干已支持 torch-aware 路径。
6. 结果输出：
   benchmark 会真实读取 stitched bricks 并组装切片，输出 overlap、brick_count、disk_reads、mean_slice_ms、estimated_fps 等指标。

该文件夹的使用方式：
1. 直接打开图形化页面：
   scripts\\open_report.bat
2. 如果要快速演示现成结果：
   直接查看 report\\index.html 和 smoke_run\\real_pipeline_summary.json
3. 如果要重新跑自带烟测数据：
   scripts\\run_smoke_test.bat
4. 如果要换成真实数据：
   把 A/B 数据放进 real_data\\
5. 运行真实数据主流程：
   python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.npy --volume-b real_data\\volume_b.npy --output-dir real_run
6. 如果重新生成了 smoke_run，想让页面显示最新结果：
   python ..\\..\\scripts\\refresh_report_payload.py
7. 如果要检查任务规模在 48GB 预算下是否成立：
   python ..\\..\\scripts\\check_4090_feasibility.py --shape-a 3000,1500,2000 --shape-b 3000,1500,2000
"""


def _engineering_summary_text() -> str:
    return """# 工程方案摘要

## 任务约束

- 目标体数据规模约为 `3000 x 1500 x 2000`。
- 单个 float32 体约 36GB，两个体直接常驻会超过 48GB 显存预算。
- 两个体的 overlap 约 10%，但具体值不是先验，需要估计。
- 显示目标是 30 Hz，真实评估必须包含 brick 读取和切片组装。

## 当前实现

1. `VolumeSource` 以 memmap 方式打开输入体，提供 `read_region` 分块读取接口。
2. `MemoryPlanner` 根据 48GB 预算规划安全 slab 形状。
3. `estimate_axis_overlap` 在给定 overlap 范围内用 NCC 搜索实际重叠长度。
4. `GPUGlobalRegistrar` 已提供 GPU-ready 的全局配准接口，当前支持 axis=0 平移，并可 CPU fallback。
5. `StreamingBrickStitcher` 按输出 brick 读取 A/B 对应区域并融合写盘。
6. `benchmark_brick_store_slices` 真实读取 brick 组装切片，不再使用空循环估算 FPS。

## 当前边界

- 第一阶段只实现 axis=0 平移估计。
- 旋转、尺度误差、复杂非刚性畸变需要下一阶段 GPU 配准和局部形变模块。
- 当前 benchmark 是 I/O 与切片组装基线，不等同于最终 GPU 纹理渲染帧率。
"""


def _presentation_checklist_text() -> str:
    return """# 汇报检查清单

## 开场要点

- 明确任务规模：`3000 x 1500 x 2000`，单 float32 体约 36GB。
- 强调两个体不能同时完整放入 48GB GPU 显存。
- 说明 overlap 约 10%，但不是固定先验，工程上需要估计。
- 说明当前已经开始把主干向 GPU 模块推进，不是只停留在 CPU demo。

## 已完成内容

- 输入通过 memmap 分块读取。
- 已有 `MemoryPlanner` 和 GPU-ready 全局配准接口。
- overlap 在范围内估计，不硬编码为 10%。
- 输出按 brick 流式写盘。
- benchmark 真实读取 bricks 组装切片。
- 交付包可独立运行 smoke test。

## 需要如实说明的边界

- 当前第一阶段只处理 axis=0 平移。
- GPU kernel、非刚性配准和最终 30 Hz 渲染器是下一阶段。
- 当前 FPS 是 brick I/O 和切片组装基线，不代表最终显示器帧率。

## 现场演示顺序

1. 打开 `report\\index.html` 讲工程约束和方案。
2. 运行 `scripts\\run_smoke_test.bat` 展示自包含可复现。
3. 展示 `smoke_run\\real_pipeline_summary.json` 中的 overlap、brick_count、disk_reads。
4. 说明替换真实数据时只需放入 `real_data\\` 并运行 `scripts\\run_real_pipeline.py`。
"""


def render_report_index_html(payload: dict | None = None) -> str:
    embedded_payload = json.dumps(payload, ensure_ascii=False) if payload is not None else "null"
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCT 图形化数据页面</title>
  <style>
    :root {
      --bg: #050913;
      --panel: rgba(6, 12, 24, 0.85);
      --card: rgba(13, 26, 44, 0.6);
      --ink: #d9f4ff;
      --muted: #9cc3d6;
      --line: rgba(120, 220, 255, 0.18);
      --accent: #51d6ff;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #10223d 0%, #050913 50%, #02040a 100%);
      color: var(--ink);
      overflow: hidden;
    }
    .layout {
      display: grid;
      grid-template-columns: 2fr 1fr;
      height: 100vh;
    }
    .hero {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      padding: 18px;
    }
    .hero canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .panel {
      padding: 24px;
      background: var(--panel);
      border-left: 1px solid var(--line);
      overflow: auto;
      backdrop-filter: blur(12px);
    }
    h1 { margin-top: 0; font-size: 28px; }
    h2 { margin: 0 0 14px; font-size: 22px; }
    p { color: var(--muted); font-size: 16px; line-height: 1.6; }
    .card {
      margin-bottom: 18px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
    }
    .slice-stage {
      position: relative;
      width: 100%;
      aspect-ratio: 4 / 3;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: rgba(0,0,0,0.18);
      cursor: grab;
    }
    .slice-stage:active { cursor: grabbing; }
    .slice-canvas { width: 100%; height: 100%; display: block; image-rendering: pixelated; }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 16px;
    }
    .toolbar input[type="range"] { width: 160px; accent-color: var(--accent); }
    .toolbar button {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255,255,255,0.05);
      color: var(--ink);
      padding: 8px 12px;
      cursor: pointer;
    }
    .toolbar label {
      color: var(--muted);
      font-size: 15px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
    }
    .legend-swatch {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      display: inline-block;
    }
    .small { font-size: 14px; color: var(--muted); }
    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; height: auto; }
      .hero { min-height: 360px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="hero">
      <canvas id="scene"></canvas>
    </div>
    <div class="panel">
      <h1>OCT 图形化数据页面</h1>
      <div class="toolbar">
        <label>View
          <select id="viewMode">
            <option value="stitched">Stitched</option>
            <option value="volume_a">Volume A</option>
            <option value="volume_b">Volume B</option>
          </select>
        </label>
        <button id="resetView">Reset View</button>
        <label>Slice <input id="sliceSlider" type="range" min="0" max="0" value="0"></label>
        <label>Zoom <input id="zoomSlider" type="range" min="100" max="400" value="100"></label>
        <label>Point Size <input id="pointSize" type="range" min="1" max="8" value="2"></label>
      </div>
      <div class="card"><strong>Algorithm Output</strong><div id="summary">Loading summary...</div></div>
      <div class="card"><strong>Structure Legend</strong><div id="legend"></div></div>
      <div class="card">
        <strong>Slice View</strong>
        <div class="slice-stage" id="sliceStage" style="margin-top:12px;">
          <canvas id="sliceCanvas" class="slice-canvas"></canvas>
        </div>
        <div class="small" id="sliceInfo" style="margin-top:10px;">Loading smoke_run...</div>
      </div>
      <div class="card">
        <strong>Current Algorithm</strong>
        <ul>
          <li>输入通过 memmap 分块读取。</li>
          <li>MemoryPlanner 负责 slab 预算。</li>
          <li>overlap 在 5% 到 20% 范围内估计。</li>
          <li>全局配准主干可走 torch FFT CUDA。</li>
          <li>输出按 brick 流式写盘。</li>
        </ul>
      </div>
    </div>
  </div>
  <script>
    const payload = __EMBEDDED_PAYLOAD__;
    const summaryNode = document.getElementById('summary');
    const legend = document.getElementById('legend');
    const sliceInfo = document.getElementById('sliceInfo');
    const sliceStage = document.getElementById('sliceStage');
    const sliceCanvas = document.getElementById('sliceCanvas');
    const sliceCtx = sliceCanvas.getContext('2d');
    const scene = document.getElementById('scene');
    const sceneCtx = scene.getContext('2d');
    const resetView = document.getElementById('resetView');
    const sliceSlider = document.getElementById('sliceSlider');
    const zoomSlider = document.getElementById('zoomSlider');
    const pointSizeInput = document.getElementById('pointSize');
    const viewMode = document.getElementById('viewMode');

    let summary = null;
    let slices = [];
    let pointClouds = {};
    let currentSlice = 0;
    let zoom = 1;
    let panX = 0;
    let panY = 0;
    let rotY = 0.7;
    let rotX = -0.4;
    let dragging = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let sceneDragging = false;
    let sceneStartX = 0;
    let sceneStartY = 0;
    let frameTick = 0;

    function resizeCanvas() {
      sliceCanvas.width = sliceStage.clientWidth;
      sliceCanvas.height = sliceStage.clientHeight;
      scene.width = scene.clientWidth;
      scene.height = scene.clientHeight;
    }

    async function boot() {
      resizeCanvas();
      if (!payload) {
        summaryNode.textContent = 'No embedded report payload found.';
        return;
      }
      summary = payload.summary;
      slices = payload.slices;
      pointClouds = payload.point_clouds || {};
      sliceSlider.max = String(slices.length - 1);
      updateSummary();
      updateLegend();
      await drawSlice(0);
      requestAnimationFrame(loop);
    }

    function updateSummary() {
      summaryNode.innerHTML = `
        <div>Registration mode: ${summary.registration.mode}</div>
        <div>Fusion mode: ${summary.fusion.mode}</div>
        <div>Overlap voxels: ${summary.estimated_overlap_voxels}</div>
        <div>Brick count: ${summary.brick_count}</div>
        <div>Slice FPS baseline: ${summary.benchmark.estimated_fps.toFixed(2)}</div>
        <div>Planned slab: ${summary.memory_budget.planned_slab_shape.join(' x ')}</div>
        <div>GPU budget: ${(summary.memory_budget.max_gpu_bytes / (1024 ** 3)).toFixed(1)} GB</div>
      `;
    }

    function currentPointCloud() {
      return pointClouds[viewMode.value];
    }

    function updateLegend() {
      const cloud = currentPointCloud();
      if (!cloud) {
        legend.innerHTML = '<div class="small">No point cloud loaded.</div>';
        return;
      }
      legend.innerHTML = cloud.legend.map(item => `
        <div class="legend-item">
          <span class="legend-swatch" style="background:${item.color}"></span>
          <span>${item.label}</span>
        </div>
      `).join('');
    }

    async function drawSlice(index) {
      currentSlice = index;
      sliceSlider.value = String(index);
      const slice = slices[index];
      const [height, width] = slice.shape;
      const image = sliceCtx.createImageData(width, height);
      for (let i = 0; i < slice.data.length; i++) {
        const value = slice.data[i];
        const idx = i * 4;
        image.data[idx] = value;
        image.data[idx + 1] = value;
        image.data[idx + 2] = Math.min(255, value + 28);
        image.data[idx + 3] = 255;
      }
      const offscreen = document.createElement('canvas');
      offscreen.width = width;
      offscreen.height = height;
      offscreen.getContext('2d').putImageData(image, 0, 0);

      sliceCtx.clearRect(0, 0, sliceCanvas.width, sliceCanvas.height);
      const drawWidth = sliceCanvas.width * zoom;
      const drawHeight = sliceCanvas.height * zoom;
      sliceCtx.save();
      sliceCtx.translate(panX, panY);
      sliceCtx.drawImage(offscreen, 0, 0, drawWidth, drawHeight);
      sliceCtx.restore();

      sliceInfo.textContent = `Slice ${index + 1}/${slices.length} | zoom ${zoom.toFixed(2)}x | pan (${Math.round(panX)}, ${Math.round(panY)})`;
    }

    function rotatePoint(point) {
      const cloud = currentPointCloud();
      const [x, y, z] = point;
      const cx = cloud.shape[0] / 2;
      const cy = cloud.shape[1] / 2;
      const cz = cloud.shape[2] / 2;
      let px = x - cx;
      let py = y - cy;
      let pz = z - cz;

      const cosy = Math.cos(rotY), siny = Math.sin(rotY);
      const cosx = Math.cos(rotX), sinx = Math.sin(rotX);
      let x1 = px * cosy - pz * siny;
      let z1 = px * siny + pz * cosy;
      let y1 = py * cosx - z1 * sinx;
      let z2 = py * sinx + z1 * cosx;
      return [x1, y1, z2];
    }

    function projectPoint(rotated, zoomFactor) {
      const cloud = currentPointCloud();
      const maxDim = Math.max(cloud.shape[0], cloud.shape[1], cloud.shape[2]);
      const perspective = 1.0 / (1.0 + rotated[2] / (maxDim * 1.4));
      const sx = scene.width / 2 + rotated[0] * zoomFactor * perspective * 6.0;
      const sy = scene.height / 2 + rotated[1] * zoomFactor * perspective * 6.0;
      return [sx, sy, perspective];
    }

    function colorForLabel(label, fallback) {
      const colors = {
        'Low Response': '#2a6f97',
        'Medium Response': '#51d6ff',
        'High Response': '#ffe66d',
        'A-only': '#51d6ff',
        'B-only': '#ff9a4d',
        'Overlap': '#f6ff7e',
      };
      return colors[label] || fallback;
    }

    function loop() {
      frameTick += 1;
      sceneCtx.clearRect(0, 0, scene.width, scene.height);
      const cloud = currentPointCloud();
      if (cloud) {
        const zoomFactor = Number(zoomSlider.value) / 100;
        const baseSize = Number(pointSizeInput.value);
        for (let i = 0; i < cloud.points.length; i++) {
          const rotated = rotatePoint(cloud.points[i]);
          const [sx, sy, scale] = projectPoint(rotated, zoomFactor);
          const size = Math.max(0.5, baseSize * scale * 2.0);
          const intensity = cloud.values[i];
          const alpha = Math.min(0.95, 0.15 + intensity * 0.85);
          const color = colorForLabel(cloud.labels[i], cloud.color || '#7bf0b2');
          const r = parseInt(color.slice(1, 3), 16);
          const g = parseInt(color.slice(3, 5), 16);
          const b = parseInt(color.slice(5, 7), 16);
          sceneCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
          sceneCtx.beginPath();
          sceneCtx.arc(sx, sy, size, 0, Math.PI * 2);
          sceneCtx.fill();
        }
      }
      requestAnimationFrame(loop);
    }

    resetView.addEventListener('click', () => {
      zoom = 1;
      panX = 0;
      panY = 0;
      rotX = -0.4;
      rotY = 0.7;
      zoomSlider.value = '100';
      drawSlice(currentSlice);
    });

    sliceSlider.addEventListener('input', () => {
      drawSlice(Number(sliceSlider.value));
    });

    zoomSlider.addEventListener('input', () => {
      zoom = Number(zoomSlider.value) / 100;
      drawSlice(currentSlice);
    });

    viewMode.addEventListener('change', () => {
      updateLegend();
    });

    sliceStage.addEventListener('mousedown', (event) => {
      dragging = true;
      dragStartX = event.clientX - panX;
      dragStartY = event.clientY - panY;
    });

    window.addEventListener('mouseup', () => {
      dragging = false;
    });

    window.addEventListener('mousemove', (event) => {
      if (!dragging) return;
      panX = event.clientX - dragStartX;
      panY = event.clientY - dragStartY;
      drawSlice(currentSlice);
    });

    sliceStage.addEventListener('wheel', (event) => {
      event.preventDefault();
      const current = Number(zoomSlider.value);
      const next = Math.max(100, Math.min(400, current - event.deltaY * 0.05));
      zoomSlider.value = String(next);
      zoom = next / 100;
      drawSlice(currentSlice);
    }, { passive: false });

    scene.addEventListener('mousedown', (event) => {
      sceneDragging = true;
      sceneStartX = event.clientX;
      sceneStartY = event.clientY;
    });

    scene.addEventListener('wheel', (event) => {
      event.preventDefault();
      const current = Number(zoomSlider.value);
      const next = Math.max(100, Math.min(400, current - event.deltaY * 0.05));
      zoomSlider.value = String(next);
      zoom = next / 100;
    }, { passive: false });

    window.addEventListener('mouseup', () => {
      sceneDragging = false;
    });

    window.addEventListener('mousemove', (event) => {
      if (!sceneDragging) return;
      const dx = event.clientX - sceneStartX;
      const dy = event.clientY - sceneStartY;
      rotY += dx * 0.01;
      rotX += dy * 0.01;
      sceneStartX = event.clientX;
      sceneStartY = event.clientY;
    });

    window.addEventListener('resize', () => {
      resizeCanvas();
      drawSlice(currentSlice);
    });

    boot();
  </script>
</body>
</html>
""".replace("__EMBEDDED_PAYLOAD__", embedded_payload)


def _run_commands_text() -> str:
    return """# 运行命令

## 检查输入体

```powershell
python scripts\\read_input_volume.py --input real_data\\volume_a.npy
```

RAW 输入需要提供 shape 和 dtype：

```powershell
python scripts\\read_input_volume.py --input real_data\\volume_a.raw --shape 3000,1500,2000 --dtype uint16
```

## 运行真实 pipeline

```powershell
python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.npy --volume-b real_data\\volume_b.npy --output-dir real_run
```

RAW 输入示例：

```powershell
python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.raw --volume-b real_data\\volume_b.raw --shape-a 3000,1500,2000 --shape-b 3000,1500,2000 --dtype-a uint16 --dtype-b uint16 --output-dir real_run
```

## 调整 overlap 搜索范围

```powershell
python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.npy --volume-b real_data\\volume_b.npy --output-dir real_run --overlap-range 0.05,0.20
```

## 烟测

```powershell
scripts\\run_smoke_test.bat
```
"""


def _real_data_readme_text() -> str:
    return """把真实 OCT A/B 数据放在此目录。

推荐命名：
1. volume_a.npy + volume_b.npy
2. volume_a.tiff + volume_b.tiff
3. volume_a.raw + volume_b.raw

说明：
- `.npy` 推荐保存为 uint16 或 float16/float32，程序会用 memmap 打开。
- `.raw` 必须在命令里提供 shape 和 dtype。
- A/B 的 y/z 尺寸当前要求一致。
- overlap 不需要给定精确值，默认会在 5% 到 20% 范围搜索。
"""


def _run_smoke_test_bat_text() -> str:
    return """@echo off
set SCRIPT_DIR=%~dp0
set BUNDLE_DIR=%SCRIPT_DIR%..
python "%SCRIPT_DIR%run_real_pipeline.py" --volume-a "%BUNDLE_DIR%\\smoke_data\\volume_a.npy" --volume-b "%BUNDLE_DIR%\\smoke_data\\volume_b.npy" --output-dir "%BUNDLE_DIR%\\smoke_run" --brick-size 7,6,5 --overlap-range 0.05,0.20
pause
"""


def _open_report_bat_text() -> str:
    return """@echo off
set SCRIPT_DIR=%~dp0
start "" "%SCRIPT_DIR%..\\report\\index.html"
"""


def _readme_text() -> str:
    return """OCT Merge Task 使用说明

一、先怎么展示
1. 直接打开：
   web\\index.html
2. 如果浏览器对本地页面有限制，也可以双击：
   scripts\\open_web_bundle.bat

二、当前包里包含什么
1. case\\
   当前模拟生成的两个同尺寸数据体，以及 preview 数据
2. preview_run\\
   preview 级算法输出，包括 stitched bricks 和 summary
3. web\\
   浏览器展示页面和 payload
4. src\\oct_merge_task\\
   当前源码
5. scripts\\read_input_volume.py
   用于读取并检查真实数据体格式
6. scripts\\display_real_data.py
   用于自动识别 real_data 文件夹中的 A/B 两个体数据并直接读入
7. real_data\\README.txt
   说明真实数据应该怎样放置

三、当前模拟数据满足的约束
1. A/B 两个数据体尺寸相同
2. 当前 quarter-scale 配置：
   - A: 750 x 375 x 500
   - B: 750 x 375 x 500
3. overlap: 75 体素
4. overlap ratio: 10%

四、别的电脑需要什么环境
如果只是打开网页看结果：
- 一般不需要 Python 环境

如果要继续跑程序：
- Python 3.9+
- numpy
- torch
- tifffile

五、真实数据输入格式
当前已提供真实数据读取入口，支持：
1. .npy
2. .tiff
3. .raw

六、真实数据自动识别入口
入口脚本：
scripts\\display_real_data.py

它会自动在 `real_data\\` 文件夹里查找以下命名组合：
1. volume_a.npy + volume_b.npy
2. volume_a.tiff + volume_b.tiff
3. volume_a.tif + volume_b.tif
4. volume_a.raw + volume_b.raw

如果找到，就自动读取 A/B 两个体，并输出它们的 shape、dtype、最小值和最大值。

七、真实数据读取入口是怎么实现的
入口脚本：
scripts\\read_input_volume.py

它内部调用：
src\\oct_merge_task\\io\\real_input.py 中的 load_volume_file(...)

当前实现方式：
1. .npy
   - 使用 numpy.load 读取
   - 统一转为 float32
2. .tif / .tiff
   - 使用 tifffile.imread 读取 3D stack
   - 统一转为 float32
3. .raw
   - 需要额外提供 --shape x,y,z
   - 需要额外提供 --dtype
   - 使用 numpy.fromfile 读取后 reshape 成三维体

八、如何检查真实数据能不能读
命令：
python scripts\\read_input_volume.py --input 你的文件路径

如果输入是 raw，还需要补：
python scripts\\read_input_volume.py --input 你的文件路径 --shape x,y,z --dtype uint16

九、当前算法实现思路（已完成部分）
1. 先生成一个更大的底层 world
2. 再从 world 裁出两个同尺寸体数据 A 和 B
3. overlap 比例固定为 10%
4. preview 级别对 B 体加入刚体差异和局部畸变
5. 在 preview 上运行：
   - VolumeStore 建金字塔
   - GlobalRegistrar 做刚体配准
   - LocalRefiner 做局部修正
   - SimpleStitcher 生成 stitched
   - BrickStitcher 输出 stitched bricks
   - SliceBenchmark 输出切片性能指标
6. 最后把 preview 结果导出为浏览器 payload 并显示

十、目前已经做了什么
1. 新项目已经独立于旧 demo 工程
2. 已有 quarter-scale 模拟
3. A/B 同尺寸与 10% overlap 已在结果里明确输出
4. 已有 preview 级配准、局部修正、brick 输出、benchmark
5. 已有浏览器展示页面
6. 页面里已区分：
   - Volume A
   - Volume B
   - Stitched
   - A-only / B-only / Overlap

十一、目前还需要继续完善的地方
1. 浏览器页面还需要更强的联动与交互
2. 结构级语义标签现在还是启发式
3. 当前 3D 视图仍是轻量点云展示，不是最终定稿渲染器
"""


def _open_web_bat_text() -> str:
    return """@echo off
set SCRIPT_DIR=%~dp0
start "" "%SCRIPT_DIR%..\\web\\index.html"
"""
