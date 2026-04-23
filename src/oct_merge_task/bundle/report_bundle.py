from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


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
    docs_dir = bundle_dir / "docs"
    report_dir = bundle_dir / "report"
    real_data_dir = bundle_dir / "real_data"
    for path in (src_dir, scripts_dir, docs_dir, report_dir, real_data_dir):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(project_root / "src" / "oct_merge_task", src_dir / "oct_merge_task", dirs_exist_ok=True)
    for script_name in ("read_input_volume.py", "display_real_data.py", "run_real_pipeline.py"):
        shutil.copy2(project_root / "scripts" / script_name, scripts_dir / script_name)

    _write_text(bundle_dir / "README_汇报说明.txt", _real_task_readme_text())
    _write_text(docs_dir / "工程方案摘要.md", _engineering_summary_text())
    _write_text(docs_dir / "运行命令.md", _run_commands_text())
    _write_text(docs_dir / "汇报检查清单.md", _presentation_checklist_text())
    _write_text(report_dir / "index.html", _report_index_html_text())
    _write_text(real_data_dir / "README.txt", _real_data_readme_text())
    _write_text(scripts_dir / "open_report.bat", _open_report_bat_text())
    _write_text(scripts_dir / "run_smoke_test.bat", _run_smoke_test_bat_text())

    smoke_data_dir = bundle_dir / "smoke_data"
    if include_smoke_data:
        _write_smoke_data(smoke_data_dir)

    manifest = {
        "bundle_type": "real_task_oct_merge",
        "bundle_dir": str(bundle_dir),
        "src_dir": str(src_dir),
        "scripts_dir": str(scripts_dir),
        "docs_dir": str(docs_dir),
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


def _real_task_readme_text() -> str:
    return """# OCT 真实任务汇报包

这个文件夹用于汇报和复现实验，不依赖原工程目录即可运行核心真实数据 pipeline。

## 已覆盖的真实工程问题

1. overlap 不是固定先验。当前只给默认范围 5% 到 20%，算法在低分辨 preview 上估计实际 overlap。
2. 输入体数据不默认整块转 float32。`.npy` 和 `.raw` 使用 memmap，`.tif/.tiff` 使用 tifffile memmap。
3. 拼接输出按 brick 流式写盘，不创建完整 stitched float32 体。
4. benchmark 会真实读取 stitched bricks 并组装切片，报告 disk_reads、mean_slice_ms、estimated_fps。
5. 当前第一阶段以 axis=0 平移配准为主，为后续 GPU 配准、局部形变和 30 Hz 渲染器保留接口。

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


def _engineering_summary_text() -> str:
    return """# 工程方案摘要

## 任务约束

- 目标体数据规模约为 `3000 x 1500 x 2000`。
- 单个 float32 体约 36GB，两个体直接常驻会超过 48GB 显存预算。
- 两个体的 overlap 约 10%，但具体值不是先验，需要估计。
- 显示目标是 30 Hz，真实评估必须包含 brick 读取和切片组装。

## 当前实现

1. `VolumeSource` 以 memmap 方式打开输入体，提供 `read_region` 分块读取接口。
2. `estimate_axis_overlap` 在给定 overlap 范围内用 NCC 搜索实际重叠长度。
3. `StreamingBrickStitcher` 按输出 brick 读取 A/B 对应区域并融合写盘。
4. `benchmark_brick_store_slices` 真实读取 brick 组装切片，不再使用空循环估算 FPS。

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

## 已完成内容

- 输入通过 memmap 分块读取。
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


def _report_index_html_text() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCT 真实任务工程汇报</title>
  <style>
    :root {
      --bg: #07110f;
      --panel: #10221d;
      --ink: #e9fff7;
      --muted: #9fc8ba;
      --accent: #7bf0b2;
      --warn: #ffd166;
      --line: rgba(123, 240, 178, 0.22);
    }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
      background:
        radial-gradient(circle at 20% 0%, rgba(123, 240, 178, 0.18), transparent 34%),
        linear-gradient(135deg, #06100e 0%, #0b1714 46%, #020504 100%);
      color: var(--ink);
    }
    main { max-width: 1120px; margin: 0 auto; padding: 56px 24px; }
    .hero {
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 40px;
      background: rgba(16, 34, 29, 0.78);
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
    }
    h1 { font-size: clamp(34px, 6vw, 72px); line-height: 0.95; margin: 0 0 18px; }
    h2 { margin: 0 0 16px; font-size: 26px; }
    p { color: var(--muted); font-size: 18px; line-height: 1.65; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 24px; }
    .card {
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      background: rgba(255,255,255,0.035);
    }
    .metric { font-size: 34px; color: var(--accent); font-weight: 700; }
    .warn { color: var(--warn); }
    code { color: var(--accent); }
    ol, ul { color: var(--muted); font-size: 17px; line-height: 1.65; padding-left: 22px; }
    section { margin-top: 26px; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } .hero { padding: 26px; } }
  </style>
</head>
<body>
  <main>
    <div class="hero">
      <h1>OCT 三维体真实拼接任务</h1>
      <p>目标不是继续演示小 demo，而是处理真实工程约束：大体数据、48GB 显存预算、overlap 非固定先验、按 brick 流式输出，以及可复现实测 benchmark。</p>
      <div class="grid">
        <div class="card"><div class="metric">36GB</div><p>单个 `3000 x 1500 x 2000` float32 体约占用。</p></div>
        <div class="card"><div class="metric">48GB</div><p>目标单卡显存预算，不能容纳两个完整 float32 体和输出。</p></div>
        <div class="card"><div class="metric">~10%</div><p>overlap 只是先验范围中心，实际值由算法估计。</p></div>
      </div>
    </div>
    <section class="grid">
      <div class="card"><h2>内存策略</h2><p>输入体用 memmap 分块读取，融合阶段按输出 brick 拉取局部区域，不创建完整 stitched float32 体。</p></div>
      <div class="card"><h2>Overlap 估计</h2><p>默认在 5% 到 20% 范围内低分辨搜索，使用 NCC 估计 axis=0 平移重叠。</p></div>
      <div class="card"><h2>真实 Benchmark</h2><p>benchmark 会实际读取 stitched bricks 并组装切片，报告 disk_reads、mean_slice_ms、estimated_fps。</p></div>
    </section>
    <section class="card">
      <h2>演示命令</h2>
      <p>先运行烟测验证包独立可用：</p>
      <code>scripts\\run_smoke_test.bat</code>
      <p>替换真实数据后运行：</p>
      <code>python scripts\\run_real_pipeline.py --volume-a real_data\\volume_a.npy --volume-b real_data\\volume_b.npy --output-dir real_run</code>
    </section>
    <section class="card">
      <h2>当前边界</h2>
      <ul>
        <li>第一阶段实现 axis=0 平移配准和 streaming brick fusion。</li>
        <li>旋转、尺度误差、非刚性形变、GPU kernel 与最终 30 Hz 渲染器是下一阶段。</li>
        <li>当前 FPS 是 I/O 与切片组装基线，不等同于最终显示帧率。</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


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
