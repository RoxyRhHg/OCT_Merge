# OCT 真实任务汇报包

这个文件夹用于汇报和复现实验，不依赖原工程目录即可运行核心真实数据 pipeline。

## 已覆盖的真实工程问题

1. overlap 不是固定先验。当前只给默认范围 5% 到 20%，算法在低分辨 preview 上估计实际 overlap。
2. 输入体数据不默认整块转 float32。`.npy` 和 `.raw` 使用 memmap，`.tif/.tiff` 使用 tifffile memmap。
3. 拼接输出按 brick 流式写盘，不创建完整 stitched float32 体。
4. benchmark 会真实读取 stitched bricks 并组装切片，报告 disk_reads、mean_slice_ms、estimated_fps。
5. 当前第一阶段以 axis=0 平移配准为主，为后续 GPU 配准、局部形变和 30 Hz 渲染器保留接口。

## 快速运行

1. 可先运行 `scripts\run_smoke_test.bat` 验证包能独立工作。
2. 将真实数据放入 `real_data\`，参考 `real_data\README.txt`。
3. 使用 `python scripts\run_real_pipeline.py --volume-a real_data\volume_a.npy --volume-b real_data\volume_b.npy --output-dir real_run` 运行真实数据拼接。

## 文件夹说明

- `src\oct_merge_task\`: 核心源码。
- `scripts\`: 检查输入、运行真实 pipeline、烟测脚本。
- `docs\`: 汇报摘要和运行命令。
- `real_data\`: 放真实 A/B 体数据。
- `smoke_data\`: 小型自测数据，便于演示。
