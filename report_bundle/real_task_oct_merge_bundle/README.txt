OCT 真实任务汇报包

这个文件夹用于直接演示真实任务主流程。

包含内容：
- src\ : 核心源码
- real_data\ : 放真实 A/B 数据
- scripts\ : 读取输入、运行真实 pipeline、烟测、打开汇报页
- report\ : 交互式汇报页
- smoke_data\ : 自带小型演示输入
- smoke_run\ : 可直接展示的运行结果

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
   scripts\open_report.bat
2. 如果要快速演示现成结果：
   直接查看 report\index.html 和 smoke_run\real_pipeline_summary.json
3. 如果要重新跑自带烟测数据：
   scripts\run_smoke_test.bat
4. 如果要换成真实数据：
   把 A/B 数据放进 real_data\
5. 运行真实数据主流程：
   python scripts\run_real_pipeline.py --volume-a real_data\volume_a.npy --volume-b real_data\volume_b.npy --output-dir real_run
6. 如果重新生成了 smoke_run，想让页面显示最新结果：
   python ..\..\scripts\refresh_report_payload.py
7. 如果要检查任务规模在 48GB 预算下是否成立：
   python ..\..\scripts\check_4090_feasibility.py --shape-a 3000,1500,2000 --shape-b 3000,1500,2000
