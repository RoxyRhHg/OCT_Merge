OCT 真实任务汇报包

这个文件夹用于直接演示真实任务主流程。

包含内容：
- src\ : 核心源码
- real_data\ : 放真实 A/B 数据
- scripts\ : 读取输入、运行真实 pipeline、烟测、打开汇报页
- report\ : 交互式汇报页
- smoke_data\ : 自带小型演示输入
- smoke_run\ : 可直接展示的运行结果

当前能力：
- overlap 不是固定先验，默认在 5% 到 20% 范围估计
- 输入通过 memmap 读取，不默认整块转 float32
- 输出按 brick 流式写盘
- 已加入 MemoryPlanner 和 GPU-ready 全局配准接口
- 当前可 CPU fallback

运行方式：
1. 打开汇报页：scripts\open_report.bat
2. 烟测：scripts\run_smoke_test.bat
3. 真实数据运行：
   python scripts\run_real_pipeline.py --volume-a real_data\volume_a.npy --volume-b real_data\volume_b.npy --output-dir real_run

当前边界：
- 第一阶段只支持 axis=0 平移配准
- GPU 局部精修、GPU 融合优化和最终 30Hz 渲染器仍在后续阶段
