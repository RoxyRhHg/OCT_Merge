OCT Merge Task 使用说明

一、先怎么展示
1. 直接打开：
   web\index.html
2. 如果浏览器对本地页面有限制，也可以双击：
   scripts\open_web_bundle.bat

二、当前包里包含什么
1. case\
   当前模拟生成的两个同尺寸数据体，以及 preview 数据
2. preview_run\
   preview 级算法输出，包括 stitched bricks 和 summary
3. web\
   浏览器展示页面和 payload
4. src\oct_merge_task\
   当前源码
5. scripts\read_input_volume.py
   用于读取并检查真实数据体格式
6. scripts\display_real_data.py
   用于自动识别 real_data 文件夹中的 A/B 两个体数据并直接读入
7. real_data\README.txt
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
scripts\display_real_data.py

它会自动在 `real_data\` 文件夹里查找以下命名组合：
1. volume_a.npy + volume_b.npy
2. volume_a.tiff + volume_b.tiff
3. volume_a.tif + volume_b.tif
4. volume_a.raw + volume_b.raw

如果找到，就自动读取 A/B 两个体，并输出它们的 shape、dtype、最小值和最大值。

六点五、真实数据拼接入口
入口脚本：
scripts\run_real_pipeline.py

示例：
python scripts\run_real_pipeline.py --volume-a real_data\volume_a.npy --volume-b real_data\volume_b.npy --output-dir real_run

如果输入是 raw，需要补充 shape 和 dtype：
python scripts\run_real_pipeline.py --volume-a real_data\volume_a.raw --volume-b real_data\volume_b.raw --shape-a 3000,1500,2000 --shape-b 3000,1500,2000 --dtype-a uint16 --dtype-b uint16 --output-dir real_run

当前真实数据 pipeline 的工程约束：
1. overlap 不是固定先验，只给默认范围 5% 到 20%，算法会在低分辨 preview 上估计实际 overlap。
2. 真实输入通过 memmap / tifffile memmap 打开，不默认整块转 float32。
3. 拼接输出按 brick 流式写入 stitched_bricks，不创建完整 float32 stitched volume。
4. benchmark 会实际读取 stitched bricks 并组装切片，输出 disk_reads、mean_slice_ms、estimated_fps。
5. 当前配准第一阶段仍以 axis=0 平移为主；真实旋转、尺度误差和复杂非刚性畸变需要下一阶段 GPU 配准/局部形变模块继续补齐。

七、真实数据读取入口是怎么实现的
入口脚本：
scripts\read_input_volume.py

它内部调用：
src\oct_merge_task\io\real_input.py 中的 load_volume_file(...)

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
python scripts\read_input_volume.py --input 你的文件路径

如果输入是 raw，还需要补：
python scripts\read_input_volume.py --input 你的文件路径 --shape x,y,z --dtype uint16

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
4. GPU 版 overlap 配准、brick 融合 kernel 和 30 Hz 生产渲染器仍需继续实现
