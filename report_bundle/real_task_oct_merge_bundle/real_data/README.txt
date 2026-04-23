把真实 OCT A/B 数据放在此目录。

推荐命名：
1. volume_a.npy + volume_b.npy
2. volume_a.tiff + volume_b.tiff
3. volume_a.raw + volume_b.raw

说明：
- `.npy` 推荐保存为 uint16 或 float16/float32，程序会用 memmap 打开。
- `.raw` 必须在命令里提供 shape 和 dtype。
- A/B 的 y/z 尺寸当前要求一致。
- overlap 不需要给定精确值，默认会在 5% 到 20% 范围搜索。
