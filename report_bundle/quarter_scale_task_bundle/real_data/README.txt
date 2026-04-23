把真实数据放在这个文件夹中，程序会自动识别常见格式。

当前优先识别以下命名方式：
1. volume_a.npy + volume_b.npy
2. volume_a.tiff + volume_b.tiff
3. volume_a.tif + volume_b.tif
4. volume_a.raw + volume_b.raw

说明：
- A 和 B 必须是一对同尺寸体数据
- 如果使用 raw，后续仍需要提供 shape 和 dtype
