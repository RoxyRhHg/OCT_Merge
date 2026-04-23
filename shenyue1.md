
---

## 一、仓库现状诊断（基于具体代码）

### 1.1 已实现的模块与真实状态

| 模块 | 文件 | 现状 | 问题 |
|------|------|------|------|
| **数据生成** | `half_scale_task.py` | 已可用 | 纯CPU生成，使用 `np.memmap` 写入，符合设计规范 |
| **IO层** | `volume_source.py`, `real_input.py` | 基础可用 | 支持 `.npy`/`.raw`/`.tif` 的 memmap，但**无GPU直连** |
| **全局配准** | `global_registrar.py` | 仅CPU暴力搜索 | 三层嵌套 `for` 循环遍历 `search_radius`，**时间复杂度 O(n³)**，无GPU加速 |
| **局部精修** | `local_refiner.py` | 控制点网格+块匹配 | 同样纯CPU，控制点间距默认 `(8,8,8)`，patch半径 `(2,2,2)`，搜索半径 `(1,1,1)` |
| **融合** | `simple_stitch.py`, `brick_stitch.py` | 简单拼接 | 仅做平移拼接+均值融合，**无GPU张量操作** |
| **Benchmark** | `slice_benchmark.py` | 可用 | 统计切片读取时间，但未测GPU显存带宽 |
| **真实数据流水线** | `real_pipeline.py` | 可用 | `StreamingBrickStitcher` 实现了分块流式处理，但仍是CPU |

### 1.2 核心矛盾

**任务要求**：GPU拼接、48G显存内处理两个三维体、30Hz实时显示。  
**当前代码**：**100% NumPy CPU实现**，没有任何 `torch.Tensor` 或 CUDA Kernel。

> 原始任务数据量：`3000×1500×2000 × 2 × 4 bytes ≈ 72 GB`（单精度），远超48G。必须做**显存分块（tiling）+ 流式处理**。

---

## 二、算法层面详细建议

### 2.1 显存管理策略（最关键）

当前代码完全没有显存预算控制。建议采用 **Slab-based Streaming**：

```python
# 建议新增: src/oct_merge_task/gpu/memory_planner.py
@dataclass
class MemoryBudget:
    max_gpu_bytes: int = 40 * (1024**3)  # 40G留给算法，8G留给系统/OS
    dtype: torch.dtype = torch.float32
    
    def max_slab_shape(self, full_shape: Tuple[int,int,int]) -> Tuple[int,int,int]:
        # 以X轴为切片方向，每次只load一个slab到GPU
        bytes_per_xy_plane = full_shape[1] * full_shape[2] * 4
        max_planes = int(self.max_gpu_bytes // (bytes_per_xy_plane * 2))  # A+B两份
        slab_depth = min(full_shape[0], max(16, max_planes // 2))
        return (slab_depth, full_shape[1], full_shape[2])
```

**处理逻辑**：
1. 将 Volume A 和 Volume B 沿 X 轴切成 `slab`
2. 每次只把当前 slab + 下一个 slab 的 overlap 区域送入 GPU
3. 配准、融合、写回 brick，立即释放显存
4. 用 `torch.cuda.empty_cache()` 在 slab 边界处强制回收

### 2.2 全局配准 GPU 化（替换 `global_registrar.py`）

当前 `GlobalRegistrar._search_translation` 是 Python 三层循环，必须重写。

**建议算法**：基于 FFT 的相位相关（Phase Correlation）或 PyTorch 的 `F.conv3d`。

```python
# 建议重写为: src/oct_merge_task/registration/gpu_global_registrar.py
import torch
import torch.nn.functional as F

class GPUGlobalRegistrar:
    def __init__(self, overlap_voxels: int, device="cuda"):
        self.overlap = overlap_voxels
        self.device = device
    
    def estimate_translation(self, vol_a: torch.Tensor, vol_b: torch.Tensor) -> dict:
        # vol_a, vol_b: [D, H, W] on GPU
        # 只在 overlap 区域做互相关
        overlap_a = vol_a[-self.overlap:, :, :]
        overlap_b = vol_b[:self.overlap, :, :]
        
        # 使用 FFT-based cross-correlation (速度远优于暴力搜索)
        corr = self._phase_correlate_3d(overlap_a, overlap_b)
        peak = torch.stack(torch.where(corr == corr.max()), dim=0).squeeze()
        
        shift = peak - torch.tensor(corr.shape, device=self.device) // 2
        return {"tx": shift[0].item(), "ty": shift[1].item(), "tz": shift[2].item()}
    
    def _phase_correlate_3d(self, a, b):
        # FFT互相关，O(n log n) 复杂度
        A = torch.fft.rfftn(a)
        B = torch.fft.rfftn(b)
        cross = A * B.conj()
        cross /= (torch.abs(cross) + 1e-8)
        return torch.fft.irfftn(cross, s=a.shape)
```

**为什么选 FFT Phase Correlation**：
- 当前暴力搜索在 half-scale (`1500×750×1000`) 上，搜索半径 `(2,2,1)` 还能跑；但 full-scale (`3000×1500×2000`) 配准不可行。
- FFT 相位相关对刚性平移是标准解法，且 PyTorch 的 `torch.fft` 在 GPU 上极快。
- 对于旋转 `-4°`（见 `types.py`），先在下采样 preview 上做，再 refine。

### 2.3 局部精修 GPU 化（替换 `local_refiner.py`）

当前 `LocalRefiner` 在每个控制点做 `(3×3×3=27)` 次 patch 匹配的 Python 循环。

**建议**：改为 **Optical Flow 风格的局部变形场估计**，用 PyTorch 的 `grid_sample`。

```python
class GPULocalRefiner:
    def __init__(self, control_spacing=(64, 64, 64)):
        self.spacing = control_spacing
        
    def fit(self, ref: torch.Tensor, mov: torch.Tensor, global_tx: dict):
        # ref, mov: GPU tensors
        # 1. 在 overlap 区域提取
        tx = global_tx["tx"]
        overlap = ref.shape[0] - tx
        
        ref_overlap = ref[-overlap:, :, :]
        mov_overlap = mov[:overlap, :, :]
        
        # 2. 构建稀疏控制点网格
        grid_shape = tuple(max(2, s // sp) for s, sp in zip(ref_overlap.shape, self.spacing))
        
        # 3. 用 Lucas-Kanade 或简单的块匹配计算位移场
        # 这里建议先用最简单可工作的版本：3D Normalized Cross-Correlation via FFT
        displacement = self._sparse_ncc_flow(ref_overlap, mov_overlap, grid_shape)
        return displacement  # [3, D, H, W] flow field
    
    def _sparse_ncc_flow(self, ref, mov, grid_shape):
        # 对每个控制点，在其邻域做 NCC，返回位移向量
        # 可用 unfold + matmul 在 GPU 上并行化所有控制点
        pass
```

**关键优化点**：
- `local_refiner.py` 里的 `LocalDisplacementField.sample` 已经是三线性插值，可以直接换成 `torch.nn.functional.grid_sample`，速度提升 100x 以上。
- 控制点间距从 `(8,8,8)` 改为 `(64,64,64)` 或 `(128,128,128)`，否则控制点太多，full-scale 上不可算。

### 2.4 融合（Fusion）GPU 化

当前 `SimpleStitcher.stitch` 在 CPU 上做 `np.zeros` + 切片赋值 + 均值融合。

**建议**：在 GPU 上直接做 **加权融合（Weighted Blending）**：

```python
class GPUFusion:
    def stitch(self, vol_a: torch.Tensor, vol_b: torch.Tensor, 
               tx: int, flow_field=None) -> torch.Tensor:
        # 创建输出张量
        out_shape = (
            max(vol_a.shape[0], tx + vol_b.shape[0]),
            max(vol_a.shape[1], vol_b.shape[1]),
            max(vol_a.shape[2], vol_b.shape[2])
        )
        output = torch.zeros(out_shape, device=vol_a.device, dtype=torch.float32)
        weight = torch.zeros_like(output)
        
        # 写入 A
        output[:vol_a.shape[0], :, :] += vol_a
        weight[:vol_a.shape[0], :, :] += 1.0
        
        # 写入 B（如有 flow_field 先 warp）
        if flow_field is not None:
            mov_warped = self._warp_volume(vol_b, flow_field)
        else:
            mov_warped = vol_b
            
        output[tx:tx+vol_b.shape[0], :, :] += mov_warped
        weight[tx:tx+vol_b.shape[0], :, :] += 1.0
        
        # 归一化
        return output / weight.clamp(min=1e-6)
    
    def _warp_volume(self, vol, flow):
        # flow: [3, D, H, W]
        # 使用 grid_sample 做 3D 变形
        grid = self._flow_to_grid(flow)
        return F.grid_sample(vol[None,None,...], grid[None,...], 
                            mode='bilinear', padding_mode='zeros', align_corners=True)
```

### 2.5 30Hz 实时显示架构

当前完全没有渲染器。要实现 30Hz，必须：

1. **Brick + Mipmap 预计算**：拼接完成后，生成多分辨率 brick（类似 `VolumeStore.build_pyramid`，但要在 GPU 上做）。
2. **按需切片抽取**：显示器只有 1080p，不需要全分辨率。用户查看时，从 brick 中读取当前视口对应的切片。
3. **双缓冲 + CUDA-OpenGL Interop**（可选，高级）或 **PyTorch → NumPy → OpenCV 显示**（简单版）。

**建议实现路径**：
- 第一阶段（汇报可用）：用 `matplotlib` 或 `napari` 做切片浏览，不追求 30Hz，证明算法正确。
- 第二阶段：用 `pygfx` / `vispy` / 自定义 WebGL（`src/oct_merge_task/web/` 已有目录）做 30Hz 渲染。

---

## 三、汇报包（Report Bundle）详细建议

当前 `report_bundle/` 下有 `quarter_scale_task_bundle` 和 `real_task_oct_merge_bundle`，但内容未读取到。建议按以下结构组织：

### 3.1 汇报包目录结构

```
report_bundle/
├── 01_algorithm_design/          # 算法设计文档
│   ├── memory_budget_analysis.md  # 显存预算计算
│   ├── registration_pipeline.md   # 配准流程图
│   └── fusion_strategy.md         # 融合策略说明
├── 02_implementation/            # 实现细节
│   ├── gpu_kernels.md             # CUDA/PyTorch 算子说明
│   ├── tiling_strategy.md         # 分块策略
│   └── benchmark_results.md       # 性能基准测试
├── 03_validation/                # 验证
│   ├── half_scale_results/        # Half-scale 可视化结果
│   │   ├── slice_comparison.png   # A/B/Stitched 对比
│   │   └── displacement_field.png # 位移场可视化
│   └── metrics.json               # NCC score, MSE, SSIM
├── 04_demo/                      # 可运行演示
│   ├── run_half_scale_demo.py     # 一键运行 half-scale
│   ├── run_real_data_demo.py      # 真实数据入口
│   └── requirements.txt
└── README.md                      # 总览
```

### 3.2 关键汇报内容建议

#### （1）显存预算表（必须放在汇报里）

| 数据阶段 | 尺寸 | dtype | 显存占用 | 说明 |
|---------|------|-------|---------|------|
| Volume A (full) | 3000×1500×2000 | float32 | 36 GB | 无法全驻留 |
| Volume B (full) | 3000×1500×2000 | float32 | 36 GB | 无法全驻留 |
| Slab A (streaming) | 256×1500×2000 | float32 | 3.0 GB | 单次 slab |
| Slab B (streaming) | 256×1500×2000 | float32 | 3.0 GB | 单次 slab |
| Overlap corr buffer | 256×1500×2000 (complex) | complex64 | 6.0 GB | FFT 缓冲区 |
| Displacement field | 1500×750×1000 × 3 | float32 | 13.5 GB | 仅在 half-scale |
| **Total (streaming)** | - | - | **~12 GB** | **远小于48G，安全** |

> 结论：采用 Slab-based Streaming 后，full-scale 任务可在 48G 显存内完成。

#### （2）算法流程图（用于汇报 PPT）

```
[Disk: volume_a.raw] ──► [CPU Memmap] ──► [GPU Slab Loader]
                                      │
[Disk: volume_b.raw] ──► [CPU Memmap] ──► [GPU Slab Loader]
                                      │
                                      ▼
                         [GPU Global Reg: FFT Phase Corr]
                                      │
                                      ▼
                         [GPU Local Refine: Sparse NCC Flow]
                                      │
                                      ▼
                         [GPU Fusion: Weighted Blending]
                                      │
                                      ▼
                         [GPU Brick Writer] ──► [Disk Bricks]
                                      │
                                      ▼
                         [Benchmark: Slice FPS]
```

#### （3）Benchmark 必须包含的指标

当前 `slice_benchmark.py` 只测了磁盘读取时间。汇报时必须补充：

```python
# 建议新增到 benchmark 脚本
metrics = {
    "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    "registration_time_s": reg_time,
    "fusion_time_s": fusion_time,
    "slice_read_fps": 1000.0 / mean_slice_ms,
    "ncc_score": ncc,
    "output_shape": stitched_shape,
    "brick_count": brick_count,
}
```

---

## 四、给 GPT-5.4 的开发指令建议

如果你要用 GPT-5.4 继续开发，建议按以下顺序给它下指令（每步一个独立任务）：

### Step 1: GPU 化全局配准
> "Rewrite `global_registrar.py` to use PyTorch FFT-based phase correlation on CUDA. The input should be `torch.Tensor` slabs, not full volumes. Keep the same interface but add a `device='cuda'` parameter. Remove all Python for-loops for searching."

### Step 2: GPU 化局部精修
> "Replace `local_refiner.py` with a GPU version using `torch.nn.functional.grid_sample` for displacement field warping. Use a coarser control point spacing (64,64,64) for full-scale. The `LocalDisplacementField` should be a `torch.Tensor` of shape `[3, D, H, W]`."

### Step 3: GPU 融合 + Slab Streaming
> "Create `gpu_fusion.py` that implements slab-based streaming stitch. It should process a pair of 3000×1500×2000 volumes in chunks of 256×1500×2000 to stay under 48GB VRAM. Use weighted averaging in the overlap region."

### Step 4: 30Hz 预览渲染器
> "Implement a simple slice renderer in `src/oct_merge_task/web/` using WebGL or PyQtGraph that can browse stitched bricks at 30 FPS. It should read bricks from disk on-demand and support zoom/pan for 1080p display."

### Step 5: 端到端测试
> "Write an integration test that runs the full pipeline (generate half-scale case → GPU register → GPU fuse → brick store → benchmark) and asserts peak GPU memory < 40GB and slice read FPS > 30."

---

## 五、当前代码的具体修复点

最后，基于我读到的代码，有几个**必须立即修复的 bug/问题**：

1. **`global_registrar.py` 的 `axis` 参数未使用**：`estimate_multiscale` 接收了 `axis` 但 `_search_translation` 里硬编码了 X 轴逻辑。如果未来支持 Y/Z 轴拼接，这里会出错。

2. **`simple_stitch.py` 的 `local_field` 应用逻辑错误**：
   ```python
   z_shift = int(round(float(displacement[..., 2].mean())))
   source[:, :, z] = np.roll(source[:, :, z], shift=z_shift, axis=1)
   ```
   这里只做了 Z 方向的 `np.roll`，但 `LocalDisplacementField` 实际上可能包含 X/Y/Z 三个方向的位移。应该使用完整的 3D warp。

3. **`half_scale_task.py` 的 `_rotate_volume_z_local` 使用了 `torch` 但没有 import 检查**：该函数内部 `import torch`，但如果在无 GPU 机器上运行会失败。建议加上 fallback。

4. **缺少 `pyproject.toml` 或 `setup.py`**：作为一个 package，当前只有目录结构，没有安装配置。建议添加：
   ```toml
   [project]
   name = "oct-merge-task"
   dependencies = ["numpy", "torch", "tifffile"]
   ```

