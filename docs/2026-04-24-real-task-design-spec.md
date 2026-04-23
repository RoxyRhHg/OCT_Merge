# Real Task Design Spec

Date: 2026-04-24

## 1. Goal

Refocus the project from a quarter-scale demonstration into a real-task-oriented OCT merge system that is explicitly designed for:

- input volumes near `3000 x 1500 x 2000`
- overlap around 10 percent, but not known exactly in advance
- a single 48GB GPU budget
- a development path toward GPU registration, GPU-aware streaming fusion, and eventually interactive viewing

This spec has two layers:

1. the full long-term architecture for the real task
2. the scoped version that must be delivered today for live reporting

The two layers must stay aligned. The reporting build should be a real stage on the path to the final system, not a disconnected presentation-only artifact.

## 2. Problem Constraints

The original task requires merging two very large OCT volumes.

Approximate memory facts:

- one `3000 x 1500 x 2000` volume in `float32` is about 36GB
- two such full volumes in `float32` are about 72GB
- a naive full-volume GPU workflow exceeds a 48GB budget before accounting for buffers, transforms, or output

Therefore the system must not rely on:

- fully materializing both input volumes as dense `float32` tensors
- full stitched output tensors held in memory or VRAM
- brute-force Python loops over full-scale data

The system must rely on:

- memmap or disk-backed input access
- chunked or slab-based GPU staging
- brick-based output writing
- explicit memory budgeting

## 3. Current Codebase Reality

The existing repository already has useful starting points:

- real input reading with memmap-style access via `volume_source.py`
- a real-data-oriented streaming stitch path in `real_pipeline.py`
- brick-backed outputs and slice benchmarks
- bundle generation and smoke-test packaging

However, the key limitation remains:

- the algorithmic core is still mainly CPU-based
- the global registration path is still not GPU-first
- local refinement is not trustworthy enough to present as a final real-task solution
- the current benchmark does not prove a final 30Hz renderer

This spec treats the current system as a good real-task foundation, but not as a completed delivery.

## 4. Design Principles

1. Real-task scale is the reference, not quarter-scale convenience.
2. Every major stage must be compatible with out-of-core execution.
3. GPU effort should be focused on the highest-leverage stages first.
4. Reporting artifacts must reflect real engineering progress, not decorative mockups.
5. Today's build must be honest about current boundaries.

## 5. Full Long-Term Architecture

The intended end-to-end architecture is:

`VolumeSource -> MemoryPlanner -> Preview/Overlap Estimation -> GPU Global Registration -> Streaming Brick Stitch -> Benchmark -> Interactive Viewer`

### 5.1 VolumeSource Layer

Responsibilities:

- open `.npy`, `.raw`, `.tif`, `.tiff` inputs
- avoid eager full-volume float32 conversion
- expose shape, dtype, and region reads

Requirements:

- input arrays remain disk-backed where possible
- A/B inputs must currently agree on the non-stitched axes
- region reads must remain the primary data access pattern

This layer already exists in basic form and should remain the input backbone.

### 5.2 MemoryPlanner Layer

New module to add:

- `src/oct_merge_task/gpu/memory_planner.py`

Responsibilities:

- calculate safe slab shapes from a VRAM budget
- account for input staging plus temporary buffers
- define conservative working sizes for registration and fusion

Design rule:

- memory planning must come from budgeted shapes and bounded tensor lifetimes
- it must not depend on calling `torch.cuda.empty_cache()` as a primary control mechanism

### 5.3 Overlap Estimation Layer

Responsibilities:

- estimate real overlap within a prior range such as 5 to 20 percent
- avoid assuming a fixed 10 percent overlap
- operate on downsampled data or overlap ROIs

Current status:

- the existing real pipeline already searches overlap in a bounded range

Long-term direction:

- keep overlap estimation cheap and robust
- make its output feed directly into GPU global registration

### 5.4 GPU Global Registration Layer

This is the first priority GPU module.

New module to add:

- `src/oct_merge_task/registration/gpu_global_registrar.py`

Stage-1 scope:

- support `axis=0` only
- estimate translation in overlap ROI
- use PyTorch on CUDA
- prefer FFT phase correlation over brute-force search
- keep a CPU fallback for environments without GPU

Why this stage comes first:

- it directly addresses the main bottleneck in the current CPU design
- it demonstrates credible movement toward real-task GPU handling
- it is easier to scope cleanly than dense local refinement

Non-goals for this stage:

- full rigid rotation estimation
- multi-axis arbitrary stitching
- full nonrigid warping

### 5.5 Local Refinement Layer

This remains a second-stage module, not today's primary deliverable.

Current problems:

- the existing CPU implementation is too dense for full-scale use
- the current stitch-time use of `local_field` is not strong enough to present as a final 3D warp solution

Long-term direction:

- restrict refinement to overlap ROI
- use sparse control points rather than dense full-scale flow
- eventually migrate interpolation and warping toward GPU operations such as `grid_sample`

Design choice:

- today's reporting build should treat local refinement as optional or experimental
- it should not be central to the claim of real-task readiness

### 5.6 Streaming Fusion Layer

Responsibilities:

- stitch output incrementally
- avoid full stitched-volume allocation
- write output as bricks
- be ready to move overlap fusion logic to GPU

Design rule:

- the real-task path must remain brick-first and streaming-first
- it must not create a dense final stitched tensor on GPU just for convenience

Current status:

- `real_pipeline.py` already has a useful streaming brick stitch pattern

Long-term direction:

- preserve the current streaming structure
- move local overlap computation toward torch-based kernels where useful
- keep output immediately writeable to brick storage

### 5.7 Benchmark Layer

Responsibilities:

- measure real brick reads and slice assembly
- report real timings rather than synthetic placeholders
- later include GPU-side metrics

Required metrics for the real-task path:

- overlap estimate
- stitched output shape
- brick count
- disk reads
- mean slice milliseconds
- estimated slice FPS

Future GPU metrics:

- peak GPU memory
- registration time
- fusion time

### 5.8 Viewer Layer

This is explicitly split into two stages.

Stage A:

- practical slice viewer or lightweight results browser
- suitable for development and reporting

Stage B:

- a true interactive viewer designed toward a 30Hz experience
- brick-aware, on-demand, resolution-aware rendering

Today's scope is Stage A only.

The project must not claim a completed 30Hz renderer yet.

## 6. Immediate Code Corrections Required

The following issues should be fixed early because they affect credibility and correctness.

### 6.1 `global_registrar.py` axis semantics

The current interface suggests an axis parameter, but the logic effectively assumes axis 0.

Action:

- either explicitly constrain the current CPU registrar to axis 0
- or implement axis-aware behavior properly

For today's reporting version, explicit constraint is preferred over pretending broader support.

### 6.2 `simple_stitch.py` local-field logic

The current local-field usage is not a reliable 3D warp implementation.

Action:

- do not present it as a final local deformation solution
- either simplify its role or mark it as preview-only behavior

### 6.3 Packaging metadata

The project currently needs a minimal package definition.

Action:

- add `pyproject.toml`
- declare core dependencies such as `numpy`, `torch`, and `tifffile`

This improves reproducibility and makes the real-task bundle more credible.

## 7. Today's Reporting Build

Today's deliverable is not the final system. It is a reportable real-task build with real execution value.

### 7.1 Reporting Build Goals

The reporting build must show:

- real-task scale awareness
- a real input path
- real streaming and brick output
- at least one major GPU-oriented algorithmic step clearly under active implementation
- real benchmark outputs
- honest boundaries

It must not rely on fake performance claims or presentation-only effects.

### 7.2 Reporting Build Scope

Today's reporting build should include:

- updated real-task design
- memory-planning-aware architecture
- first-priority GPU registration work
- tightened CPU fallback behavior
- a minimal real-task bundle for direct demonstration

### 7.3 Reporting Bundle Structure

The reporting bundle must be minimal.

It should contain only:

- `src/`
- `real_data/`
- `scripts/`
- `report/`
- `results/` or `smoke_run/`
- `README.txt`

This bundle is meant to be easy to hand to a teacher and easy to demonstrate from a single folder.

It should not include a large documentation tree beyond what is strictly necessary.

### 7.4 Reporting Bundle Content Rules

Inside the reporting bundle:

- `src/` contains the actual runnable source
- `real_data/` explains where to place real input volumes
- `scripts/` contains only the entry scripts needed for inspection, running, and demo
- `report/` contains a live presentation page for direct explanation
- `results/` or `smoke_run/` contains immediately showable outputs
- `README.txt` contains only the shortest operational guidance

The reporting page must explain:

- task scale and VRAM constraint
- current pipeline
- what is already real
- how to run on real inputs
- what remains incomplete

## 8. Delivery Message for Today

The correct reporting message is:

- the project is now organized around the real task, not a toy preview
- real inputs can already be attached to a streaming pipeline
- the system already avoids naive full-volume residency
- GPU handling is being introduced where it matters most
- the current build is a real engineering stage, not the final 30Hz GPU end state

The incorrect message would be:

- that the final GPU real-time system is already complete

## 9. Implementation Order After Spec Approval

The first implementation slice after this spec should be:

1. add packaging metadata
2. tighten current interface boundaries and preview-only behavior
3. add `MemoryPlanner`
4. implement `GPUGlobalRegistrar` first version
5. connect GPU-ready registration into the real-task path
6. simplify and rebuild the reporting bundle around the real-task deliverable
7. refresh smoke-test outputs and live report page

## 10. Success Criteria

This stage is successful if:

- the design clearly centers the original real task
- the codebase is reorganized toward GPU-capable handling rather than preview-only behavior
- the reporting bundle is minimal, runnable, and teacher-friendly
- the reporting build can honestly demonstrate real input handling, streaming output, benchmark results, and the GPU-oriented direction

## 11. Explicit Non-Goals for Today

These are not required today:

- full nonrigid GPU deformation
- final 30Hz production viewer
- full multi-axis stitching support
- claiming completion of the original task

Today's goal is a credible, minimal, real-task-facing stage that can be shown live and extended directly.
