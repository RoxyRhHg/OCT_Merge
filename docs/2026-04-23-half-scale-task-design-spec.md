# Half-Scale Task Design Spec

Date: 2026-04-23

## 1. Goal

Build a fresh OCT stitching project focused on the algorithmic architecture rather than presentation polish.

The first target is a half-scale simulation of the original task:

- full target task: `3000 x 1500 x 2000`
- half-scale task target: `1500 x 750 x 1000`

The project must prioritize:

1. memory-aware volume handling
2. overlap-driven rigid registration
3. local residual correction in the overlap band
4. brick-based stitched output
5. measurable slice-display performance

The browser-based presentation layer is a second-stage deliverable and must be built on top of the algorithmic outputs rather than driving the architecture.

## 2. Core Constraints

The full task volume is too large for naive dense float32 residency. Even the half-scale simulation is large enough to expose memory-management issues. Therefore:

- full-resolution half-scale arrays must not be freely duplicated
- disk-backed generation and brick-oriented processing are required
- preview-level runs are allowed for algorithm verification
- full-resolution half-scale generation is used to validate task-scale data flow

## 3. Scope of This New Project

Included:

- half-scale case generation
- preview-scale registration and fusion
- disk-backed stitched bricks
- benchmark tools for slice reading
- real-input-ready architecture

Excluded from the first phase:

- polished browser UI
- final 30 Hz production renderer
- full-resolution half-scale dense registration

## 4. Project Principles

1. The algorithm is the primary product.
2. Every stage must leave reusable outputs on disk.
3. Half-scale full-resolution data is for task-scale realism.
4. Preview data is for algorithm execution and iteration speed.
5. Later browser visualization must consume saved outputs rather than re-implement the pipeline.

## 5. Target Workflow

The intended workflow is:

1. generate a half-scale task-sized case on disk
2. generate preview volumes from that case
3. run rigid registration on preview volumes
4. run local refinement on preview overlap
5. generate stitched bricks
6. benchmark slice reading and cache behavior
7. feed those outputs into a browser-based visualization layer later

## 6. Modules

### 6.1 Case Generation

Responsibilities:

- create `volume_a.npy` and `volume_b.npy` at half-scale task size
- create `preview_volume_a.npy` and `preview_volume_b.npy`
- persist metadata to disk

Requirements:

- full-resolution case creation must be memory-aware
- preview generation must be deterministic

### 6.2 Volume IO

Responsibilities:

- memmap-backed volume access
- slab and brick reads
- future extension to `.npy`, `.tiff`, `.raw`, and slice-folder inputs

### 6.3 Global Registration

Responsibilities:

- estimate rigid transform on preview volumes
- use coarse-to-fine overlap-driven search

### 6.4 Local Refinement

Responsibilities:

- estimate residual deformation inside overlap only
- use sparse low-freedom correction

### 6.5 Brick Fusion

Responsibilities:

- generate stitched output in bricks
- support disk-backed storage

### 6.6 Benchmarking

Responsibilities:

- benchmark brick-backed slice reading
- benchmark cache and prefetch settings
- emit JSON summaries suitable for reports

## 7. Deliverables of Phase 1

Phase 1 must produce:

- task-sized case generation script
- preview pipeline script
- custom input stitching script
- slice benchmark script
- benchmark sweep script
- tests covering those tools

## 8. Success Criteria

Phase 1 is considered successful if:

- half-scale full-resolution case files can be generated to disk
- preview pipeline can run independently from an existing case
- stitched output is persisted as bricks
- slice benchmark outputs quantitative metrics
- real input paths can be stitched through the same pipeline

## 9. Risks

Primary risks:

- full-resolution case generation may become too slow if implemented with dense temporary arrays
- rotation or local transform applied at full task scale may become impractical on CPU
- preview may diverge too much from full-scale behavior if downsampling is not chosen carefully

## 10. Mitigations

- use memmap-backed writes for full-resolution generation
- confine heavy geometric transforms to preview-level execution
- keep metadata linking full-scale and preview-scale outputs
- benchmark every performance-critical step

## 11. Next Phase After Algorithm Core

Once the algorithm workflow is stable:

- build a local browser-based visualization client
- make particle-heavy stitched rendering the primary visual
- keep auxiliary slice panels for analysis
- aim toward responsive interactive browsing informed by benchmark results
