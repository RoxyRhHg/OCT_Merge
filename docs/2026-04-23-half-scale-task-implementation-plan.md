# Half-Scale Task Implementation Plan

Date: 2026-04-23

## Milestone 1: Project Foundation

Goal:

- establish new package structure
- define reusable dataclasses
- implement half-scale case generation and preview workflow
- cover with tests

Deliverables:

- `src/oct_merge_task/types.py`
- `src/oct_merge_task/tools/half_scale_task.py`
- `tests/test_half_scale_task.py`

Acceptance criteria:

- task-sized case generation writes full-resolution arrays to disk
- preview pipeline can run from existing generated case

## Milestone 2: Real Input Path

Goal:

- support direct use of real or realistic volume inputs

Deliverables:

- `src/oct_merge_task/io/real_input.py`
- `src/oct_merge_task/tools/custom_pipeline.py`
- `scripts/run_custom_stitch.py`
- tests for `.npy`, `.tiff`, `.raw`, and slice-folder input

Acceptance criteria:

- at least three input forms feed through the same stitch pipeline

## Milestone 3: Benchmark Tools

Goal:

- measure the stitched display path quantitatively

Deliverables:

- `src/oct_merge_task/tools/slice_benchmark.py`
- `src/oct_merge_task/tools/benchmark_sweep.py`
- corresponding scripts and tests

Acceptance criteria:

- emits JSON summaries with slice time, estimated fps, and cache stats

## Milestone 4: Browser-Facing Output Contract

Goal:

- define a clean output model for the future browser UI

Deliverables:

- run metadata
- brick layout metadata
- benchmark summaries
- transform summaries

Acceptance criteria:

- browser UI can later load these outputs without rerunning the algorithm

## Immediate Coding Slice

For the first pass in this new project:

1. create basic package files
2. write failing tests for the first milestone
3. implement the smallest working half-scale task generation and preview workflow
4. run targeted tests

## Implementation Notes

- keep this project independent from the earlier demo-focused project
- reuse only ideas, not tangled UI assumptions
- prefer disk-backed outputs over in-memory convenience
