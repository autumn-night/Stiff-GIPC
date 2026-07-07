# Stiff-GIPC

GPU Incremental Potential Contact (IPC) framework for stiff affine-deformable
simulation (cloth, elastic solids, rigid bodies, and hybrids). C++17 / CUDA 17
research codebase; ACM Transactions on Graphics, 2025.

## Working Goal & Repository Ownership

The user's objective is **algorithm-level optimization of the
`StiffGIPC_CEMAS_SRBK` baseline** (the default and most complete stack). The
current phase is **bottleneck analysis** of the linear system / PCG /
preconditioner path. Treat changes to the upstream solver internals
(`StiffGIPC/GIPC.cu`, `GIPC_PDerivative.cuh`, `femEnergy.cu`, `MASPreconditioner.cu`,
`linear_system/`) as high-risk and prefer additive, runtime-switchable hooks.

The following directories are the user's own work layered on top of the
upstream codebase — not part of the published StiffGIPC release:

- **`app/`** — the headless benchmark harness (`gipc_bench`) and shared
  bootstrap/scene/settings logic the user wrote to evaluate bottlenecks.
- **`datasets/`** — data the user collected. Subdirectories prefixed `exp_`
  (e.g. `exp_0528`, `exp_0615`) are the user's own experiments; each holds
  bench-suite JSONs and per-variant `settings/` files. `stanford/` and other
  non-`exp_` entries are imported reference meshes/scenes.
- **`Output/`** — experiment results produced by `gipc_bench` (gitignored).

Everything else under `StiffGIPC/` and `MeshProcess/` is upstream solver code.

## Quick Reference

- **Language:** C++17 + CUDA 17, CMake (>= 3.18), NVIDIA GPU required
- **No CUDA in this environment:** code cannot be built or run here. Builds
  happen on a CUDA-equipped host; this checkout is for editing/analysis only.
- **Dependencies:** CUDA >= 11, Eigen3 3.4.0, nlohmann-json, METIS, muda
  (vendored under `StiffGIPC/muda/`). freeglut/GLEW only for the viewer.
- **Build (headless):** `cmake -S . -B build -DBUILD_GIPC_VIEWER=OFF && cmake --build build`
- **Build (full):** `cmake -S . -B build && cmake --build build`
- **Targets:** `stiffgipc_core` (lib), `gipc_bench` (headless benchmark),
  `gipc_viewer` (OpenGL viewer, optional).
- **Bench run:** `./build/gipc_bench --suite datasets/exp_0528/bench_sweep_stiffness.json`
  or `./build/gipc_bench --scene cloth_bunny --baseline StiffGIPC_CEMAS_SRBK`

## Critical Rules

- Three baselines are fixed and must stay runtime-switchable, never collapsed
  into one compile-time path: `GIPC`, `StiffGIPC_SRBK`, `StiffGIPC_CEMAS_SRBK`
  (default). See [backends guide](docs/agent-guides/backends.md).
- Scene/step/tolerance/material parameters must be identical across baselines
  in any benchmark comparison.
- Output and `*.md` files are gitignored. Force-add (`git add -f`) anything that
  must be shared, or add a gitignore exception.

## Detailed Guides

- [Architecture & Code Layout](docs/agent-guides/architecture.md) - directory
  map, core subsystems, linear system / preconditioner structure.
- [Build & Benchmark Workflow](docs/agent-guides/build-and-bench.md) - build
  options, `gipc_bench` CLI, suite JSON format, output layout.
- [Backends & Runtime Config](docs/agent-guides/backends.md) - baseline matrix,
  `RuntimeBackendConfig`, preconditioner-improvement flags, compile defines.
- [Experiments & Analysis](docs/agent-guides/experiments.md) - `exp_XXXX`
  organization, metrics, Python analysis scripts.
- [Code Style & Conventions](docs/agent-guides/code-style.md) - clang-format,
  naming, header/include rules, git workflow.
