# Architecture & Code Layout

## Overview

Stiff-GIPC is a GPU-first IPC simulator. The top-level `GIPC` class
(`StiffGIPC/GIPC.cu` / `GIPC.cuh`) owns the Newton solve loop, collision
detection, barrier/friction energies, and the linear system. Around it sit
extracted subsystems for affine-body dynamics, the linear system / PCG solver,
and an application layer that separates headless benchmarking from the OpenGL
viewer.

## Directory Map

```
StiffGIPC/                 # Core simulation library (compiled into stiffgipc_core)
├── GIPC.cu / GIPC.cuh     # Main solver: Newton loop, CCD, barrier, line search
├── GIPC_PDerivative.cuh   # Autodiff-style Hessian/gradient kernels (large, ~215KB)
├── femEnergy.cu/.cuh      # FEM elasticity + bending energy kernels
├── ACCD.cu/.cuh           # Adaptive collision detection (ACCD)
├── mlbvh.cu/.cuh          # Morton-code BVH for collision queries
├── MASPreconditioner.cu   # MAS (mass-aggregated) preconditioner, GPU + CEMAS paths
├── FrictionUtils.cuh      # Friction tangent basis / lagrange update
├── gpu_eigen_libs.cu/.cuh # Small GPU linear algebra helpers (mat3/mat3x2)
├── QRSVD.hpp / givens.hpp # QR-based SVD used by ARAP-style energies
├── load_mesh.cpp/.h       # Tet mesh loading, material assignment, mesh sorting
├── gl_main.cu             # Viewer entrypoint (only built with BUILD_GIPC_VIEWER)
├── gipc/                  # Runtime config, statistics, type definitions
├── linear_system/         # Linear system, PCG, preconditioners, SpMV  (see below)
├── abd_system/            # Affine Body Dynamics system + energy/Jacobian
├── cuda_tools/            # Device buffer RAII helpers
└── muda/                  # Vendored CUDA framework (fork of MuGdxy/muda)

app/                       # Application layer (entrypoints + shared bootstrap)
├── common/                # Shared by viewer and bench
│   ├── scene_registry.*  # Builtin scene setup + dataset scene loading
│   ├── sim_bootstrap.*   # CUDA init, FEM init, host->device upload
│   └── sim_settings.*    # parameterSetting.txt parser -> GIPC fields
├── bench/                 # Headless benchmark target (gipc_bench)
│   ├── main.cpp          # CLI parsing (--suite / --scene / single-run)
│   ├── benchmark_config.*# Suite + run config, JSON/manifest loading
│   ├── benchmark_runner.*# Drives warmup + measured frames, writes reports
│   └── benchmark_report.*# raw.json / summary.csv / meta.json emission
└── viewer/                # Viewer target (gipc_viewer), thin wrapper around gl_main

MeshProcess/               # Mesh processing + METIS partitioning (add_subdirectory)
Assets/                    # Scene files, tet/tri meshes, sorted meshes (runtime data)
datasets/                  # Experiment datasets + bench suite JSONs (see experiments.md)
scripts/                   # Python analysis + visualization (stdlib only)
docs/                      # Design notes, experiment plans (mostly local, gitignored)
analysis/                  # Generated analysis reports + figures + JSON
Output/                    # All run output (gitignored)
```

## The `gipc/` Subdirectory

Holds runtime-switchable configuration that decouples benchmark selection from
the legacy solver internals.

- `runtime_config.h/.cpp` — `BenchmarkBaseline` enum, `RuntimeBackendConfig`
  struct, parsers, and `apply_preconditioner_config_from_json()`. This is the
  single source of truth for backend selection. See
  [backends.md](backends.md).
- `statistics.h/.cpp` — `gipc::Statistics::instance()` singleton accumulating
  per-frame timing/iteration counters into a JSON object.
- `gipc.h/.cu` — thin facade; `type_define.h`, `body_type.h`,
  `abd_fem_count_info.h`, `tet_local_info.h` — shared POD structs.

## Linear System Stack (`StiffGIPC/linear_system/`)

This is the primary target for preconditioner research. Layout:

```
linear_system/
├── linear_system/         # Top-level orchestration
│   ├── global_linear_system.*   # Assembles + solves the full block system
│   ├── global_matrix.h          # GIPCTripletMatrix / block-sparse storage
│   ├── i_linear_system_solver.h # IterativeSolver interface
│   ├── i_preconditioner.h       # IPreconditioner interface
│   └── linear_subsystem.*       # Per-DoF-type subsystem (fem/abd)
├── solver/
│   └── pcg_solver.*             # PCG with sub-phase timing (Precond/SpMV/Dot/Axpby)
├── preconditioner/
│   ├── fem_mas_preconditioner.* # FEM mass-aggregated Schwarz
│   ├── abd_preconditioner.*     # Affine-body preconditioner
│   └── diag_preconditioner.*    # Diagonal fallback
├── subsystem/
│   ├── fem_linear_subsystem.*   # Assembles FEM block rows/cols
│   └── abd_linear_subsystem.*   # Assembles ABD block rows/cols
└── utils/
    ├── converter.*              # Triplet -> CSR/BSR, legacy<->SRBK format
    └── spmv.*                   # SpMV backends (LegacyGipc vs SRBK)
```

Key design notes:

- The backend matrix format (Legacy triplet vs SRBK block-sparse) is selected at
  runtime via `RuntimeBackendConfig.assembly_backend` / `spmv_backend`, not via
  compile-time macros. Do not introduce `#ifdef` forks here.
- `PCGSolver` exposes `set_tolerance()` to allow dynamic per-Newton-step CG
  tolerance (used by the inexact-Newton preconditioner flag).
- PCG sub-phase timers accumulate without `cudaDeviceSynchronize` to avoid
  per-iteration sync cost; values are read out at frame boundaries.

## The `GIPC` Class

`GIPC` is a monolithic class (~250 fields in the header). Important members:

- `runtime_backend_config` — drives assembly/SpMV/MAS backend selection.
- `benchmark_mode`, `verbose_output`, `write_statistics_file`,
  `write_legacy_time_cost_file` — output toggles, set by `benchmark_runner`.
- `IPC_Solver(device_TetraData&)` — top-level per-frame entrypoint.
- `solve_subIP(...)` — one Newton step; returns the `time0..time4` segment
  timings that map to the benchmark metrics (see
  [experiments.md](experiments.md)).
- `m_abd_system`, `m_global_linear_system` — owned subsystems constructed in
  `create_LinearSystem()` / `init_abd_system()`.

When adding new solver behavior, prefer extending `RuntimeBackendConfig` and the
`linear_system/` interfaces rather than adding new branches inside `GIPC.cu`.

## Include Conventions

- Public headers in `StiffGIPC/` are included as `<GIPC.cuh>`,
  `<device_fem_data.cuh>`, etc. (the `StiffGIPC/` dir is a PUBLIC include path).
- `gipc/` headers are included as `<gipc/runtime_config.h>`.
- `app/` headers use project-relative paths: `"app/bench/benchmark_runner.h"`.
- `linear_system/` headers use nested paths:
  `<linear_system/linear_system/global_linear_system.h>`,
  `<linear_system/solver/pcg_solver.h>`.
