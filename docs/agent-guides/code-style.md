# Code Style & Conventions

## Formatting

Enforced by `.clang-format` (LLVM-based). Run `clang-format -i <file>` before
committing. Key settings:

| Setting | Value |
|---------|-------|
| `ColumnLimit` | 80 |
| `IndentWidth` / `TabWidth` | 4, `UseTab: Never` |
| `BreakBeforeBraces` | Custom — Allman style (braces on new line for class, function, namespace, control, struct, enum; `BeforeCatch`/`BeforeElse` true) |
| `PointerAlignment` | Left (`int* p`, not `int *p`) |
| `SortIncludes` | false (preserve manual order) |
| `AlignConsecutiveAssignments` / `AlignConsecutiveDeclarations` | true |
| `NamespaceIndentation` | Inner |
| `AllowShortFunctionsOnASingleLine` | Inline only |
| `Standard` | Cpp11 |

`.editorconfig` sets UTF-8 for `*.{c,cpp,cu,h,hpp,hxx}`.

## Naming

The codebase mixes a legacy style (in `StiffGIPC/GIPC.cu` and older kernels)
with a newer style (in `app/`, `gipc/`, `linear_system/`). Follow the newer
style for new code; do not mass-rename legacy identifiers.

| Element | Convention | Examples |
|---------|-----------|----------|
| Classes / structs / enums | `PascalCase` | `GIPC`, `GlobalLinearSystem`, `RuntimeBackendConfig`, `BenchmarkBaseline` |
| Enumerators | `PascalCase` with `UPPER` tokens | `StiffGIPC_CEMAS_SRBK`, `LegacyGipc` |
| Namespaces | `lowercase` | `gipc`, `app::bench`, `app::common` |
| Free functions (new code) | `snake_case` | `load_benchmark_suite`, `resolve_runtime_backend_config`, `is_builtin_scene` |
| Methods (legacy `GIPC`) | `PascalCase` / `camelCase` | `IPC_Solver`, `buildBVH`, `computeGradientAndHessian` |
| Member variables | mixed — prefer `snake_case_` trailing underscore or `m_` prefix for new code | `m_config`, `m_time_spmv`, `runtime_backend_config` |
| Files | lowercase, extension denotes kind | `.cu`/`.cuh` (CUDA), `.cpp`/`.h` (host), `.hpp`/`.inl` (header-only templates) |

## Headers & Includes

- Prefer `#pragma once`. Legacy headers use `#ifndef _FOO_H_` guards; leave
  them unless rewriting the file.
- Include order (manual, since `SortIncludes` is off): own header, project
  headers, third-party, standard library. Example from
  `app/bench/benchmark_runner.cpp`:
  ```cpp
  #include "app/bench/benchmark_runner.h"
  #include "app/common/sim_bootstrap.h"
  #include <GIPC.cuh>
  #include <device_fem_data.cuh>
  #include <gipc/statistics.h>
  #include <algorithm>
  #include <filesystem>
  ```
- Public headers under `StiffGIPC/` are included with angle brackets and no
  directory prefix (`<GIPC.cuh>`) because `StiffGIPC/` is a PUBLIC include
  path. `gipc/` headers use `<gipc/...>`. `app/` and `linear_system/` headers
  use their full relative path. See [architecture.md](architecture.md#include-conventions).

## CUDA Conventions

- CUDA 17; `--extended-lambda`, `--expt-relaxed-constexpr`,
  `--default-stream=per-thread` are PUBLIC compile options on `stiffgipc_core`.
- `--use_fast_math` and `-lineinfo` are enabled for release builds.
- Device kernels live in `.cu` files; host-callable inline helpers in `.cuh`.
- Prefer `muda` abstractions (`muda::DeviceDenseVector`, views) in new
  `linear_system/` code over raw `cudaMalloc`/`cudaMemcpy`.
- Use `cuda_tools/` RAII buffers for owned device memory rather than manual
  `MALLOC_DEVICE_MEM`/`FREE_DEVICE_MEM` patterns.

## Backend Selection Discipline

Backend selection is **runtime** via `RuntimeBackendConfig`, never compile-time
`#ifdef`. Do not introduce preprocessor forks that select between `GIPC` /
`SRBK` / `CEMAS` paths. See [backends.md](backends.md).

## Git Workflow

- Commit messages in this repo are terse (e.g. `minor`, `0612 backup`,
  `benchmark development over`). When making substantive changes, prefer a
  short imperative summary that names the area, e.g.
  `pcg: add per-iteration axpby timer` or `bench: support ipc-sim dataset`.
- `*.md`, `*.pdf`, `*.tar.gz`, `*.sh`, `Output/*`, `build*`, `summary*` are
  gitignored. To share a markdown doc or script, either `git add -f` it or add
  a negation rule to `.gitignore`. Force-add is the established pattern
  (`README.md` and `docs/target.md` are tracked this way).
- Do not commit run output (`Output/`) or downloaded datasets
  (`datasets/*.tar.gz`, unpacked mesh dirs are already ignored as needed).

## Adding New Code

- New solver behavior that is backend-sensitive: extend `RuntimeBackendConfig`
  and branch at runtime.
- New benchmark metric: add the per-frame field to `gipc::Statistics`, read it
  in `summarize_frames()` (`benchmark_report.cpp`), and add a column to the
  `summary.csv` header. `csv_to_json.py` will pick it up automatically.
- New experiment: create `datasets/exp_MMDD/` with suite JSONs + `settings/`.
  See [experiments.md](experiments.md).
