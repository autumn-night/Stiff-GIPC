# Build & Benchmark Workflow

## Build Configuration

CMake options (all in `CMakeLists.txt`):

| Option | Default | Effect |
|--------|---------|--------|
| `BUILD_GIPC_VIEWER` | `ON` | Build `gipc_viewer` (needs freeglut/GLEW/OpenGL) |
| `BUILD_GIPC_BENCH` | `ON` | Build `gipc_bench` (headless, no GL deps) |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | Override for reproducible arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` |

`compile_commands.json` is exported for IDE/clangd integration.

Compile defines hard-coded PRIVATE on `stiffgipc_core` (see
[backends.md](backends.md)):

```
USE_SNK1  ADAPTIVE_KAPPA  USE_FRICTION  USE_QUADRATIC_BENDING
```

The git short hash and build type are baked in as `GIPC_GIT_HASH` /
`GIPC_BUILD_TYPE`; asset/output dirs as `GIPC_ASSETS_DIR` / `GIPC_OUTPUT_DIR`.

### Headless build (recommended for benchmark hosts)

```bash
cmake -S . -B build -DBUILD_GIPC_VIEWER=OFF
cmake --build build
```

### Full build (with viewer)

Requires `libglew-dev freeglut3-dev libeigen3-dev nlohmann-json3-dev` (Linux).
On Windows use vcpkg; set `CMAKE_TOOLCHAIN_FILE` to the vcpkg toolchain.

```bash
cmake -S . -B build
cmake --build build
```

Binaries land in `build/` (`gipc_bench`, `gipc_viewer`). There is no install
rule; run from the repo root so the `Assets/` / `datasets/` relative paths
resolve.

## `gipc_bench` CLI

### Single run

```bash
./build/gipc_bench --scene cloth_bunny --baseline StiffGIPC_CEMAS_SRBK \
                   --frames 100 --warmup 10
```

Flags (see `app/bench/main.cpp` for the parser):

| Flag | Purpose |
|------|---------|
| `--scene <name>` | Builtin scene name (see below) |
| `--baseline <name>` | `GIPC` \| `StiffGIPC_SRBK` \| `StiffGIPC_CEMAS_SRBK` |
| `--frames <n>` | Measured frames (default 100) |
| `--warmup <n>` | Warmup frames excluded from stats (default 10) |
| `--frame-start <n>` | Skip N frames before warmup (resume a sim) |
| `--frame-cap <n>` | Limit measured frames to `min(frames, cap)` |
| `--seed <n>` | RNG seed |
| `--settings <path>` | `parameterSetting.txt` override |
| `--output <dir>` | Override `output_root` |
| `--no-save-surface` | Skip surface mesh serialization |
| `--dataset <name>` | `stanford` \| `ipc-sim` (with `--scene local_dataset`) |
| `--task-id <id>` | Task within the dataset (e.g. `ArmadilloStand_0`) |
| `--asset-root <dir>` | Directory containing dataset meshes |
| `--manifest <path>` | Load run config from a manifest JSON; CLI flags override |
| `--notes <text>` | Free-form note recorded in `meta.json` |

`--baseline` parsing is case/underscore/punctuation-insensitive, so
`stiffgipc cemas+srbk` also works.

### Suite run

```bash
./build/gipc_bench --suite datasets/exp_0528/bench_sweep_stiffness.json
```

### Builtin scenes

Defined in `app/common/scene_registry.cpp`. Each has a friendly name and a
legacy `caseN` alias:

| Friendly | Alias | Content |
|----------|-------|---------|
| `box_pipe` | `case1` | Box pipe |
| `cloth_bunny` | `case2` | Cloth over bunny |
| `wrecking_ball` | `case3` | Wrecking ball |
| `fixed_cloth` | `case4` | Fixed cloth |
| `mat_twist` | `case5` | Material twist |
| `box_pile` | `case6` | Box pile |

For external meshes use `--scene local_dataset` with `--dataset`/`--task-id`/
`--asset-root`. Supported datasets:

- `stanford` — one `.obj` per task; `--task-id` is the obj stem
  (e.g. `ArmadilloStand_0`).
- `ipc-sim` — IPC benchmark `.txt` scene files; `--task-id` is the path
  relative to the dataset root without extension
  (e.g. `tutorialExamples/2cubesFall`). Scenes using `meshCO` are skipped.

## Suite JSON Format

Schema (see `load_benchmark_suite()` in `app/bench/benchmark_config.cpp`):

```jsonc
{
  "output_root": "Output/benchmarks",      // repo-relative if starts with Output/
  "datasets_root": "datasets",             // for auto_discover
  "dataset_selector": "stanford",          // "" = all
  "task_selector": "",                     // "" = all
  "auto_discover": false,                  // scan datasets/ for benchmark.json manifests
  "defaults": {                            // applied to every run
    "frames": 100,
    "warmup": 10,
    "baseline": "StiffGIPC_CEMAS_SRBK",
    "settings": "Assets/scene/parameterSetting.txt",
    "dataset": "stanford",
    "task_id": "ArmadilloStand_0",
    "scene": "local_dataset",
    "asset_root": "../stanford",
    "preconditioner_config": { /* see backends.md */ }
  },
  "runs": [
    { "settings": "settings/parameterSetting_poisson_0.3.txt",
      "output_tag": "poisson_0.3", "notes": "poisson=0.3" }
  ]
}
```

Per-run keys (all optional; inherit from `defaults`): `scene`, `baseline`,
`frames`, `warmup`, `settings`, `output_root`, `seed`, `frame_start`,
`frame_cap`, `no_save_surface`, `task_id`, `dataset`, `asset_root`,
`output_tag`, `notes`, `manifest`, `preconditioner_config`.

Path resolution: a path whose first component is `Assets`, `datasets`, or
`Output` is resolved relative to the repo root; otherwise relative to the suite
file's directory. Absolute paths are used as-is.

`auto_discover` scans each `datasets/<name>/` subdir for a `benchmark.json`
manifest (the `stanford`/`ipc-sim` datasets use their own mesh-based discovery
instead).

## Output Layout

For each run, output goes to:

```
<output_root>/<dataset or scene>/<task_id or "">/<baseline>/[<output_tag>]/
  raw.json     # per-frame timings + counters (see experiments.md for fields)
  meta.json    # run config + git hash + CLI args
```

A single `summary.csv` aggregating all runs is written to `<output_root>/` and
updated after every run in the suite (so partial results survive a crash).

## Parameter Settings File

`Assets/scene/parameterSetting.txt` and the per-experiment variants under
`datasets/exp_XXXX/settings/` are space-separated `key value` lines parsed by
`load_simulation_settings()` (`app/common/sim_settings.cpp`). Order is fixed:

```
density  poisson  friction  ground_friction  cloth_thickness
cloth_young  bend_young  cloth_density  strain_rate  soft_motion_rate
cd_buff_scale  motion_rate  ipc_dt  pcg_threshold  newton_threshold  relative_dhat
```

The leading token on each line is ignored (it is a human-readable label). When
adding a sweep, copy an existing file and change only the relevant field.
