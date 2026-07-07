# Backends & Runtime Config

## Baseline Matrix

Three baselines are fixed and must stay runtime-switchable. They are the core
of every benchmark comparison and **must never be collapsed into a single
compile-time path**. Defined in `StiffGIPC/gipc/runtime_config.h`.

| Baseline | Assembly | SpMV | MAS | Notes |
|----------|----------|------|-----|-------|
| `GIPC` | LegacyGipc | LegacyGipc | GpuMas | Original GIPC path (Morton partition) |
| `StiffGIPC_SRBK` | SRBK | SRBK | GpuMas | SRBK linear system, legacy MAS |
| `StiffGIPC_CEMAS_SRBK` (default) | SRBK | SRBK | CEMAS | Full StiffGIPC stack |

Resolution lives in `resolve_runtime_backend_config()`:

```cpp
RuntimeBackendConfig resolve_runtime_backend_config(BenchmarkBaseline baseline);
```

Do **not** approximate `GIPC` as `Diag + SRBK` — it must use the genuine legacy
assembly/SpMV/MAS code paths. Scene, timestep, tolerance, and material
parameters must be identical across baselines in any comparison.

## `RuntimeBackendConfig`

```cpp
struct RuntimeBackendConfig {
    BenchmarkBaseline     baseline         = StiffGIPC_CEMAS_SRBK;
    LinearAssemblyBackend assembly_backend = SRBK;   // LegacyGipc | SRBK
    SpmvBackend           spmv_backend     = SRBK;   // LegacyGipc | SRBK
    MasBackend            mas_backend      = CEMAS;  // GpuMas | CEMAS

    // Preconditioner improvement flags (all default false / 0)
    bool   bend_aware_neighbor      = false;
    bool   contact_aware_precond    = false;
    bool   inexact_newton           = false;
    double inexact_eta_early        = 1e-2;
    double inexact_eta_mid          = 1e-3;
    int    inexact_early_steps      = 5;
    int    inexact_mid_steps        = 15;
    bool   precond_reuse            = false;
    int    precond_reuse_interval   = 0;
    double precond_reuse_cpnum_threshold = 0.1;
    bool   diag_cluster_stats       = false;
};
```

### Preconditioner Improvement Flags

These are research toggles layered on top of the default CEMAS+SRBK
preconditioner. They are off by default and enabled per-run via the
`preconditioner_config` block in a suite JSON (applied by
`apply_preconditioner_config_from_json()`).

| Flag | What it does |
|------|--------------|
| `bend_aware_neighbor` | Step 1: add hinge opposite vertices as MAS neighbors |
| `contact_aware_precond` | Step 2: Schur-complement enhancement for cross-cluster contacts |
| `inexact_newton` | Step 3: dynamic per-Newton-step CG tolerance |
| `inexact_eta_early` / `inexact_eta_mid` | Step 3: CG tol for early / mid Newton steps |
| `inexact_early_steps` / `inexact_mid_steps` | Step 3: how many steps count as early / mid |
| `precond_reuse` | Step 4: reuse aggregation structure across frames |
| `precond_reuse_interval` | Step 4: reaggregate every N frames (0 = only cpNum-trigger) |
| `precond_reuse_cpnum_threshold` | Step 4: relative cpNum change that forces reaggregation |
| `diag_cluster_stats` | Step 0: collect cross-cluster triplet diagnostics |

Example (from `datasets/exp_0615/inexact_newton.json`):

```jsonc
"preconditioner_config": {
  "inexact_newton": true,
  "inexact_eta_early": 0.01,
  "inexact_eta_mid": 0.001,
  "inexact_early_steps": 5,
  "inexact_mid_steps": 15
}
```

When adding a new toggle, add the field to `RuntimeBackendConfig`, the JSON
application in `runtime_config.cpp`, and document it here.

## Compile-Time Defines

These are set PRIVATE on `stiffgipc_core` in `CMakeLists.txt` and apply to the
whole library. They enable energy/feature paths, not backend selection:

| Define | Effect |
|--------|--------|
| `USE_SNK1` | SNK1 barrier stiffness schedule |
| `ADAPTIVE_KAPPA` | Per-scene adaptive kappa selection |
| `USE_FRICTION` | Compile friction lagrange/tangent kernels |
| `USE_QUADRATIC_BENDING` | Quadratic bending energy (vs linear) |

Do not toggle these to switch baselines — backend selection is runtime-only.

## Adding a New Backend Variant

1. Add the enum value to the relevant backend enum in `runtime_config.h`.
2. Return it from `resolve_runtime_backend_config()` for a new
   `BenchmarkBaseline`, or leave it user-selectable.
3. Add the branch in the consuming code (e.g. `spmv.cu`, `converter.cu`,
   `MASPreconditioner.cu`). Keep it a runtime `if`/`switch`, never an
   `#ifdef`.
4. Add a `to_string()` case and a parser case if it needs CLI/JSON exposure.
5. Update this table.
