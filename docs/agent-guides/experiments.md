# Experiments & Analysis

## Experiment Organization

Experiments live under `datasets/exp_MMDD/` (month-day of the run). Each
experiment contains one or more bench suite JSONs and, usually, a `settings/`
directory with per-variant `parameterSetting_*.txt` files.

```
datasets/exp_0528/
├── bench_sweep_bend.json          # suite JSONs (one per swept parameter)
├── bench_sweep_dhat.json
├── bench_sweep_friction.json
├── bench_sweep_stiffness.json
├── bench_sweep_stretch.json
├── bench_sweep_timestep.json
└── settings/                       # parameterSetting_<variant>.txt files
```

A sweep suite sets shared `defaults` (scene, dataset, task_id, baseline) and
each `runs[]` entry overrides only the swept parameter via a different
`settings` file plus an `output_tag` and `notes`. See
`datasets/exp_0528/bench_sweep_stiffness.json` for the canonical pattern.

To batch-run every suite in an experiment, copy the `run_all.sh` pattern from
`datasets/exp_0615/`:

```bash
for suite in diag_baseline.json bend_aware_sweep.json ...; do
    ./build/gipc_bench --suite "datasets/exp_0615/$suite"
done
```

### Existing experiments

| Experiment | Purpose | Granularity |
|------------|---------|-------------|
| `exp_0528` | Parameter sweeps (bend, dhat, friction, stiffness, stretch, timestep) on Stanford `ArmadilloStand_0` | total breakdown |
| `exp_0604` | Total time breakdown (paired with exp_0528) | total breakdown |
| `exp_0608`, `exp_0609` | LSolver internal breakdown | LSolver breakdown |
| `exp_0610` | PCG internal breakdown | PCG breakdown |
| `exp_0615` | MAS preconditioner improvements (bend-aware, contact-aware, inexact-Newton, precond-reuse, diag baseline) | optimization strategy |

When adding a new experiment, name it `exp_MMDD`, add the suite JSONs under
`datasets/exp_MMDD/`, and reuse settings files from a prior experiment where
possible (experiments commonly reference `../exp_0528/settings/...`).

## Metrics

The benchmark emits three granularity levels. All are per-frame averages over
the measured (post-warmup) frames, written to `summary.csv`. Per-frame raw
values go into `raw.json`.

### Total breakdown (the headline metrics)

| Field | Meaning |
|-------|---------|
| `avg_Hess_ms` | Per-frame gradient/Hessian assembly time |
| `avg_LSolver_ms` | Per-frame linear system solve time |
| `avg_LineS_ms` | Per-frame line search time |
| `avg_Misc_ms` | `TimeTot - Hess - LSolver - LineS` |
| `avg_TimeTot_ms` | Full per-frame solve time |
| `avg_newton` | Newton iterations per frame |
| `avg_cg` | CG iterations per frame |
| `avg_contact_pairs` | Average contact pairs |
| `std_TimeTot_ms` | Stddev of total time |

Internally `GIPC::solve_subIP()` returns `time0..time4` segments:
`time0->Hess`, `time1->LSolver`, `time2->CCD/feasible-step raw`,
`time3->LineS`, `time4->postLineSearch raw`. `time2`/`time4` are kept in the
raw report but not surfaced in the headline table; `Misc` is the aggregate
residual.

### LSolver breakdown

`avg_LSolver_SubsystemAssemble_ms`, `avg_LSolver_TripletOps_ms`,
`avg_LSolver_PreconditionerAssemble_ms`, `avg_LSolver_PCG_ms`,
`avg_LSolver_SolutionDistribute_ms`.

### PCG breakdown

`avg_PCG_PreconditionerApply_ms`, `avg_PCG_SpMV_ms`, `avg_PCG_Dot_ms`,
`avg_PCG_Axpby_ms`. Accumulated in `PCGSolver` without per-iteration
`cudaDeviceSynchronize` to avoid skewing timings.

## Analysis Pipeline

All scripts are stdlib-only Python except `visualize_experiments.py` (needs
numpy + matplotlib). Run from the repo root.

1. **Run experiments** on a CUDA host. Results land under
   `Output/experiments/exp_MMDD/.../summary.csv` (+ `raw.json`, `meta.json`).
2. **CSV -> JSON**: `python scripts/csv_to_json.py`
   - Recursively globs `Output/experiments/**/summary.csv`.
   - Writes one JSON per CSV to `docs/json_output/` plus `index.json`.
   - Columns are discovered dynamically (never hardcoded), so new metric
     columns are picked up automatically.
3. **Analyze**: `python scripts/analyze_experiments.py`
   - Reads `docs/json_output/*.json`, groups by the four analysis groups
     (total breakdown / LSolver breakdown / PCG breakdown / strategy
     comparison).
   - Writes aggregated JSON to `docs/json_output/analysis/`.
   - Writes a Markdown report to `docs/analysis_report.md`.
4. **Visualize**: `conda run -n daily python scripts/visualize_experiments.py`
   - Reads `analysis/json_output/analysis/` (copy the aggregated JSON there
     first).
   - Renders pie charts + stacked bar charts to `analysis/figures/`.

The generated `analysis/analysis_report.md` is the canonical human-readable
summary of a batch of experiments.

## Reproducibility Checklist

- Pin `baseline`, `frames`, `warmup`, `settings`, `dataset`, `task_id`,
  `asset_root` identically across runs you intend to compare.
- Record the GPU name + CUDA runtime version (captured automatically in
  `meta.json`) — absolute timings are GPU-dependent.
- Keep the `settings/parameterSetting_*.txt` files in the repo (under
  `datasets/exp_MMDD/settings/`) so a sweep can be re-run later.
- `summary.csv` is rewritten after every run in a suite, so partial results
  survive a mid-suite crash; copy the directory aside before re-running.
