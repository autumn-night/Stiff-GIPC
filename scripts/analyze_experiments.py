#!/usr/bin/env python3
"""Analyze experiment JSON data and generate analysis outputs + Markdown report.

Reads 32 JSON data files from docs/json_output/, performs grouped aggregation
analysis across four analysis groups, writes 4 analysis JSON files to
docs/json_output/analysis/, and generates a Markdown report at
docs/analysis_report.md.

Analysis groups:
  1. exp_0528 + exp_0604 -- total time breakdown (17-column format A)
  2. exp_0608 + exp_0609 -- LSolver internal breakdown (22-column format B)
  3. exp_0610            -- PCG internal breakdown (26-column format C)
  4. exp_0615            -- optimization strategy comparison

Only the Python standard library is used (json, pathlib, datetime, re).
"""

import json
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "docs" / "json_output"
OUTPUT_DIR = INPUT_DIR / "analysis"
REPORT_PATH = PROJECT_ROOT / "docs" / "analysis_report.md"

# ---------------------------------------------------------------------------
# Field groups for breakdown analysis
# ---------------------------------------------------------------------------

TOTAL_BREAKDOWN_FIELDS = [
    "avg_Hess_ms",
    "avg_LSolver_ms",
    "avg_LineS_ms",
    "avg_Misc_ms",
]

LSOLVER_BREAKDOWN_FIELDS = [
    "avg_LSolver_SubsystemAssemble_ms",
    "avg_LSolver_TripletOps_ms",
    "avg_LSolver_PreconditionerAssemble_ms",
    "avg_LSolver_PCG_ms",
    "avg_LSolver_SolutionDistribute_ms",
]

PCG_BREAKDOWN_FIELDS = [
    "avg_PCG_PreconditionerApply_ms",
    "avg_PCG_SpMV_ms",
    "avg_PCG_Dot_ms",
    "avg_PCG_Axpby_ms",
]

# Short display names for Markdown tables
TOTAL_BREAKDOWN_LABELS = {
    "avg_Hess_ms": "Hess",
    "avg_LSolver_ms": "LSolver",
    "avg_LineS_ms": "LineS",
    "avg_Misc_ms": "Misc",
}

LSOLVER_BREAKDOWN_LABELS = {
    "avg_LSolver_SubsystemAssemble_ms": "SubAsm",
    "avg_LSolver_TripletOps_ms": "Triplet",
    "avg_LSolver_PreconditionerAssemble_ms": "Precond",
    "avg_LSolver_PCG_ms": "PCG",
    "avg_LSolver_SolutionDistribute_ms": "Dist",
}

PCG_BREAKDOWN_LABELS = {
    "avg_PCG_PreconditionerApply_ms": "PrecondApply",
    "avg_PCG_SpMV_ms": "SpMV",
    "avg_PCG_Dot_ms": "Dot",
    "avg_PCG_Axpby_ms": "Axpby",
}

# Human-readable Chinese titles for each subdir
SUBDIR_TITLES = {
    "exp_sweep_bend": "弯曲刚度扫描",
    "exp_sweep_dhat": "接触距离扫描",
    "exp_sweep_friction": "摩擦系数扫描",
    "exp_sweep_stiffness": "整体刚度扫描",
    "exp_sweep_stretch": "拉伸刚度扫描",
    "exp_sweep_timestep": "时间步扫描",
    "interaction_dhat_bend": "dhat-bend 交互",
    "interaction_dhat_dt": "dhat-dt 交互",
    "interaction_dt_bend": "dt-bend 交互",
    "focus_dhat_dt": "dhat-dt 聚焦",
    "focus_dt_bend": "dt-bend 聚焦",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def safe_ratio(part, total):
    """Compute part/total rounded to 4 decimal places.

    Returns 0.0 when *total* is zero or None.
    """
    if not total:
        return 0.0
    return round(part / total, 4)


def safe_div(a, b):
    """Compute a/b rounded to 2 decimal places.

    Returns 0.0 when *b* is zero or None.
    """
    if not b:
        return 0.0
    return round(a / b, 2)


def cg_per_newton(row):
    """Compute avg_cg / avg_newton for a data row."""
    return safe_div(row.get("avg_cg"), row.get("avg_newton"))


def speedup(baseline_time, optimized_time):
    """Compute speedup = baseline / optimized, rounded to 3 decimals."""
    if not baseline_time or not optimized_time:
        return 0.0
    return round(baseline_time / optimized_time, 3)


def load_json(filename):
    """Load a JSON file from the input directory."""
    path = INPUT_DIR / filename
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_json_files(prefixes):
    """Return sorted list of JSON file paths matching the given prefixes."""
    result = []
    for prefix in prefixes:
        result.extend(sorted(INPUT_DIR.glob(f"{prefix}__*.json")))
    return sorted(result)


def compute_breakdown(row, fields, total_field):
    """Compute a breakdown dict and sum for the given fields.

    Each entry is {value_ms, ratio} where ratio = value/total.
    Fields missing from the row are skipped.
    """
    total = row.get(total_field)
    breakdown = {}
    for field in fields:
        val = row.get(field)
        if val is None:
            continue
        breakdown[field] = {
            "value_ms": val,
            "ratio": safe_ratio(val, total),
        }
    breakdown_sum = round(sum(v["ratio"] for v in breakdown.values()), 4)
    return breakdown, breakdown_sum


def compute_total_breakdown(row):
    return compute_breakdown(row, TOTAL_BREAKDOWN_FIELDS, "avg_TimeTot_ms")


def compute_lsolver_breakdown(row):
    return compute_breakdown(row, LSOLVER_BREAKDOWN_FIELDS, "avg_LSolver_ms")


def compute_pcg_breakdown(row):
    return compute_breakdown(row, PCG_BREAKDOWN_FIELDS, "avg_LSolver_PCG_ms")


def compute_all_breakdowns(row):
    """Compute all three breakdown layers for a row.

    Returns a dict with breakdown, breakdown_sum, lsolver_breakdown,
    lsolver_breakdown_sum, pcg_breakdown, pcg_breakdown_sum.
    Only layers whose fields exist in the row will have non-empty dicts.
    """
    tb, tbs = compute_total_breakdown(row)
    lb, lbs = compute_lsolver_breakdown(row)
    pb, pbs = compute_pcg_breakdown(row)
    return {
        "breakdown": tb,
        "breakdown_sum": tbs,
        "lsolver_breakdown": lb,
        "lsolver_breakdown_sum": lbs,
        "pcg_breakdown": pb,
        "pcg_breakdown_sum": pbs,
    }


def match_scenario(notes):
    """Match a row's notes string to a baseline scenario.

    Returns one of "high_bend", "large_dt", "default", or None.
    """
    if not notes:
        return None
    if "高 bend" in notes or "高bend" in notes:
        return "high_bend"
    if "大 dt" in notes or "大dt" in notes or "大时间步" in notes:
        return "large_dt"
    if "默认" in notes:
        return "default"
    return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_group1(timestamp):
    """Group 1: exp_0528 + exp_0604 -- total time breakdown."""
    files = list_json_files(["exp_0528", "exp_0604"])
    experiments = []
    for f in files:
        data = load_json(f.name)
        rows_out = []
        for row in data["rows"]:
            breakdown, breakdown_sum = compute_total_breakdown(row)
            rows_out.append({
                "notes": row.get("notes"),
                "avg_TimeTot_ms": row.get("avg_TimeTot_ms"),
                "avg_newton": row.get("avg_newton"),
                "avg_cg": row.get("avg_cg"),
                "avg_contact_pairs": row.get("avg_contact_pairs"),
                "cg_per_newton": cg_per_newton(row),
                "breakdown": breakdown,
                "breakdown_sum": breakdown_sum,
            })
        experiments.append({
            "experiment": data["experiment"],
            "subdir": data["subdir"],
            "json_file": f.name,
            "rows": rows_out,
        })
    return {
        "group": "exp_0528_0604",
        "analysis_type": "total_time_breakdown",
        "generated_at": timestamp,
        "experiments": experiments,
    }


def analyze_group2(timestamp):
    """Group 2: exp_0608 + exp_0609 -- LSolver internal breakdown."""
    files = list_json_files(["exp_0608", "exp_0609"])
    experiments = []
    for f in files:
        data = load_json(f.name)
        rows_out = []
        for row in data["rows"]:
            tb, tbs = compute_total_breakdown(row)
            lb, lbs = compute_lsolver_breakdown(row)
            rows_out.append({
                "notes": row.get("notes"),
                "avg_TimeTot_ms": row.get("avg_TimeTot_ms"),
                "avg_LSolver_ms": row.get("avg_LSolver_ms"),
                "avg_newton": row.get("avg_newton"),
                "avg_cg": row.get("avg_cg"),
                "avg_contact_pairs": row.get("avg_contact_pairs"),
                "cg_per_newton": cg_per_newton(row),
                "breakdown": tb,
                "breakdown_sum": tbs,
                "lsolver_breakdown": lb,
                "lsolver_breakdown_sum": lbs,
            })
        experiments.append({
            "experiment": data["experiment"],
            "subdir": data["subdir"],
            "json_file": f.name,
            "rows": rows_out,
        })
    return {
        "group": "exp_0608_0609",
        "analysis_type": "lsolver_breakdown",
        "generated_at": timestamp,
        "experiments": experiments,
    }


def analyze_group3(timestamp):
    """Group 3: exp_0610 -- PCG internal breakdown."""
    files = list_json_files(["exp_0610"])
    experiments = []
    for f in files:
        data = load_json(f.name)
        rows_out = []
        for row in data["rows"]:
            tb, tbs = compute_total_breakdown(row)
            lb, lbs = compute_lsolver_breakdown(row)
            pb, pbs = compute_pcg_breakdown(row)
            rows_out.append({
                "notes": row.get("notes"),
                "avg_TimeTot_ms": row.get("avg_TimeTot_ms"),
                "avg_LSolver_ms": row.get("avg_LSolver_ms"),
                "avg_LSolver_PCG_ms": row.get("avg_LSolver_PCG_ms"),
                "avg_newton": row.get("avg_newton"),
                "avg_cg": row.get("avg_cg"),
                "avg_contact_pairs": row.get("avg_contact_pairs"),
                "cg_per_newton": cg_per_newton(row),
                "breakdown": tb,
                "breakdown_sum": tbs,
                "lsolver_breakdown": lb,
                "lsolver_breakdown_sum": lbs,
                "pcg_breakdown": pb,
                "pcg_breakdown_sum": pbs,
            })
        experiments.append({
            "experiment": data["experiment"],
            "subdir": data["subdir"],
            "json_file": f.name,
            "rows": rows_out,
        })
    return {
        "group": "exp_0610",
        "analysis_type": "pcg_breakdown",
        "generated_at": timestamp,
        "experiments": experiments,
    }


def _full_row_with_breakdowns(row):
    """Build a row dict with all three breakdown layers plus key metrics."""
    bd = compute_all_breakdowns(row)
    return {
        "notes": row.get("notes"),
        "avg_TimeTot_ms": row.get("avg_TimeTot_ms"),
        "avg_LSolver_ms": row.get("avg_LSolver_ms"),
        "avg_LSolver_PCG_ms": row.get("avg_LSolver_PCG_ms"),
        "avg_newton": row.get("avg_newton"),
        "avg_cg": row.get("avg_cg"),
        "avg_contact_pairs": row.get("avg_contact_pairs"),
        "cg_per_newton": cg_per_newton(row),
        **bd,
    }


def analyze_group4(timestamp):
    """Group 4: exp_0615 -- optimization strategy comparison."""

    # --- Load diag_baseline as the baseline ---
    diag_data = load_json("exp_0615__diag_baseline.json")
    baseline_scenarios = {}
    diag_rows_full = []
    for row in diag_data["rows"]:
        scenario = match_scenario(row.get("notes"))
        entry = _full_row_with_breakdowns(row)
        entry["scenario"] = scenario
        diag_rows_full.append(entry)
        if scenario:
            baseline_scenarios[scenario] = {
                "notes": row.get("notes"),
                "avg_TimeTot_ms": row.get("avg_TimeTot_ms"),
                "avg_LSolver_ms": row.get("avg_LSolver_ms"),
                "avg_LSolver_PCG_ms": row.get("avg_LSolver_PCG_ms"),
                "cg_per_newton": cg_per_newton(row),
                "breakdown": entry["breakdown"],
                "breakdown_sum": entry["breakdown_sum"],
                "lsolver_breakdown": entry["lsolver_breakdown"],
                "lsolver_breakdown_sum": entry["lsolver_breakdown_sum"],
                "pcg_breakdown": entry["pcg_breakdown"],
                "pcg_breakdown_sum": entry["pcg_breakdown_sum"],
            }

    # --- Load inexact_newton high_bend row for variant comparison ---
    inexact_data = load_json("exp_0615__inexact_newton.json")
    inexact_high_bend = None
    for row in inexact_data["rows"]:
        if match_scenario(row.get("notes")) == "high_bend":
            inexact_high_bend = row
            break
    inexact_hb_time = (
        inexact_high_bend.get("avg_TimeTot_ms") if inexact_high_bend else None
    )
    inexact_hb_cgn = (
        cg_per_newton(inexact_high_bend) if inexact_high_bend else None
    )

    # --- Comparison helper ---
    def _comparison_row(row, baseline_time, baseline_cgn):
        """Build a comparison row with speedup and cg_reduction."""
        opt_time = row.get("avg_TimeTot_ms")
        bd = compute_all_breakdowns(row)
        opt_cgn = cg_per_newton(row)
        if baseline_time and opt_time:
            sp = speedup(baseline_time, opt_time)
        else:
            sp = None
        if baseline_cgn and opt_cgn:
            cg_red = round(1 - (opt_cgn / baseline_cgn), 4)
        else:
            cg_red = None
        return {
            "notes": row.get("notes"),
            "baseline_time_ms": baseline_time,
            "optimized_time_ms": opt_time,
            "speedup": sp,
            "baseline_cg_per_newton": baseline_cgn,
            "optimized_cg_per_newton": opt_cgn,
            "cg_reduction": cg_red,
            **bd,
        }

    # --- Scenario-based comparisons (vs diag_baseline) ---
    scenario_groups = [
        "bend_aware",
        "contact_aware",
        "inexact_newton",
        "precond_reuse",
    ]
    comparisons = {}
    for group_name in scenario_groups:
        data = load_json(f"exp_0615__{group_name}.json")
        rows_out = []
        for row in data["rows"]:
            scenario = match_scenario(row.get("notes"))
            baseline = baseline_scenarios.get(scenario)
            if baseline:
                b_time = baseline["avg_TimeTot_ms"]
                b_cgn = baseline["cg_per_newton"]
            else:
                b_time = None
                b_cgn = None
            entry = _comparison_row(row, b_time, b_cgn)
            entry["baseline_scenario"] = scenario
            rows_out.append(entry)
        comparisons[group_name] = rows_out

    # --- inexact_newton_variants (vs inexact_newton high_bend) ---
    variants_data = load_json("exp_0615__inexact_newton_variants.json")
    variants_out = []
    for row in variants_data["rows"]:
        entry = _comparison_row(row, inexact_hb_time, inexact_hb_cgn)
        entry["baseline_source"] = "inexact_newton/high_bend"
        variants_out.append(entry)
    comparisons["inexact_newton_variants"] = variants_out

    # --- supplement_5e8 (internal comparison, row[0] is baseline) ---
    supp_data = load_json("exp_0615__supplement_5e8.json")
    supp_rows = supp_data["rows"]
    supp_baseline = supp_rows[0] if supp_rows else None
    supp_baseline_time = (
        supp_baseline.get("avg_TimeTot_ms") if supp_baseline else None
    )
    supp_out = []
    for i, row in enumerate(supp_rows):
        opt_time = row.get("avg_TimeTot_ms")
        bd = compute_all_breakdowns(row)
        sp = speedup(supp_baseline_time, opt_time) if supp_baseline_time else 0.0
        supp_out.append({
            "notes": row.get("notes"),
            "row_index": i,
            "avg_TimeTot_ms": opt_time,
            "speedup_vs_baseline": sp,
            "avg_LSolver_ms": row.get("avg_LSolver_ms"),
            "cg_per_newton": cg_per_newton(row),
            **bd,
        })

    return {
        "group": "exp_0615",
        "analysis_type": "optimization_comparison",
        "generated_at": timestamp,
        "baseline": {
            "subdir": "diag_baseline",
            "scenarios": baseline_scenarios,
            "all_rows": diag_rows_full,
        },
        "inexact_newton_high_bend": {
            "notes": inexact_high_bend.get("notes") if inexact_high_bend else None,
            "avg_TimeTot_ms": inexact_hb_time,
            "cg_per_newton": inexact_hb_cgn,
        },
        "comparisons": comparisons,
        "supplement_5e8": {
            "baseline_notes": supp_baseline.get("notes") if supp_baseline else None,
            "baseline_time_ms": supp_baseline_time,
            "rows": supp_out,
        },
    }


# ---------------------------------------------------------------------------
# Markdown formatting helpers
# ---------------------------------------------------------------------------


def fmt(val, decimals=1):
    """Format a numeric value; '-' for None."""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def fmt_pct(ratio, decimals=1):
    """Format a ratio (0-1) as a percentage string; '-' for None."""
    if ratio is None:
        return "-"
    return f"{ratio * 100:.{decimals}f}%"


def md_table(headers, rows):
    """Generate a Markdown table from headers and row data.

    Each row is a list of pre-formatted strings.
    """
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["-------" for _ in headers]) + "|")
    for row in rows:
        cells = [str(v) if v is not None else "-" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def subdir_title(experiment, subdir):
    """Generate a human-readable subsection title."""
    title = SUBDIR_TITLES.get(subdir, subdir)
    return f"{title} ({experiment}/{subdir})"


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def generate_group1_tables(group1_data):
    """Generate Markdown for group1 (total time breakdown)."""
    sections = []
    counter = 0
    for exp in group1_data["experiments"]:
        counter += 1
        title = subdir_title(exp["experiment"], exp["subdir"])
        lines = [f"### 2.{counter} {title}", ""]

        headers = [
            "notes", "TimeTot(ms)", "Hess(ms)", "Hess%",
            "LSolver(ms)", "LSolver%", "LineS(ms)", "LineS%",
            "Misc(ms)", "Misc%", "CG/Newton",
        ]
        rows = []
        for r in exp["rows"]:
            bd = r["breakdown"]
            rows.append([
                r.get("notes", "-"),
                fmt(r.get("avg_TimeTot_ms")),
                fmt(bd.get("avg_Hess_ms", {}).get("value_ms")),
                fmt_pct(bd.get("avg_Hess_ms", {}).get("ratio")),
                fmt(bd.get("avg_LSolver_ms", {}).get("value_ms")),
                fmt_pct(bd.get("avg_LSolver_ms", {}).get("ratio")),
                fmt(bd.get("avg_LineS_ms", {}).get("value_ms")),
                fmt_pct(bd.get("avg_LineS_ms", {}).get("ratio")),
                fmt(bd.get("avg_Misc_ms", {}).get("value_ms")),
                fmt_pct(bd.get("avg_Misc_ms", {}).get("ratio")),
                fmt(r.get("cg_per_newton"), 1),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def generate_group2_tables(group2_data):
    """Generate Markdown for group2 (LSolver breakdown)."""
    sections = []
    counter = 0
    for exp in group2_data["experiments"]:
        counter += 1
        title = subdir_title(exp["experiment"], exp["subdir"])
        lines = [f"### 3.{counter} {title}", ""]

        headers = [
            "notes", "LSolver(ms)",
            "SubAsm(ms)", "SubAsm%",
            "Triplet(ms)", "Triplet%",
            "Precond(ms)", "Precond%",
            "PCG(ms)", "PCG%",
            "Dist(ms)", "Dist%",
            "CG/Newton",
        ]
        rows = []
        for r in exp["rows"]:
            lb = r["lsolver_breakdown"]
            def _val(field):
                return fmt(lb.get(field, {}).get("value_ms"))
            def _pct(field):
                return fmt_pct(lb.get(field, {}).get("ratio"))
            rows.append([
                r.get("notes", "-"),
                fmt(r.get("avg_LSolver_ms")),
                _val("avg_LSolver_SubsystemAssemble_ms"),
                _pct("avg_LSolver_SubsystemAssemble_ms"),
                _val("avg_LSolver_TripletOps_ms"),
                _pct("avg_LSolver_TripletOps_ms"),
                _val("avg_LSolver_PreconditionerAssemble_ms"),
                _pct("avg_LSolver_PreconditionerAssemble_ms"),
                _val("avg_LSolver_PCG_ms"),
                _pct("avg_LSolver_PCG_ms"),
                _val("avg_LSolver_SolutionDistribute_ms"),
                _pct("avg_LSolver_SolutionDistribute_ms"),
                fmt(r.get("cg_per_newton"), 1),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def generate_group3_tables(group3_data):
    """Generate Markdown for group3 (PCG breakdown)."""
    sections = []
    counter = 0
    for exp in group3_data["experiments"]:
        counter += 1
        title = subdir_title(exp["experiment"], exp["subdir"])
        lines = [f"### 4.{counter} {title}", ""]

        headers = [
            "notes", "PCG(ms)",
            "PrecondApply(ms)", "PrecondApply%",
            "SpMV(ms)", "SpMV%",
            "Dot(ms)", "Dot%",
            "Axpby(ms)", "Axpby%",
            "CG/Newton",
        ]
        rows = []
        for r in exp["rows"]:
            pb = r["pcg_breakdown"]
            def _val(field):
                return fmt(pb.get(field, {}).get("value_ms"))
            def _pct(field):
                return fmt_pct(pb.get(field, {}).get("ratio"))
            rows.append([
                r.get("notes", "-"),
                fmt(r.get("avg_LSolver_PCG_ms")),
                _val("avg_PCG_PreconditionerApply_ms"),
                _pct("avg_PCG_PreconditionerApply_ms"),
                _val("avg_PCG_SpMV_ms"),
                _pct("avg_PCG_SpMV_ms"),
                _val("avg_PCG_Dot_ms"),
                _pct("avg_PCG_Dot_ms"),
                _val("avg_PCG_Axpby_ms"),
                _pct("avg_PCG_Axpby_ms"),
                fmt(r.get("cg_per_newton"), 1),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")
        sections.append("\n".join(lines))
    return "\n".join(sections)


def generate_group4_tables(group4_data):
    """Generate Markdown for group4 (optimimization comparison)."""
    sections = []
    baseline = group4_data["baseline"]
    scenarios = baseline["scenarios"]
    comparisons = group4_data["comparisons"]
    supp = group4_data["supplement_5e8"]

    # --- 5.1 诊断基线 ---
    lines = ["### 5.1 诊断基线 (diag_baseline)", ""]
    headers = [
        "场景", "notes", "TimeTot(ms)", "LSolver(ms)",
        "CG/Newton", "PCG(ms)",
        "PrecondApply%", "SpMV%", "Dot%", "Axpby%",
    ]
    rows = []
    for r in baseline["all_rows"]:
        scenario = r.get("scenario") or "-"
        pb = r.get("pcg_breakdown", {})
        rows.append([
            scenario,
            r.get("notes", "-"),
            fmt(r.get("avg_TimeTot_ms")),
            fmt(r.get("avg_LSolver_ms")),
            fmt(r.get("cg_per_newton"), 1),
            fmt(r.get("avg_LSolver_PCG_ms")),
            fmt_pct(pb.get("avg_PCG_PreconditionerApply_ms", {}).get("ratio")),
            fmt_pct(pb.get("avg_PCG_SpMV_ms", {}).get("ratio")),
            fmt_pct(pb.get("avg_PCG_Dot_ms", {}).get("ratio")),
            fmt_pct(pb.get("avg_PCG_Axpby_ms", {}).get("ratio")),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    sections.append("\n".join(lines))

    # --- 5.2 - 5.4: scenario-based comparisons ---
    comp_titles = {
        "bend_aware": "5.2 bend-aware 优化",
        "contact_aware": "5.3 contact-aware 优化",
        "inexact_newton": "5.4 Inexact Newton",
    }
    for group_name in ["bend_aware", "contact_aware", "inexact_newton"]:
        lines = [f"### {comp_titles[group_name]}", ""]
        headers = [
            "notes", "基线场景", "基线时间(ms)", "优化时间(ms)",
            "加速比", "基线CG/N", "优化CG/N", "CG降幅",
        ]
        rows = []
        for r in comparisons[group_name]:
            rows.append([
                r.get("notes", "-"),
                r.get("baseline_scenario") or "-",
                fmt(r.get("baseline_time_ms")),
                fmt(r.get("optimized_time_ms")),
                fmt(r.get("speedup"), 3),
                fmt(r.get("baseline_cg_per_newton"), 1),
                fmt(r.get("optimized_cg_per_newton"), 1),
                fmt_pct(r.get("cg_reduction")),
            ])
        lines.append(md_table(headers, rows))
        lines.append("")
        sections.append("\n".join(lines))

    # --- 5.5 Inexact Newton 变体对比 ---
    lines = ["### 5.5 Inexact Newton 变体对比", ""]
    lines.append(
        f"> 基线：inexact_newton 高bend 行 "
        f"(TimeTot={fmt(inexact_hb_time := group4_data['inexact_newton_high_bend']['avg_TimeTot_ms'])} ms, "
        f"CG/N={fmt(group4_data['inexact_newton_high_bend']['cg_per_newton'], 1)})"
    )
    lines.append("")
    headers = [
        "变体", "基线时间(ms)", "优化时间(ms)", "加速比",
        "基线CG/N", "优化CG/N", "CG降幅",
    ]
    rows = []
    for r in comparisons["inexact_newton_variants"]:
        rows.append([
            r.get("notes", "-"),
            fmt(r.get("baseline_time_ms")),
            fmt(r.get("optimized_time_ms")),
            fmt(r.get("speedup"), 3),
            fmt(r.get("baseline_cg_per_newton"), 1),
            fmt(r.get("optimized_cg_per_newton"), 1),
            fmt_pct(r.get("cg_reduction")),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    sections.append("\n".join(lines))

    # --- 5.6 预条件器复用 ---
    lines = ["### 5.6 预条件器复用", ""]
    headers = [
        "notes", "基线场景", "基线时间(ms)", "优化时间(ms)",
        "加速比", "基线CG/N", "优化CG/N", "CG降幅",
    ]
    rows = []
    for r in comparisons["precond_reuse"]:
        rows.append([
            r.get("notes", "-"),
            r.get("baseline_scenario") or "-",
            fmt(r.get("baseline_time_ms")),
            fmt(r.get("optimized_time_ms")),
            fmt(r.get("speedup"), 3),
            fmt(r.get("baseline_cg_per_newton"), 1),
            fmt(r.get("optimized_cg_per_newton"), 1),
            fmt_pct(r.get("cg_reduction")),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    sections.append("\n".join(lines))

    # --- 5.7 5e8场景逐步优化 ---
    lines = ["### 5.7 5e8场景逐步优化 (supplement_5e8)", ""]
    lines.append(
        f"> 基线：{supp['baseline_notes']} "
        f"(TimeTot={fmt(supp['baseline_time_ms'])} ms)"
    )
    lines.append("")
    headers = ["notes", "TimeTot(ms)", "加速比", "LSolver(ms)", "CG/Newton"]
    rows = []
    for r in supp["rows"]:
        rows.append([
            r.get("notes", "-"),
            fmt(r.get("avg_TimeTot_ms")),
            fmt(r.get("speedup_vs_baseline"), 3),
            fmt(r.get("avg_LSolver_ms")),
            fmt(r.get("cg_per_newton"), 1),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    sections.append("\n".join(lines))

    # --- 5.8 优化效果汇总 ---
    lines = ["### 5.8 优化效果汇总", ""]
    headers = ["优化策略", "场景", "基线时间(ms)", "优化时间(ms)", "加速比", "CG降幅"]
    rows = []
    strategy_labels = {
        "bend_aware": "bend-aware",
        "contact_aware": "contact-aware",
        "inexact_newton": "Inexact Newton",
        "precond_reuse": "预条件器复用",
        "inexact_newton_variants": "Inexact Newton 变体",
    }
    for group_name in ["bend_aware", "contact_aware", "inexact_newton",
                       "precond_reuse", "inexact_newton_variants"]:
        for r in comparisons[group_name]:
            if r.get("speedup") is None:
                continue
            rows.append([
                strategy_labels.get(group_name, group_name),
                r.get("baseline_scenario") or r.get("baseline_source", "-"),
                fmt(r.get("baseline_time_ms")),
                fmt(r.get("optimized_time_ms")),
                fmt(r.get("speedup"), 3),
                fmt_pct(r.get("cg_reduction")),
            ])
    # supplement_5e8 rows (skip baseline row 0)
    for r in supp["rows"]:
        if r.get("row_index", 0) == 0:
            continue
        rows.append([
            "supplement_5e8",
            f"step {r['row_index']}",
            fmt(supp["baseline_time_ms"]),
            fmt(r.get("avg_TimeTot_ms")),
            fmt(r.get("speedup_vs_baseline"), 3),
            "-",
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    sections.append("\n".join(lines))

    return "\n".join(sections)


def generate_conclusions(group1_data, group2_data, group3_data, group4_data):
    """Generate the 'key findings' section based on analysis data."""
    lines = ["## 六、关键发现总结", ""]

    # 1. Highest LSolver% in exp_0528
    max_lsolver_ratio = 0.0
    max_lsolver_notes = ""
    for exp in group1_data["experiments"]:
        if exp["experiment"] != "exp_0528":
            continue
        for r in exp["rows"]:
            ratio = r["breakdown"].get("avg_LSolver_ms", {}).get("ratio", 0.0)
            if ratio > max_lsolver_ratio:
                max_lsolver_ratio = ratio
                max_lsolver_notes = r.get("notes", "")
    if max_lsolver_ratio > 0.90:
        lines.append(
            f"1. **线性求解器是绝对瓶颈**：在 exp_0528 中，"
            f"LSolver 占比最高达 {max_lsolver_ratio * 100:.1f}%"
            f"（{max_lsolver_notes}），超过 90% 阈值，表明线性求解器"
            f"是总时间的绝对瓶颈。"
        )
    else:
        lines.append(
            f"1. 在 exp_0528 中，LSolver 占比最高为 "
            f"{max_lsolver_ratio * 100:.1f}%（{max_lsolver_notes}）。"
        )
    lines.append("")

    # 2. Highest PrecondApply% in exp_0610
    max_precond_ratio = 0.0
    max_precond_notes = ""
    for exp in group3_data["experiments"]:
        for r in exp["rows"]:
            pb = r.get("pcg_breakdown", {})
            ratio = pb.get("avg_PCG_PreconditionerApply_ms", {}).get("ratio", 0.0)
            if ratio > max_precond_ratio:
                max_precond_ratio = ratio
                max_precond_notes = r.get("notes", "")
    if max_precond_ratio > 0.40:
        lines.append(
            f"2. **预条件器应用是PCG瓶颈**：在 exp_0610 中，"
            f"PrecondApply 占 PCG 比例最高达 {max_precond_ratio * 100:.1f}%"
            f"（{max_precond_notes}），超过 40% 阈值。"
        )
    else:
        lines.append(
            f"2. 在 exp_0610 中，PrecondApply 占 PCG 比例最高为 "
            f"{max_precond_ratio * 100:.1f}%（{max_precond_notes}）。"
        )
    lines.append("")

    # 3. Highest speedup in exp_0615
    max_speedup = 0.0
    max_speedup_desc = ""
    comparisons = group4_data["comparisons"]
    for group_name, rows in comparisons.items():
        for r in rows:
            sp = r.get("speedup")
            if sp is not None and sp > max_speedup:
                max_speedup = sp
                max_speedup_desc = f"{group_name}: {r.get('notes', '')}"
    supp_rows = group4_data["supplement_5e8"]["rows"]
    for r in supp_rows:
        sp = r.get("speedup_vs_baseline", 0.0)
        if sp > max_speedup:
            max_speedup = sp
            max_speedup_desc = f"supplement_5e8: {r.get('notes', '')}"
    if max_speedup > 1.5:
        lines.append(
            f"3. **显著加速**：在 exp_0615 中，最大加速比为 {max_speedup:.3f}"
            f"（{max_speedup_desc}），超过 1.5x 阈值。"
        )
    else:
        lines.append(
            f"3. 在 exp_0615 中，最大加速比为 {max_speedup:.3f}"
            f"（{max_speedup_desc}）。"
        )
    lines.append("")

    # 4. breakdown_sum anomalies
    lines.append("4. **breakdown_sum 异常行**（偏离 1.0 超过 ±5%）：")
    lines.append("")
    anomaly_lines = []

    def _check_sums(rows, source, sum_fields):
        for r in rows:
            for field in sum_fields:
                val = r.get(field)
                if val is not None and abs(val - 1.0) > 0.05:
                    anomaly_lines.append(
                        f"   - {source}: {r.get('notes', '?')} "
                        f"→ {field} = {val}"
                    )

    for exp in group1_data["experiments"]:
        _check_sums(exp["rows"], f"{exp['experiment']}/{exp['subdir']}",
                     ["breakdown_sum"])
    for exp in group2_data.get("experiments", []):
        _check_sums(exp["rows"], f"{exp['experiment']}/{exp['subdir']}",
                     ["breakdown_sum", "lsolver_breakdown_sum"])
    for exp in group3_data["experiments"]:
        _check_sums(exp["rows"], f"{exp['experiment']}/{exp['subdir']}",
                     ["breakdown_sum", "lsolver_breakdown_sum", "pcg_breakdown_sum"])
    # group4
    for r in group4_data["baseline"]["all_rows"]:
        _check_sums([r], "exp_0615/diag_baseline",
                     ["breakdown_sum", "lsolver_breakdown_sum", "pcg_breakdown_sum"])
    for gname, rows in comparisons.items():
        _check_sums(rows, f"exp_0615/{gname}",
                     ["breakdown_sum", "lsolver_breakdown_sum", "pcg_breakdown_sum"])
    _check_sums(supp_rows, "exp_0615/supplement_5e8",
                 ["breakdown_sum", "lsolver_breakdown_sum", "pcg_breakdown_sum"])

    if anomaly_lines:
        for al in anomaly_lines:
            lines.append(al)
    else:
        lines.append("   - 无异常行，所有 breakdown_sum 均在 ±5% 范围内。")
    lines.append("")

    return "\n".join(lines)


def generate_markdown(group1_data, group2_data, group3_data, group4_data,
                      timestamp, total_files):
    """Generate the complete Markdown analysis report."""
    parts = []

    # Header
    parts.append("# 实验数据分析报告")
    parts.append("")
    parts.append(f"> 生成时间：{timestamp}")
    parts.append(f"> 数据来源：docs/json_output/ ({total_files} 个实验数据文件)")
    parts.append("")

    # Section 1: Overview
    parts.append("## 一、实验概述")
    parts.append("")
    parts.append("- 6个实验日期（exp_0528 ~ exp_0615），32个子实验")
    parts.append("- 场景：ArmadilloStand_0 (Stanford 数据集)")
    parts.append("- 求解器：StiffGIPC(CEMAS+SRBK)")
    parts.append("- 三种数据粒度：总时间分解(17列) → LSolver分解(22列) → PCG分解(26列)")
    parts.append("")

    # Section 2: Group 1
    parts.append("## 二、exp_0528 + exp_0604：总时间分解分析")
    parts.append("")
    parts.append(generate_group1_tables(group1_data))

    # Section 3: Group 2
    parts.append("## 三、exp_0608 + exp_0609：LSolver内部分解分析")
    parts.append("")
    parts.append(generate_group2_tables(group2_data))

    # Section 4: Group 3
    parts.append("## 四、exp_0610：PCG内部分解分析")
    parts.append("")
    parts.append(generate_group3_tables(group3_data))

    # Section 5: Group 4
    parts.append("## 五、exp_0615：优化策略对比分析")
    parts.append("")
    parts.append(generate_group4_tables(group4_data))

    # Section 6: Conclusions
    parts.append(generate_conclusions(group1_data, group2_data, group3_data, group4_data))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Run all four analysis groups
    group1_data = analyze_group1(timestamp)
    group2_data = analyze_group2(timestamp)
    group3_data = analyze_group3(timestamp)
    group4_data = analyze_group4(timestamp)

    # Write analysis JSON files
    outputs = [
        ("group1_exp0528_0604_total_breakdown.json", group1_data),
        ("group2_exp0608_0609_lsolver_breakdown.json", group2_data),
        ("group3_exp0610_pcg_breakdown.json", group3_data),
        ("group4_exp0615_optimization_comparison.json", group4_data),
    ]
    for filename, data in outputs:
        path = OUTPUT_DIR / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Written: {path}")

    # Count total input files
    total_files = len(
        [f for f in INPUT_DIR.glob("exp_*__*.json")]
    )

    # Generate Markdown report
    report = generate_markdown(
        group1_data, group2_data, group3_data, group4_data,
        timestamp, total_files,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write("\n")
    print(f"Written: {REPORT_PATH}")

    print(f"\nAnalysis complete: {len(outputs)} JSON files + 1 Markdown report.")


if __name__ == "__main__":
    main()
