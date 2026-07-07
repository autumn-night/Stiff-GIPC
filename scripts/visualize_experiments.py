#!/usr/bin/env python3
"""可视化三个分解实验的分析结果

生成 8 张图表（饼状图和堆叠柱状图）到 analysis/figures/ 目录。
使用方法：conda run -n daily python scripts/visualize_experiments.py
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis" / "json_output" / "analysis"
FIGURES_DIR = ROOT / "analysis" / "figures"

# ==================== 常量定义 ====================

SWEEP_PARAM = {
    'exp_sweep_bend': 'bend',
    'exp_sweep_dhat': 'dhat',
    'exp_sweep_friction': 'friction',
    'exp_sweep_stiffness': 'poisson',
    'exp_sweep_stretch': 'stretch',
    'exp_sweep_timestep': 'dt',
}

SWEEP_ORDER = [
    'exp_sweep_bend',
    'exp_sweep_dhat',
    'exp_sweep_friction',
    'exp_sweep_stiffness',
    'exp_sweep_stretch',
    'exp_sweep_timestep',
]

SWEEP_TITLES = {
    'exp_sweep_bend': 'Bend Stiffness Sweep',
    'exp_sweep_dhat': 'Contact Distance (dhat) Sweep',
    'exp_sweep_friction': 'Friction Sweep',
    'exp_sweep_stiffness': 'Poisson Ratio Sweep',
    'exp_sweep_stretch': 'Stretch Stiffness Sweep',
    'exp_sweep_timestep': 'Time Step Sweep',
}

# Group 1: 4 components (total time breakdown)
FIELDS_TOTAL = ['avg_Hess_ms', 'avg_LSolver_ms', 'avg_LineS_ms', 'avg_Misc_ms']
LABELS_TOTAL = {
    'avg_Hess_ms': 'Hessian',
    'avg_LSolver_ms': 'Linear Solver',
    'avg_LineS_ms': 'Line Search',
    'avg_Misc_ms': 'Misc',
}
COLORS_TOTAL = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# Group 2: 5 components (LSolver internal breakdown)
FIELDS_LSOLVER = [
    'avg_LSolver_SubsystemAssemble_ms',
    'avg_LSolver_TripletOps_ms',
    'avg_LSolver_PreconditionerAssemble_ms',
    'avg_LSolver_PCG_ms',
    'avg_LSolver_SolutionDistribute_ms',
]
LABELS_LSOLVER = {
    'avg_LSolver_SubsystemAssemble_ms': 'Subsys Assemble',
    'avg_LSolver_TripletOps_ms': 'Triplet Ops',
    'avg_LSolver_PreconditionerAssemble_ms': 'Precond Assemble',
    'avg_LSolver_PCG_ms': 'PCG',
    'avg_LSolver_SolutionDistribute_ms': 'Solution Distribute',
}
COLORS_LSOLVER = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# Group 3: 4 components (PCG internal breakdown)
FIELDS_PCG = [
    'avg_PCG_PreconditionerApply_ms',
    'avg_PCG_SpMV_ms',
    'avg_PCG_Dot_ms',
    'avg_PCG_Axpby_ms',
]
LABELS_PCG = {
    'avg_PCG_PreconditionerApply_ms': 'Precond Apply',
    'avg_PCG_SpMV_ms': 'SpMV',
    'avg_PCG_Dot_ms': 'Dot',
    'avg_PCG_Axpby_ms': 'Axpby',
}
COLORS_PCG = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# 全局样式
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ==================== 工具函数 ====================

def load_group(filename):
    """加载分析 JSON"""
    with open(ANALYSIS_DIR / filename, encoding='utf-8') as f:
        return json.load(f)


def get_sweep_experiments(group_data):
    """过滤出 6 个 sweep 实验，跳过 interaction/focus，按 SWEEP_ORDER 排序"""
    sweep_subdirs = set(SWEEP_ORDER)
    sweeps = [e for e in group_data['experiments'] if e['subdir'] in sweep_subdirs]
    # 按 SWEEP_ORDER 排序
    sweeps.sort(key=lambda e: SWEEP_ORDER.index(e['subdir']))
    return sweeps


def extract_param_value(notes, subdir):
    """从 notes 字段提取参数值作为 X 轴标签"""
    param = SWEEP_PARAM.get(subdir)
    if not param:
        return notes
    match = re.search(rf'{param}=([\d.eE+-]+)', notes)
    return match.group(1) if match else notes


def param_sort_key(notes, subdir):
    """将参数值转为 float 用于排序"""
    val_str = extract_param_value(notes, subdir)
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return float('inf')


def sort_rows(rows, subdir):
    """按参数值数值排序行"""
    return sorted(rows, key=lambda r: param_sort_key(r['notes'], subdir))


def find_row_by_note(rows, keyword):
    """通过 notes 关键字查找特定行"""
    for r in rows:
        if keyword in r['notes']:
            return r
    return None


def plot_stacked_bar(ax, rows, subdir, breakdown_key, field_order, labels, colors):
    """绘制单个子图的堆叠柱状图"""
    x_labels = [extract_param_value(r['notes'], subdir) for r in rows]
    bottoms = np.zeros(len(rows))
    for i, field in enumerate(field_order):
        values = [r[breakdown_key][field]['value_ms'] for r in rows]
        ax.bar(x_labels, values, bottom=bottoms, label=labels[field],
               color=colors[i], edgecolor='white', linewidth=0.5)
        bottoms += np.array(values)
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Time (ms)')
    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')


def plot_pie(ax, row, breakdown_key, field_order, labels, colors, title):
    """绘制单个饼图"""
    values = [row[breakdown_key][field]['value_ms'] for field in field_order]
    pie_labels = [labels[f] for f in field_order]
    ax.pie(values, labels=pie_labels, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 10})
    ax.set_title(title)


def annotate_bar_tops(ax, rows):
    """在每根柱子顶部标注总时间值"""
    for i, r in enumerate(rows):
        total = r.get('avg_TimeTot_ms')
        if total is None:
            # 对于 group2/3，使用 breakdown 中各组件之和
            continue
        label = f'{total:.0f}ms'
        ax.annotate(label, xy=(i, total), ha='center', va='bottom',
                    fontsize=9, fontweight='bold')


# ==================== Group 1: Total Time Breakdown ====================

def visualize_group1():
    """生成 Group 1 的 3 张图"""
    data = load_group('group1_exp0528_0604_total_breakdown.json')
    sweeps = get_sweep_experiments(data)
    # 只取 exp_0528 的（主实验）
    sweeps = [e for e in sweeps if e['experiment'] == 'exp_0528']

    # ---- 图表 1: 2×3 grid 堆叠柱状图 ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes_flat = axes.flatten()
    for idx, exp in enumerate(sweeps):
        ax = axes_flat[idx]
        sorted_rows = sort_rows(exp['rows'], exp['subdir'])
        plot_stacked_bar(ax, sorted_rows, exp['subdir'], 'breakdown',
                         FIELDS_TOTAL, LABELS_TOTAL, COLORS_TOTAL)
        ax.set_title(SWEEP_TITLES[exp['subdir']])
    # 隐藏多余的子图（如果有）
    for idx in range(len(sweeps), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    # 共享图例放在整图底部
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Total Time Breakdown — exp_0528 Parameter Sweeps',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIGURES_DIR / 'group1_total_all_sweeps_stacked.png')
    plt.close(fig)
    print('  [1/3] group1_total_all_sweeps_stacked.png')

    # ---- 图表 2: bend sweep 大堆叠柱状图 ----
    bend_exp = next((e for e in sweeps if e['subdir'] == 'exp_sweep_bend'), None)
    if bend_exp:
        sorted_rows = sort_rows(bend_exp['rows'], 'exp_sweep_bend')
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_stacked_bar(ax, sorted_rows, 'exp_sweep_bend', 'breakdown',
                         FIELDS_TOTAL, LABELS_TOTAL, COLORS_TOTAL)
        ax.set_title('Total Time Breakdown — Bend Stiffness Sweep')
        # 标注总时间值
        for i, r in enumerate(sorted_rows):
            total = r['avg_TimeTot_ms']
            ax.annotate(f'{total:.0f}ms', xy=(i, total), ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        # 留出顶部空间给标注
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.08)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'group1_total_bend_stacked.png')
        plt.close(fig)
        print('  [2/3] group1_total_bend_stacked.png')

    # ---- 图表 3: pie comparison (bend=1e8 vs bend=1e9) ----
    if bend_exp:
        rows = bend_exp['rows']
        row_default = find_row_by_note(rows, 'bend=1e8')
        row_extreme = find_row_by_note(rows, 'bend=1e9')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if row_default:
            plot_pie(axes[0], row_default, 'breakdown', FIELDS_TOTAL,
                     LABELS_TOTAL, COLORS_TOTAL, 'Default (bend=1e8)')
        if row_extreme:
            plot_pie(axes[1], row_extreme, 'breakdown', FIELDS_TOTAL,
                     LABELS_TOTAL, COLORS_TOTAL, 'Extreme (bend=1e9)')
        fig.suptitle('Total Time Distribution Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIGURES_DIR / 'group1_total_bend_pie_comparison.png')
        plt.close(fig)
        print('  [3/3] group1_total_bend_pie_comparison.png')


# ==================== Group 2: LSolver Breakdown ====================

def visualize_group2():
    """生成 Group 2 的 3 张图"""
    data = load_group('group2_exp0608_0609_lsolver_breakdown.json')
    sweeps = get_sweep_experiments(data)
    sweeps = [e for e in sweeps if e['experiment'] == 'exp_0608']

    # ---- 图表 4: 2×3 grid 堆叠柱状图 ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes_flat = axes.flatten()
    for idx, exp in enumerate(sweeps):
        ax = axes_flat[idx]
        sorted_rows = sort_rows(exp['rows'], exp['subdir'])
        plot_stacked_bar(ax, sorted_rows, exp['subdir'], 'lsolver_breakdown',
                         FIELDS_LSOLVER, LABELS_LSOLVER, COLORS_LSOLVER)
        ax.set_title(SWEEP_TITLES[exp['subdir']])
    for idx in range(len(sweeps), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Linear Solver Breakdown — exp_0608 Parameter Sweeps',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIGURES_DIR / 'group2_lsolver_all_sweeps_stacked.png')
    plt.close(fig)
    print('  [1/3] group2_lsolver_all_sweeps_stacked.png')

    # ---- 图表 5: bend sweep 大堆叠柱状图 ----
    bend_exp = next((e for e in sweeps if e['subdir'] == 'exp_sweep_bend'), None)
    if bend_exp:
        sorted_rows = sort_rows(bend_exp['rows'], 'exp_sweep_bend')
        fig, ax = plt.subplots(figsize=(10, 7))
        plot_stacked_bar(ax, sorted_rows, 'exp_sweep_bend', 'lsolver_breakdown',
                         FIELDS_LSOLVER, LABELS_LSOLVER, COLORS_LSOLVER)
        ax.set_title('Linear Solver Breakdown — Bend Stiffness Sweep')
        # 标注 LSolver 总时间值
        for i, r in enumerate(sorted_rows):
            total = r['avg_LSolver_ms']
            ax.annotate(f'{total:.0f}ms', xy=(i, total), ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.08)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'group2_lsolver_bend_stacked.png')
        plt.close(fig)
        print('  [2/3] group2_lsolver_bend_stacked.png')

    # ---- 图表 6: pie comparison (bend=1e7 vs bend=1e9) ----
    if bend_exp:
        rows = bend_exp['rows']
        row_flexible = find_row_by_note(rows, 'bend=1e7')
        row_stiff = find_row_by_note(rows, 'bend=1e9')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if row_flexible:
            plot_pie(axes[0], row_flexible, 'lsolver_breakdown', FIELDS_LSOLVER,
                     LABELS_LSOLVER, COLORS_LSOLVER, 'Flexible (bend=1e7)')
        if row_stiff:
            plot_pie(axes[1], row_stiff, 'lsolver_breakdown', FIELDS_LSOLVER,
                     LABELS_LSOLVER, COLORS_LSOLVER, 'Very Stiff (bend=1e9)')
        fig.suptitle('Linear Solver Internal Distribution Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIGURES_DIR / 'group2_lsolver_bend_pie_comparison.png')
        plt.close(fig)
        print('  [3/3] group2_lsolver_bend_pie_comparison.png')


# ==================== Group 3: PCG Breakdown ====================

def visualize_group3():
    """生成 Group 3 的 2 张图（图表 7 和 8）"""
    data = load_group('group3_exp0610_pcg_breakdown.json')
    sweeps = get_sweep_experiments(data)
    sweeps = [e for e in sweeps if e['experiment'] == 'exp_0610']

    # ---- 图表 7: 2×3 grid 堆叠柱状图 ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes_flat = axes.flatten()
    for idx, exp in enumerate(sweeps):
        ax = axes_flat[idx]
        sorted_rows = sort_rows(exp['rows'], exp['subdir'])
        plot_stacked_bar(ax, sorted_rows, exp['subdir'], 'pcg_breakdown',
                         FIELDS_PCG, LABELS_PCG, COLORS_PCG)
        ax.set_title(SWEEP_TITLES[exp['subdir']])
    for idx in range(len(sweeps), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    handles, lbls = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('PCG Breakdown — exp_0610 Parameter Sweeps',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIGURES_DIR / 'group3_pcg_all_sweeps_stacked.png')
    plt.close(fig)
    print('  [1/2] group3_pcg_all_sweeps_stacked.png')

    # ---- 图表 8: pie comparison (bend=1e7 vs bend=1e9) ----
    bend_exp = next((e for e in sweeps if e['subdir'] == 'exp_sweep_bend'), None)
    if bend_exp:
        rows = bend_exp['rows']
        row_flexible = find_row_by_note(rows, 'bend=1e7')
        row_stiff = find_row_by_note(rows, 'bend=1e9')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        if row_flexible:
            plot_pie(axes[0], row_flexible, 'pcg_breakdown', FIELDS_PCG,
                     LABELS_PCG, COLORS_PCG, 'Flexible (bend=1e7)')
        if row_stiff:
            plot_pie(axes[1], row_stiff, 'pcg_breakdown', FIELDS_PCG,
                     LABELS_PCG, COLORS_PCG, 'Very Stiff (bend=1e9)')
        fig.suptitle('PCG Internal Distribution Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(FIGURES_DIR / 'group3_pcg_bend_pie_comparison.png')
        plt.close(fig)
        print('  [2/2] group3_pcg_bend_pie_comparison.png')


# ==================== 主函数 ====================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print('Generating Group 1 charts...')
    visualize_group1()
    print('Generating Group 2 charts...')
    visualize_group2()
    print('Generating Group 3 charts...')
    visualize_group3()
    print(f'\nAll charts saved to {FIGURES_DIR}')
    # 打印生成的文件列表
    print('\nGenerated files:')
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f'  {f}')


if __name__ == '__main__':
    main()
