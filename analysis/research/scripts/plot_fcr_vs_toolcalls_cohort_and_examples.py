#!/usr/bin/env python3
"""【Setting C】cohort 曲线 + 个例轨迹：FCR vs 有效工具调用

输入
- analysis/fcr_vs_toolcalls_analysis.py 生成的 fcr_vs_toolcalls_data.json。
  其中 model_data.per_sample_series 为逐样本的逐 call 序列（fcr_by_call / new_facts_by_call）。

图像 1：cohort_fcr_curves_calls_ge_{N}.png
- cohort 固定：仅选择 effective_calls >= N 的样本。
- 横轴 k：第 k 次"有效工具调用"（message-constrained）。
- 纵轴：累计 FCR(k) 的均值；阴影为 IQR（25%~75%）。
- 注意：虽然 cohort 以 calls>=N 固定，但当 k>N 时，仍只有 calls>=k 的子集会参与统计，
  因此 n(k) 会逐步下降；讨论饱和时建议同时关注 n(k)。

图像 2：cohort_marginal_newfacts_calls_ge_{N}.png
- 同一 cohort 下，每次调用带来的新增 unique facts 数（new_facts_by_call）的均值与 IQR。
  可用于直观看"边际收益是否趋近 0"。

图像 3：examples_{model}_calls_{lo}_{hi}.png
- 从 effective_calls ∈ [lo, hi] 的样本中，为每个模型随机抽取若干条轨迹，
  绘制其逐 call 的累计 FCR 曲线，用于展示代表性个例（非统计结论）。
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'fcr_vs_toolcalls_data.json'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'analysis_results' / 'fcr_vs_toolcalls_v2'

# ============================================================
# 图表视觉配置区（手工指定，保证论文中多图一致性）
# ============================================================

PLOT_CONFIG = {
    # 图像尺寸（英寸）
    'figsize': (10, 8),  # 宽 × 高

    # 刻度字号
    'tick_fontsize': 16,

    # 网格设置
    'grid_alpha': 0.3,
    'x_major_locator': 4,   # X 轴主刻度间距（每隔 4 个单位一个主刻度）
    'y_major_locator': 0.1, # Y 轴主刻度间距（左轴，marginal new facts）

    # 统计显著性阈值（用于图一）
    # 当 n(k) < 该阈值时，折线变为高透明度，并在阈值处画竖线
    'significance_threshold': 50,  # n(k) 阈值
    'low_significance_alpha': 0.25,  # 低统计意义时的折线透明度
    'threshold_line_style': '--',   # 阈值竖线样式
    'threshold_line_color': '#11616B', # 阈值竖线颜色
    'threshold_line_alpha': 0.7,    # 阈值竖线透明度

    # 手工指定模型颜色（按模型名映射，与 cohort 图保持一致）
    'manual_colors': {
        'minimax-m2_1': '#DC8B70',      # 红色
        'MindWatcher_round2': '#1E50A1', # 蓝色
        'licloud_azure-gpt-5': '#11616B',
        'licloud_volcengine-kimi-k2-250711': '#11616B'
        # 可在此添加更多模型的颜色映射
    },
}

# ============================================================


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _set_axes_nk_fill(ax, xs, ns, color, alpha=0.12):
    """右轴绘制 n(k) 折线，并在折线下方填充阴影。"""
    if not xs:
        return None
    ax2 = ax.twinx()
    ax2.plot(xs, ns, color=color, linewidth=1.2, alpha=0.9)
    ax2.fill_between(xs, [0] * len(xs), ns, color=color, alpha=alpha, linewidth=0)
    ax2.set_ylabel('n(k)')
    return ax2


# 保留旧实现（背景 mask）以便需要时回退/对比

def _set_axes_mask_by_n(ax, xs, ns, color='0.85'):
    """在已有主轴 ax 上，按 n(k) 画右轴并用背景阴影表示样本量下降。

    设计目的：
    - 左轴保留核心指标（new_facts 均值）。
    - 右轴展示 n(k)，并用 mask 让读者直观看到“尾部样本量在变小”。
    """
    if not xs:
        return None
    ax2 = ax.twinx()
    ax2.plot(xs, ns, color='black', linewidth=1.2, alpha=0.75, label='n(k)')
    ax2.set_ylabel('n(k)')

    n0 = float(ns[0]) if ns and ns[0] else 1.0
    ratios = [(float(n) / n0) if n0 > 0 else 0.0 for n in ns]

    # mask：样本量越少，背景越深（最大到 0.6 的 alpha）
    for i in range(len(xs)):
        alpha = min(0.6, max(0.0, 1.0 - ratios[i]) * 0.6)
        ax.axvspan(xs[i] - 0.5, xs[i] + 0.5, color=color, alpha=alpha, linewidth=0)

    return ax2


def _percentile(sorted_vals, q: float):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _series_stats_at_k(series_list, k: int):
    """Collect values at call index k (1-indexed)."""
    vals = []
    for s in series_list:
        arr = s.get('fcr_by_call', [])
        if len(arr) >= k:
            vals.append(float(arr[k - 1]))
    vals.sort()
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    p25 = _percentile(vals, 0.25)
    p75 = _percentile(vals, 0.75)
    return mean, p25, p75, len(vals)


def _newfacts_stats_at_k(series_list, k: int):
    vals = []
    for s in series_list:
        arr = s.get('new_facts_by_call', [])
        if len(arr) >= k:
            vals.append(float(arr[k - 1]))
    vals.sort()
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    p25 = _percentile(vals, 0.25)
    p75 = _percentile(vals, 0.75)
    return mean, p25, p75, len(vals)


def plot_cohort_curves(
    data: dict,
    out_dir: Path,
    min_calls: int,
    max_calls: int = 32,
    per_model_kmax: dict | None = None,
    per_model_min_calls: dict | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model_kmax = per_model_kmax or {}
    per_model_min_calls = per_model_min_calls or {}

    # 使用全局配置
    manual_colors = PLOT_CONFIG['manual_colors']
    figsize = PLOT_CONFIG['figsize']
    tick_fontsize = PLOT_CONFIG['tick_fontsize']
    grid_alpha = PLOT_CONFIG['grid_alpha']
    x_major_locator = PLOT_CONFIG['x_major_locator']
    y_major_locator = PLOT_CONFIG['y_major_locator']
    significance_threshold = PLOT_CONFIG['significance_threshold']
    low_significance_alpha = PLOT_CONFIG['low_significance_alpha']
    threshold_line_style = PLOT_CONFIG['threshold_line_style']
    threshold_line_color = PLOT_CONFIG['threshold_line_color']
    threshold_line_alpha = PLOT_CONFIG['threshold_line_alpha']

    # cohort filter per model
    cohort = {}
    for model_name, payload in data.items():
        series = (payload.get('model_data', {}) or {}).get('per_sample_series', []) or []
        min_calls_i = int(per_model_min_calls.get(model_name, min_calls))
        series = [s for s in series if int(s.get('effective_calls', 0)) >= min_calls_i]
        if series:
            cohort[model_name] = series

    # --- FCR curve ---
    plt.figure(figsize=figsize)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    for i, (model_name, series) in enumerate(cohort.items()):
        xs, means, p25s, p75s, ns = [], [], [], [], []
        for k in range(1, max_calls + 1):
            st = _series_stats_at_k(series, k)
            if st is None:
                continue
            mean, p25, p75, n = st
            xs.append(k)
            means.append(mean)
            p25s.append(p25)
            p75s.append(p75)
            ns.append(n)
        if not xs:
            continue

        # 使用手工指定颜色或默认颜色
        color = manual_colors.get(model_name, prop_cycle[i % len(prop_cycle)] if prop_cycle else None)

        plt.plot(xs, means, marker='o', linewidth=2, markersize=3, label=model_name, color=color)
        plt.fill_between(xs, p25s, p75s, alpha=0.15, color=color)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # FCR 用 0.2 间距

    plt.xlabel('Call index k (effective tool calls)')
    plt.ylabel('Cumulative FCR after k calls')
    plt.title(f'Cohort fixed: samples with calls >= {min_calls} (showing k=1..{max_calls}); note n(k) shrinks for k>{min_calls}')
    plt.grid(True, alpha=grid_alpha)
    plt.legend()
    plt.tight_layout()
    f1 = out_dir / f'cohort_fcr_curves_calls_ge_{min_calls}.png'
    plt.savefig(f1, dpi=200, bbox_inches='tight')
    plt.close()

    # --- marginal new facts (mean) + per-model n(k) fill on secondary axis ---
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # 记录所有模型的阈值 k 位置（用于画竖线）
    threshold_ks = set()

    # 先画左轴：new facts mean（每模型可截断到不同 k_max）
    for i, (model_name, series) in enumerate(cohort.items()):
        # 这里的 k_max 仅用于控制"画到多长"，不影响样本筛选；
        # 如果不指定，则默认画满 max_calls。
        k_max_i = int(per_model_kmax.get(model_name, max_calls))
        k_max_i = max(1, min(max_calls, k_max_i))

        xs, means, ns = [], [], []
        for k in range(1, k_max_i + 1):
            st = _newfacts_stats_at_k(series, k)
            if st is None:
                continue
            mean, _p25, _p75, n = st
            xs.append(k)
            means.append(mean)
            ns.append(n)
        if not xs:
            continue

        # 使用手工指定颜色或默认颜色
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        color = manual_colors.get(model_name, prop_cycle[i % len(prop_cycle)] if prop_cycle else None)

        # 找到第一个 n(k) < significance_threshold 的位置
        threshold_idx = None
        for idx, n in enumerate(ns):
            if n < significance_threshold:
                threshold_idx = idx
                if xs[idx] not in threshold_ks:
                    threshold_ks.add(xs[idx])
                break

        # 分段绘制：阈值前用正常透明度，阈值后用低透明度
        if threshold_idx is None:
            # 所有点都在阈值以上
            ax.plot(xs, means, marker='o', linewidth=2, markersize=3, color=color, alpha=1.0)
        elif threshold_idx == 0:
            # 所有点都在阈值以下
            ax.plot(xs, means, marker='o', linewidth=2, markersize=3, color=color, alpha=low_significance_alpha)
        else:
            # 分段：[0, threshold_idx] 正常，[threshold_idx, end] 低透明度
            ax.plot(xs[:threshold_idx+1], means[:threshold_idx+1], marker='o', linewidth=2, markersize=3, color=color, alpha=1.0)
            ax.plot(xs[threshold_idx:], means[threshold_idx:], marker='o', linewidth=2, markersize=3, color=color, alpha=low_significance_alpha)

    # 画阈值竖线
    for threshold_k in sorted(threshold_ks):
        ax.axvline(x=threshold_k, linestyle=threshold_line_style, color=threshold_line_color,
                   alpha=threshold_line_alpha, linewidth=1.5, zorder=0)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_ylim(0, 0.6)  # 固定左轴刻度范围
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    ax.grid(True, alpha=grid_alpha)

    # 右轴：仅用 fill 表示每个模型的 n(k)（不画折线，避免花）
    ax2 = ax.twinx()
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', which='major', labelsize=tick_fontsize)

    for i, (model_name, series) in enumerate(cohort.items()):
        min_calls_i = int(per_model_min_calls.get(model_name, min_calls))
        k_max_i = int(per_model_kmax.get(model_name, max_calls))
        k_max_i = max(1, min(max_calls, k_max_i))

        xs, ns = [], []
        for k in range(1, k_max_i + 1):
            st = _newfacts_stats_at_k(series, k)
            if st is None:
                continue
            _mean, _p25, _p75, n = st
            xs.append(k)
            ns.append(n)
        if not xs:
            continue

        # 使用与折线相同的颜色
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        color = manual_colors.get(model_name, prop_cycle[i % len(prop_cycle)] if prop_cycle else None)

        ax2.fill_between(xs, [0] * len(xs), ns, color=color, alpha=0.10, linewidth=0)

    # legend 仅保留左轴模型名
    #（按需由用户在论文/报告中自行标注）
    plt.tight_layout()
    f2 = out_dir / f'cohort_marginal_newfacts_per_model_calls_ge.png'
    plt.savefig(f2, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {f1}")
    print(f"Saved: {f2}")


def plot_examples_per_model(data: dict, out_dir: Path, lo: int, hi: int, k_max: int = 32, n_examples: int = 5, seed: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for model_name, payload in data.items():
        series = (payload.get('model_data', {}) or {}).get('per_sample_series', []) or []
        bucket = [s for s in series if lo <= int(s.get('effective_calls', 0)) <= hi]
        if len(bucket) < n_examples:
            continue

        chosen = rng.sample(bucket, n_examples)

        plt.figure(figsize=(10, 6))
        for s in chosen:
            idx = s.get('index')
            arr = s.get('fcr_by_call', [])
            xs = list(range(1, min(len(arr), k_max) + 1))
            ys = [float(x) for x in arr[: len(xs)]]
            plt.plot(xs, ys, linewidth=1.8, alpha=0.9, label=f"idx={idx}, calls={len(arr)}")

        plt.xlabel('Call index k')
        plt.ylabel('Cumulative FCR')
        plt.title(f'{model_name}: 5 example trajectories (calls in [{lo},{hi}])')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = out_dir / f'examples_{model_name}_calls_{lo}_{hi}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default=str(DEFAULT_INPUT_PATH),
    )
    parser.add_argument(
        '--out-dir',
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument('--cohort-min-calls', type=int, default=16)
    parser.add_argument('--examples-lo', type=int, default=18)
    parser.add_argument('--examples-hi', type=int, default=20)
    parser.add_argument('--examples-n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = _load(Path(args.input))
    out_dir = Path(args.out_dir)

    plot_cohort_curves(data, out_dir, min_calls=args.cohort_min_calls)
    plot_examples_per_model(
        data,
        out_dir,
        lo=args.examples_lo,
        hi=args.examples_hi,
        n_examples=args.examples_n,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
