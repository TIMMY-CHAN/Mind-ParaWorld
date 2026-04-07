#!/usr/bin/env python3
"""【Setting C】截断到 k 的累计 FCR 曲线（单模型）

用途
- 为某个模型选择一个"足够长程且样本量仍可接受"的截断 k（例如 minimax 取 k=28），
  固定 cohort：仅使用 effective_calls >= k 的样本（例如 minimax 在 k=28 时 n≈181）。
  然后在 x=1..k 范围内绘制累计 FCR(x) 的均值曲线。

右轴：累计 Hit Precision(<=k)
- 定义与 Setting C 指标一致：hit_precision = total_hits / valid_tool_calls。
- 在可视化里，对每个 k 画：HitPrecision(<=k) = (hits up to k) / k。
  其中 hit 来自 trajectory_log.hit_logs[i].hit（0/1）。

说明
- 不再使用 std 阴影带（长程样本异质性极强，std 带通常会过宽、解释成本高）。
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'fcr_vs_toolcalls_data.json'

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

    # X 轴配置
    'x_major_locator': 4,    # X 轴主刻度间距（网格纵向间距）
    'x_lim': None,  # X 轴范围，None 表示自适应，或设置如 (0, 32) 固定

    # 左轴（FCR）配置
    'y_left_major_locator': 0.1,   # 左轴主刻度间距
    'y_left_lim': (0, 0.6),         # 左轴固定范围

    # 右轴（Hit Precision）配置
    'y_right_major_locator': 0.1,  # 右轴主刻度间距（改为 0.1，与左轴一致）
    'y_right_lim': (0, 0.6),        # 右轴固定范围

    # 手工指定模型颜色（按模型名映射，与 cohort 图保持一致）
    'manual_colors': {
        'minimax-m2_1': '#DC8B70',      # 红色
        'MindWatcher_round2': '#1E50A1', # 蓝色
        'licloud_azure-gpt-5': '#11616B',
        'licloud_volcengine-kimi-k2-250711': '#11616B'
        # 可在此添加更多模型的颜色映射
    },

    # 右轴虚线颜色（Hit Precision）
    'hit_precision_color': '0.25',  # 深灰色
}

# ============================================================


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _series_mean_at_k(series_list, k: int, key: str):
    vals = []
    for s in series_list:
        arr = s.get(key, [])
        if len(arr) >= k:
            vals.append(float(arr[k - 1]))
    if not vals:
        return None
    return sum(vals) / len(vals), len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(DEFAULT_INPUT_PATH))
    p.add_argument('--model', required=True, help='Model name key in fcr_vs_toolcalls_data.json')
    p.add_argument('--k', type=int, required=True, help='Cohort threshold: only samples with effective_calls >= k are included; x-axis plots 1..k')
    p.add_argument('--out', required=True, help='Output png path')
    args = p.parse_args()

    # 使用全局配置
    manual_colors = PLOT_CONFIG['manual_colors']
    figsize = PLOT_CONFIG['figsize']
    tick_fontsize = PLOT_CONFIG['tick_fontsize']
    grid_alpha = PLOT_CONFIG['grid_alpha']
    x_major_locator = PLOT_CONFIG['x_major_locator']
    y_left_major_locator = PLOT_CONFIG['y_left_major_locator']
    y_right_major_locator = PLOT_CONFIG['y_right_major_locator']
    y_left_lim = PLOT_CONFIG['y_left_lim']
    y_right_lim = PLOT_CONFIG['y_right_lim']
    x_lim = PLOT_CONFIG['x_lim']
    hit_precision_color = PLOT_CONFIG['hit_precision_color']

    # 如果模型不在字典中，使用 matplotlib 默认蓝色
    model_color = manual_colors.get(args.model, '#1f77b4')

    data = _load(Path(args.input))
    if args.model not in data:
        raise SystemExit(f"model '{args.model}' not found in {args.input}. keys={list(data.keys())}")

    series = (data[args.model].get('model_data', {}) or {}).get('per_sample_series', []) or []
    if not series:
        raise SystemExit(f"model '{args.model}' has empty per_sample_series")

    cohort = [s for s in series if int(s.get('effective_calls', 0)) >= args.k]
    if not cohort:
        raise SystemExit(f"model '{args.model}' has no samples with effective_calls >= {args.k}")

    xs, fcr_means, hit_precisions = [], [], []
    n_cohort = len(cohort)
    for k in range(1, args.k + 1):
        st_fcr = _series_mean_at_k(cohort, k, key='fcr_by_call')
        if st_fcr is None:
            continue
        fcr_mean, _n1 = st_fcr

        # cumulative hit precision up to k: (hits<=k)/k, then averaged across samples
        vals = []
        for s in cohort:
            hits = s.get('hit_by_call', [])
            if len(hits) >= k:
                vals.append(sum(int(x) for x in hits[:k]) / k)
        if not vals:
            continue
        hit_prec = sum(vals) / len(vals)

        xs.append(k)
        fcr_means.append(fcr_mean)
        hit_precisions.append(hit_prec)

    if not xs:
        raise SystemExit('no points to plot')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, fcr_means, marker='o', linewidth=2, markersize=3, color=model_color)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_ylim(y_left_lim)  # 固定左轴范围
    if x_lim is not None:
        ax.set_xlim(x_lim)  # 固定 X 轴范围（如果配置了）
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.yaxis.set_major_locator(MultipleLocator(y_left_major_locator))
    ax.grid(True, alpha=grid_alpha)

    ax2 = ax.twinx()
    ax2.plot(xs, hit_precisions, linestyle='--', linewidth=1.6, alpha=0.7, color=hit_precision_color)
    ax2.set_ylabel('')
    ax2.set_ylim(y_right_lim)  # 固定右轴范围
    ax2.tick_params(axis='y', which='major', labelsize=tick_fontsize)
    ax2.yaxis.set_major_locator(MultipleLocator(y_right_major_locator))

    ax.set_title('')

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
