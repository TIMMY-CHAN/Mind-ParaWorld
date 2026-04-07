#!/usr/bin/env python3
"""
FCR vs Pass@1 关系图

用途：
- 展示单个模型在 Setting B+C 中，样本级别的 FCR 与 Pass@1 的关系
- Setting A 作为上界虚线（FCR≈100%）
- 用于机制验证：FCR 是 Pass@1 的主要预测因子

数据来源：
- Setting B: setting_b_results/<model>/<prompt>/re_evaluated_results.jsonl
- Setting C: setting_c_results/<model>/re_evaluated_results.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import make_interp_spline

# ============================================================
# 配置区
# ============================================================
    # 'manual_colors': {
    #     'minimax-m2_1': '#DC8B70',      # 红色
    #     'MindWatcher_round2': '#1E50A1', # 蓝色
    #     'licloud_azure-gpt-5': '#11616B',
    #     'licloud_volcengine-kimi-k2-250711': '#11616B',
    #     'Qwen32B':'#763262'
    #     # 可在此添加更多模型的颜色映射
    # },
PLOT_CONFIG = {
    # 图像尺寸
    'figsize': (10, 8),

    # 字号
    'tick_fontsize': 16,
    'label_fontsize': 18,

    # 分桶设置
    'binning_method': 'fixed',  # 'fixed' 或 'quantile'
    'fixed_bin_width': 0.1,      # 固定间隔分桶宽度
    'quantile_n_bins': 10,       # 等频分桶数量
    'min_samples_per_bin': 5,    # 每桶最小样本数（警告阈值）

    # 网格设置
    'grid_alpha': 0.3,
    'x_major_locator': 0.1,
    'y_major_locator': 0.1,

    # 曲线样式
    'bc_curve_color': '#FD465D',      # Setting B+C 曲线颜色
    'bc_scatter_color': '#FD465D',    # 散点颜色
    'bc_scatter_alpha': 0.5,          # 散点透明度
    'bc_scatter_size': 80,            # 散点大小
    'bc_curve_linewidth': 2.5,        # 曲线粗细

    # Setting A 上界虚线
    'settingA_line_color': '#FD465D',
    'settingA_line_style': '--',
    'settingA_line_width': 2.0,
    'settingA_line_alpha': 0.7,

    # 拟合曲线平滑度
    'spline_smoothing': True,   # 是否使用样条插值平滑
    'spline_points': 100,        # 样条插值点数
}

# Setting A 上界数据（手工填入，从 summary 表格中读取）
SETTING_A_PASS1 = {
    'MindWatcher': 85.0,  # 从 setting_A_results/MindWatcher_32B/difficulty_pass1_stats.json
    'minimax-m2_1': 80.0,
    # 添加更多模型...
}

# Setting B/C 的 (FCR, Pass@1) 数据（从 analysis_results/*.md 读取）
# 用于在图上标注模型级别的整体表现
SETTING_POINTS = {
    'MindWatcher': {
        'setting_a': (1.0, 85.0),  # FCR=1.0 (oracle), Pass@1=85.0%
        'setting_b_guidance': (0.396, 44.15),
        'setting_b_fewshot': (0.414, 47.51),
        'setting_c': (0.369, 39.37),  # MindWatcher_round1
    },
    'minimax-m2_1': {
        'setting_a': (1.0, 80.0),  # 需要填入真实值
        'setting_b_guidance': (0.0, 0.0),  # 需要填入
        'setting_b_fewshot': (0.0, 0.0),
        'setting_c': (0.0, 0.0),
    },
    # 添加更多模型...
}

# ============================================================


def load_sample_data(file_path: Path):
    """
    从 re_evaluated_results.jsonl 中加载样本级数据

    Returns:
        List[Tuple[float, int]]: [(fcr, pass1), ...]
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)

                # 获取 FCR（从 metrics 或 trajectory_log）
                fcr = None
                if 'metrics' in sample and 'fcr' in sample['metrics']:
                    fcr = float(sample['metrics']['fcr'])
                elif 'trajectory_log' in sample and 'world_truth_info' in sample['trajectory_log']:
                    # 如果没有 metrics.fcr，从 trajectory_log 计算
                    traj = sample['trajectory_log']
                    world_truth = traj.get('world_truth_info', {})
                    total_facts = len(world_truth.get('atomic_facts', {}))

                    if total_facts > 0:
                        covered_facts = set()
                        for hit_log in traj.get('hit_logs', []):
                            matched_keys = hit_log.get('matched_fact_keys', [])
                            if matched_keys:
                                covered_facts.update(matched_keys)
                        fcr = len(covered_facts) / total_facts

                # 获取 Pass@1（0 或 1）
                pass1 = None

                # Setting B uses 'answer' field directly
                if 'answer' in sample:
                    answer = sample.get('answer', '')
                    if answer == 'Correct':
                        pass1 = 1
                    elif answer in ['Incorrect', 'No Answer']:
                        pass1 = 0
                # Setting C uses 'judge_status' + 'judge_response'
                elif 'judge_status' in sample:
                    judge_status = sample.get('judge_status', '')
                    if judge_status == 'success':
                        # Parse judgment from judge_response
                        judge_resp = sample.get('judge_response', '')
                        if isinstance(judge_resp, str):
                            match = re.search(r'<answer>(.*?)</answer>', judge_resp, re.DOTALL)
                            if match:
                                judgment = match.group(1).strip()
                                if 'Correct' in judgment or 'correct' in judgment:
                                    pass1 = 1
                                else:
                                    pass1 = 0
                    elif judge_status == 'skipped':
                        # Skipped samples are not evaluated, treat as incorrect
                        pass1 = 0

                if fcr is not None and pass1 is not None:
                    # 过滤异常FCR值（应该在0-1之间）
                    if 0 <= fcr <= 1.0:
                        samples.append((fcr, pass1))
                    else:
                        # 异常值，跳过
                        continue

            except Exception as e:
                continue

    return samples


def bin_data_fixed(samples, bin_width=0.1):
    """固定间隔分桶"""
    bins = defaultdict(list)

    for fcr, pass1 in samples:
        bin_idx = int(fcr / bin_width)
        bin_idx = min(bin_idx, int(1.0 / bin_width) - 1)  # 确保 fcr=1.0 在最后一个桶
        bins[bin_idx].append((fcr, pass1))

    # 计算每桶的FCR均值和平均 Pass@1
    bin_fcr_means = []
    bin_pass1_means = []
    bin_counts = []

    for bin_idx in sorted(bins.keys()):
        bin_samples = bins[bin_idx]
        bin_fcr_mean = np.mean([fcr for fcr, _ in bin_samples])
        bin_pass1_mean = np.mean([pass1 for _, pass1 in bin_samples])
        bin_count = len(bin_samples)

        bin_fcr_means.append(bin_fcr_mean)
        bin_pass1_means.append(bin_pass1_mean)
        bin_counts.append(bin_count)

    return bin_fcr_means, bin_pass1_means, bin_counts


def bin_data_quantile(samples, n_bins=10):
    """等频分桶"""
    if len(samples) < n_bins:
        # 样本太少，退化为固定分桶
        return bin_data_fixed(samples, bin_width=1.0/n_bins)

    fcrs = np.array([fcr for fcr, _ in samples])
    pass1s = np.array([p for _, p in samples])

    # 计算分位数边界
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(fcrs, quantiles)

    # 分配样本到桶
    bins = defaultdict(list)
    for fcr, pass1 in samples:
        bin_idx = np.searchsorted(bin_edges[1:], fcr)
        bins[bin_idx].append((fcr, pass1))

    # 计算每桶的FCR均值和平均 Pass@1
    bin_fcr_means = []
    bin_pass1_means = []
    bin_counts = []

    for bin_idx in sorted(bins.keys()):
        bin_samples = bins[bin_idx]
        # 桶内FCR均值
        bin_fcr_mean = np.mean([fcr for fcr, _ in bin_samples])
        bin_pass1_mean = np.mean([pass1 for _, pass1 in bin_samples])
        bin_count = len(bin_samples)

        bin_fcr_means.append(bin_fcr_mean)
        bin_pass1_means.append(bin_pass1_mean)
        bin_counts.append(bin_count)

    return bin_fcr_means, bin_pass1_means, bin_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name (e.g., MindWatcher)')
    parser.add_argument('--merge-prompts', action='store_true', default=True,
                        help='Merge both Guidance and Few-shot prompts from Setting B (default: True)')
    parser.add_argument('--no-merge-prompts', action='store_false', dest='merge_prompts',
                        help='Do not merge prompts, use only the specified prompt type')
    parser.add_argument('--setting-b-prompt', default='Guidance', help='Setting B prompt type when not merging (Guidance or Few-shot)')
    parser.add_argument('--setting-a-pass1', type=float, help='Setting A Pass@1 for upper bound (if not in config)')
    parser.add_argument('--setting-b-guidance-fcr', type=float, help='Setting B Guidance FCR')
    parser.add_argument('--setting-b-guidance-pass1', type=float, help='Setting B Guidance Pass@1')
    parser.add_argument('--setting-b-fewshot-fcr', type=float, help='Setting B Few-shot FCR')
    parser.add_argument('--setting-b-fewshot-pass1', type=float, help='Setting B Few-shot Pass@1')
    parser.add_argument('--setting-c-fcr', type=float, help='Setting C FCR')
    parser.add_argument('--setting-c-pass1', type=float, help='Setting C Pass@1')
    parser.add_argument('--binning-method', default='quantile', choices=['fixed', 'quantile'],
                        help='Binning method: fixed (固定间隔) or quantile (等频分桶, default)')
    parser.add_argument('--n-bins', type=int, default=15, help='Number of bins for quantile binning (default: 15)')
    parser.add_argument('--bin-width', type=float, default=0.1, help='Bin width for fixed binning')
    parser.add_argument('--output', required=True, help='Output figure path')
    args = parser.parse_args()

    # 配置
    cfg = PLOT_CONFIG
    model = args.model
    setting_a_pass1 = args.setting_a_pass1 or SETTING_A_PASS1.get(model)

    if setting_a_pass1 is None:
        print(f"WARNING: No Setting A Pass@1 data for {model}, upper bound line will not be drawn")

    # 数据路径
    root = PROJECT_ROOT

    # 加载数据
    samples_bc = []

    # Setting B - 根据是否合并决定加载策略
    if args.merge_prompts:
        # 合并 Guidance 和 Few-shot
        for prompt_type in ['guidance_prompt', 'fewshot_prompt']:
            setting_b_candidates = [
                root / 'setting_b_results' / prompt_type / model / 'evaluated_results.jsonl',
                root / 'setting_b_results' / prompt_type / model / 're_evaluated_results.jsonl',
            ]
            for candidate in setting_b_candidates:
                if candidate.exists():
                    print(f"Loading Setting B ({prompt_type}) from: {candidate}")
                    samples_bc.extend(load_sample_data(candidate))
                    break
            else:
                print(f"WARNING: Setting B ({prompt_type}) file not found for {model}")
    else:
        # 只加载指定的 prompt type
        prompt_dir = 'guidance_prompt' if args.setting_b_prompt.lower() == 'guidance' else 'fewshot_prompt'
        setting_b_candidates = [
            root / 'setting_b_results' / prompt_dir / model / 'evaluated_results.jsonl',
            root / 'setting_b_results' / prompt_dir / model / 're_evaluated_results.jsonl',
        ]
        setting_b_path = None
        for candidate in setting_b_candidates:
            if candidate.exists():
                setting_b_path = candidate
                break

        if setting_b_path and setting_b_path.exists():
            print(f"Loading Setting B from: {setting_b_path}")
            samples_bc.extend(load_sample_data(setting_b_path))
        else:
            print(f"WARNING: Setting B file not found: {setting_b_path}")

    # Setting C（需要找到对应的模型目录名）
    # 可能是 model_round1, model_round2 等，这里简化处理
    setting_c_candidates = [
        root / 'setting_c_results' / model / 're_evaluated_results.jsonl',
        root / 'setting_c_results' / f'{model}_round1' / 're_evaluated_results.jsonl',
        root / 'setting_c_results' / f'{model}_round2' / 're_evaluated_results.jsonl',
    ]

    setting_c_path = None
    for candidate in setting_c_candidates:
        if candidate.exists():
            setting_c_path = candidate
            break

    if setting_c_path:
        print(f"Loading Setting C from: {setting_c_path}")
        samples_bc.extend(load_sample_data(setting_c_path))
    else:
        print(f"WARNING: Setting C file not found for {model}")

    if not samples_bc:
        raise SystemExit(f"No valid samples found for {model}")

    print(f"\nTotal samples (B+C): {len(samples_bc)}")

    # 分桶
    if args.binning_method == 'quantile':
        bin_fcr_means, bin_pass1_means, bin_counts = bin_data_quantile(
            samples_bc, n_bins=args.n_bins
        )
    else:
        bin_fcr_means, bin_pass1_means, bin_counts = bin_data_fixed(
            samples_bc, bin_width=args.bin_width
        )

    print(f"\nBinning results ({args.binning_method}):")
    for fcr_mean, pass1_mean, count in zip(bin_fcr_means, bin_pass1_means, bin_counts):
        warn = " ⚠️" if count < cfg['min_samples_per_bin'] else ""
        print(f"  FCR={fcr_mean:.2f}: Pass@1={pass1_mean:.3f}, n={count}{warn}")

    # 绘图
    fig, ax = plt.subplots(figsize=cfg['figsize'])

    # 计算阴影带宽度（与样本数成正比）
    max_count = max(bin_counts)
    min_count = min(bin_counts)
    max_band_width = 0.08  # 最大样本数对应的阴影带半宽度
    min_band_width = 0.02  # 最小样本数对应的阴影带半宽度

    # 归一化样本数到阴影带宽度
    if max_count > min_count:
        band_widths = [min_band_width + (count - min_count) / (max_count - min_count) * (max_band_width - min_band_width)
                       for count in bin_counts]
    else:
        band_widths = [min_band_width] * len(bin_counts)

    # 计算阴影带上下界
    upper_bounds = [y + w for y, w in zip(bin_pass1_means, band_widths)]
    lower_bounds = [y - w for y, w in zip(bin_pass1_means, band_widths)]

    # 绘制阴影带
    ax.fill_between(bin_fcr_means, lower_bounds, upper_bounds,
                   alpha=0.2,
                   color=cfg['bc_curve_color'],
                   linewidth=0,
                   zorder=1)

    # 绘制主折线
    ax.plot(bin_fcr_means, bin_pass1_means,
            color=cfg['bc_curve_color'],
            linewidth=cfg['bc_curve_linewidth'],
            marker='o', markersize=6,
            zorder=3)

    # 添加Setting B/C的模型级别散点（不包括Setting A）
    # 散点颜色与折线一致，透明度70%
    # 优先使用命令行参数，其次使用配置文件
    scatter_color = cfg['bc_curve_color']  # 与折线颜色一致

    # Setting B Guidance: 正方形
    guidance_fcr = args.setting_b_guidance_fcr
    guidance_pass1 = args.setting_b_guidance_pass1
    if guidance_fcr is None or guidance_pass1 is None:
        if model in SETTING_POINTS and 'setting_b_guidance' in SETTING_POINTS[model]:
            guidance_fcr, guidance_pass1 = SETTING_POINTS[model]['setting_b_guidance']

    if guidance_fcr is not None and guidance_pass1 is not None:
        ax.scatter([guidance_fcr], [guidance_pass1/100], s=150, marker='s',
                  color=scatter_color, alpha=0.7, edgecolors='black', linewidths=1.5,
                  zorder=5)

    # Setting B Few-shot: 三角形
    fewshot_fcr = args.setting_b_fewshot_fcr
    fewshot_pass1 = args.setting_b_fewshot_pass1
    if fewshot_fcr is None or fewshot_pass1 is None:
        if model in SETTING_POINTS and 'setting_b_fewshot' in SETTING_POINTS[model]:
            fewshot_fcr, fewshot_pass1 = SETTING_POINTS[model]['setting_b_fewshot']

    if fewshot_fcr is not None and fewshot_pass1 is not None:
        ax.scatter([fewshot_fcr], [fewshot_pass1/100], s=150, marker='^',
                  color=scatter_color, alpha=0.7, edgecolors='black', linewidths=1.5,
                  zorder=5)

    # Setting C: 菱形
    setting_c_fcr = args.setting_c_fcr
    setting_c_pass1 = args.setting_c_pass1
    if setting_c_fcr is None or setting_c_pass1 is None:
        if model in SETTING_POINTS and 'setting_c' in SETTING_POINTS[model]:
            setting_c_fcr, setting_c_pass1 = SETTING_POINTS[model]['setting_c']

    if setting_c_fcr is not None and setting_c_pass1 is not None:
        ax.scatter([setting_c_fcr], [setting_c_pass1/100], s=150, marker='D',
                  color=scatter_color, alpha=0.7, edgecolors='black', linewidths=1.5,
                  zorder=5)

    # Setting A 上界虚线
    if setting_a_pass1 is not None:
        ax.axhline(y=setting_a_pass1/100,
                   linestyle=cfg['settingA_line_style'],
                   color=cfg['settingA_line_color'],
                   linewidth=cfg['settingA_line_width'],
                   alpha=cfg['settingA_line_alpha'],
                   zorder=2)

    # 坐标轴
    ax.set_xlabel('', fontsize=cfg['label_fontsize'])
    ax.set_ylabel('', fontsize=cfg['label_fontsize'])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', which='major', labelsize=cfg['tick_fontsize'])
    ax.xaxis.set_major_locator(MultipleLocator(cfg['x_major_locator']))
    ax.yaxis.set_major_locator(MultipleLocator(cfg['y_major_locator']))
    ax.grid(True, alpha=cfg['grid_alpha'])
    ax.set_title('')

    # 不添加图例（用户会在PPT中手工标注）

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
