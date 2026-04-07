#!/usr/bin/env python3
"""【Setting C】FCR vs 工具调用次数（简单汇总图）

统计口径
- 输入数据来自 analysis/fcr_vs_toolcalls_analysis.py 生成的 fcr_vs_toolcalls_data.json。
- x 轴为“有效工具调用次数”（message-constrained effective calls），并按 interval*2 映射：
  interval=1 表示 2 次调用、interval=2 表示 4 次调用，依此类推。

图像含义
- 每条曲线：某个模型在该调用次数刻度下的平均累计 FCR（fcr_curve.avg_fcr）。
- 阴影带：±1 std（fcr_curve.std_fcr），用于展示跨样本波动。

注意
- 该图是“非固定 cohort”的：随着调用次数增大，参与平均的样本集合会变小/变难，
  因此平均曲线不一定单调上升；若要讨论饱和/边际收益，更推荐使用 cohort 固定图。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / 'fcr_vs_toolcalls_data.json'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'analysis_results' / 'fcr_vs_toolcalls.png'


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _extract_curve(model_payload: dict):
    curve = model_payload.get('fcr_curve', []) or []
    if not curve:
        return [], [], []

    xs = [int(p['interval']) * 2 for p in curve]  # interval=1 means 2 calls
    ys = [float(p['avg_fcr']) for p in curve]
    yerr = [float(p.get('std_fcr', 0.0)) for p in curve]
    return xs, ys, yerr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        default=str(DEFAULT_INPUT_PATH),
        help='Path to fcr_vs_toolcalls_data.json',
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT_PATH),
        help='Output image path (.png)',
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = _load(in_path)

    plt.figure(figsize=(10, 6))

    for model_name, payload in data.items():
        xs, ys, yerr = _extract_curve(payload)
        if not xs:
            continue
        plt.plot(xs, ys, marker='o', linewidth=2, markersize=4, label=model_name)
        lower = [max(0.0, y - e) for y, e in zip(ys, yerr)]
        upper = [min(1.0, y + e) for y, e in zip(ys, yerr)]
        plt.fill_between(xs, lower, upper, alpha=0.15)

    plt.xlabel('Tool calls (effective, message-constrained)')
    plt.ylabel('Cumulative FCR')
    plt.title('FCR vs Tool Calls (2 calls per tick)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
