#!/usr/bin/env python3
"""打印各模型 n(k) 表，辅助选择每个模型的 k_max / K。

n(k) 含义
- n(k) = 该模型中“有效工具调用次数 >= k”的样本数。
- 这里的有效工具调用次数来自 analysis/fcr_vs_toolcalls_analysis.py 的 per_sample_series 序列长度。

用法
python3 analysis/scripts/report_nk_for_fcr_toolcalls.py \
  --input /.../fcr_vs_toolcalls_data.json \
  --max-k 32

输出
- 打印一组关键 k（可在代码里改 ks_show），以及建议的 k_max（按 n>=200/100/50）。
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--max-k', type=int, default=32)
    args = p.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ks_show = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, args.max_k]

    print('n(k) report (effective_calls >= k)')
    print('json:', args.input)
    print('models:', list(data.keys()))

    for model in data.keys():
        series = data[model]['model_data']['per_sample_series']
        ns = []
        for k in range(1, args.max_k + 1):
            n = sum(1 for s in series if len(s.get('new_facts_by_call', [])) >= k)
            ns.append(n)

        print(f'\n== {model} ==')
        print('total series:', len(series))
        for k in ks_show:
            if 1 <= k <= args.max_k:
                print(f'  k={k:2d}: n(k)={ns[k-1]}')

        def max_k_with(th):
            ok = [k for k in range(1, args.max_k + 1) if ns[k - 1] >= th]
            return max(ok) if ok else None

        print('  suggest k_max with n>=200:', max_k_with(200), ' n>=100:', max_k_with(100), ' n>=50:', max_k_with(50))


if __name__ == '__main__':
    main()
