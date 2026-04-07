#!/usr/bin/env python3
"""
comprehensive_analysis.py - 完整的推理结果分析

包含:
1. 整体统计
2. 按类别分析
3. 错误分析
4. 工具调用统计
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Any


def main(input_file: str):
    """完整分析推理结果"""

    print(f"📋 分析推理结果: {input_file}\n")

    # 初始化统计
    stats = {
        'total_samples': 0,
        'success': 0,
        'failed': 0,
        'category_data': defaultdict(lambda: {
            'count': 0,
            'success': 0,
            'total_tool_calls': 0,
            'total_hits': 0,
            'fcr_values': [],
        }),
        'tool_call_stats': {
            'total': 0,
            'successful': 0,
            'format_errors': 0,
        },
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            stats['total_samples'] += 1

            # 基础统计
            status = result.get('status', 'unknown')
            if status == 'finished':
                stats['success'] += 1
            else:
                stats['failed'] += 1

            # 类别统计
            category = result.get('category', 'unknown')
            metrics = result.get('metrics', {})

            cat_data = stats['category_data'][category]
            cat_data['count'] += 1
            if status == 'finished':
                cat_data['success'] += 1
            cat_data['total_tool_calls'] += metrics.get('total_tool_calls', 0)
            cat_data['total_hits'] += metrics.get('total_hits', 0)

            fcr = metrics.get('fcr', 0)
            if fcr > 0:
                cat_data['fcr_values'].append(fcr)

    # 打印整体统计
    print("="*80)
    print("  整体统计")
    print("="*80)
    print()
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功: {stats['success']} ({stats['success']/stats['total_samples']*100:.1f}%)")
    print(f"失败: {stats['failed']} ({stats['failed']/stats['total_samples']*100:.1f}%)")
    print()

    # 打印类别统计（按 FCR 排序）
    print("="*80)
    print("  按类别性能分析")
    print("="*80)
    print()

    category_summary = []
    for category, data in stats['category_data'].items():
        if data['count'] > 0:
            summary = {
                'category': category,
                'count': data['count'],
                'success_rate': data['success'] / data['count'] * 100,
                'avg_tool_calls': data['total_tool_calls'] / data['count'],
                'hit_precision': data['total_hits'] / data['total_tool_calls'] if data['total_tool_calls'] > 0 else 0,
                'avg_fcr': sum(data['fcr_values']) / len(data['fcr_values']) if data['fcr_values'] else 0,
            }
            category_summary.append(summary)

    # 按 FCR 排序
    category_summary.sort(key=lambda x: x['avg_fcr'], reverse=True)

    # 打印
    print(f"{'类别':<35s} {'数量':>6s} {'成功率':>8s} {'平均调用':>8s} {'命中率':>8s} {'FCR':>8s}")
    print("-"*80)

    for summary in category_summary:
        print(f"{summary['category']:<35s} "
              f"{summary['count']:>6d} "
              f"{summary['success_rate']:>7.1f}% "
              f"{summary['avg_tool_calls']:>7.1f} "
              f"{summary['hit_precision']:>7.1%} "
              f"{summary['avg_fcr']:>7.1%}")

    print()

    # 性能最好和最差的类别
    best = max(category_summary, key=lambda x: x['avg_fcr'])
    worst = min(category_summary, key=lambda x: x['avg_fcr'])

    print("🏆 性能最好的类别:")
    print(f"  {best['category']}")
    print(f"  FCR: {best['avg_fcr']:.1%}")
    print(f"  命中率: {best['hit_precision']:.1%}")
    print()

    print("⚠️  性能最差的类别:")
    print(f"  {worst['category']}")
    print(f"  FCR: {worst['avg_fcr']:.1%}")
    print(f"  命中率: {worst['hit_precision']:.1%}")
    print()

    # 保存统计结果
    output_file = input_file.replace('.jsonl', '_stats.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'overall': {
                'total_samples': stats['total_samples'],
                'success': stats['success'],
                'failed': stats['failed'],
                'success_rate': stats['success'] / stats['total_samples'] * 100,
            },
            'category_stats': {cat: {
                'count': data['count'],
                'success': data['success'],
                'success_rate': data['success'] / data['count'] * 100,
                'avg_tool_calls': data['total_tool_calls'] / data['count'],
                'total_hits': data['total_hits'],
                'hit_precision': data['total_hits'] / data['total_tool_calls'] if data['total_tool_calls'] > 0 else 0,
                'avg_fcr': sum(data['fcr_values']) / len(data['fcr_values']) if data['fcr_values'] else 0,
            } for cat, data in stats['category_data'].items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"📁 详细统计已保存: {output_file}")


if __name__ == "__main__":
    import sys

    input_file = "results_full/inference_results_full_with_category.jsonl"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    main(input_file)
