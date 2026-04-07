#!/usr/bin/env python3
"""
generate_main_table.py - 生成主表格统计数据

功能：
- 从 evaluated_results.jsonl 读取数据
- 按难度分级（Easy/Medium/Hard）统计指标
- 生成主表格：Model | Difficulty | Pass@1 | FCR | Hit Rate | Avg Turns

难度定义：
- Easy: 1-5 atomic facts
- Medium: 6-10 atomic facts
- Hard: ≥11 atomic facts

使用方法：
    python generate_main_table.py \
        --inputs results/model1/evaluated_results.jsonl results/model2/evaluated_results.jsonl \
        --labels "Model1" "Model2" \
        --output main_table.md
"""

import json
import argparse
import os
from collections import defaultdict
import numpy as np


def assign_difficulty(atomic_facts_count):
    """根据原子事实数量分配难度等级"""
    if atomic_facts_count <= 5:
        return "Easy"
    elif atomic_facts_count <= 10:
        return "Medium"
    else:
        return "Hard"


def calculate_metrics_by_difficulty(result_file):
    """
    按难度分级统计指标

    Returns:
        dict: {
            'Easy': {'pass@1': x, 'fcr': x, 'hit_rate': x, 'avg_turns': x, 'count': n},
            'Medium': {...},
            'Hard': {...},
            'Overall': {...}
        }
    """
    difficulty_data = {
        'Easy': {'correct': 0, 'partial': 0, 'fcr_values': [], 'hit_rates': [], 'turns': [], 'count': 0},
        'Medium': {'correct': 0, 'partial': 0, 'fcr_values': [], 'hit_rates': [], 'turns': [], 'count': 0},
        'Hard': {'correct': 0, 'partial': 0, 'fcr_values': [], 'hit_rates': [], 'turns': [], 'count': 0},
    }

    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except:
                continue

            # 获取原子事实数量
            traj = data.get('trajectory_log', {})
            world_truth = traj.get('world_truth_info', {})
            atomic_facts = world_truth.get('atomic_facts', [])
            facts_count = len(atomic_facts)

            # 分配难度
            difficulty = assign_difficulty(facts_count)
            diff_data = difficulty_data[difficulty]
            diff_data['count'] += 1

            # Pass@1
            evaluation = data.get('evaluation', {})
            judgment = evaluation.get('judgment', '')
            if judgment == 'Correct':
                diff_data['correct'] += 1
            elif judgment == 'Partial':
                diff_data['partial'] += 1

            # FCR
            metrics = data.get('metrics', {})
            fcr = metrics.get('fcr', None)
            if fcr is not None:
                diff_data['fcr_values'].append(fcr)

            # Hit Rate (使用 hit_precision)
            hit_precision = metrics.get('hit_precision', None)
            if hit_precision is not None:
                diff_data['hit_rates'].append(hit_precision * 100)

            # Avg Turns
            actual_turns = metrics.get('actual_turns', None)
            if actual_turns is not None:
                diff_data['turns'].append(actual_turns)

    # 计算统计量
    results = {}
    for difficulty in ['Easy', 'Medium', 'Hard']:
        data = difficulty_data[difficulty]
        count = data['count']

        if count > 0:
            results[difficulty] = {
                'pass@1': data['correct'] / count * 100,
                'partial': data['partial'] / count * 100,
                'fcr': np.mean(data['fcr_values']) if data['fcr_values'] else 0,
                'fcr_std': np.std(data['fcr_values']) if data['fcr_values'] else 0,
                'hit_rate': np.mean(data['hit_rates']) if data['hit_rates'] else 0,
                'hit_rate_std': np.std(data['hit_rates']) if data['hit_rates'] else 0,
                'avg_turns': np.mean(data['turns']) if data['turns'] else 0,
                'turns_std': np.std(data['turns']) if data['turns'] else 0,
                'count': count
            }
        else:
            results[difficulty] = {
                'pass@1': 0, 'partial': 0, 'fcr': 0, 'fcr_std': 0,
                'hit_rate': 0, 'hit_rate_std': 0, 'avg_turns': 0, 'turns_std': 0, 'count': 0
            }

    # 计算 Overall
    all_correct = sum(difficulty_data[d]['correct'] for d in ['Easy', 'Medium', 'Hard'])
    all_partial = sum(difficulty_data[d]['partial'] for d in ['Easy', 'Medium', 'Hard'])
    all_count = sum(difficulty_data[d]['count'] for d in ['Easy', 'Medium', 'Hard'])
    all_fcr = [v for d in ['Easy', 'Medium', 'Hard'] for v in difficulty_data[d]['fcr_values']]
    all_hit = [v for d in ['Easy', 'Medium', 'Hard'] for v in difficulty_data[d]['hit_rates']]
    all_turns = [v for d in ['Easy', 'Medium', 'Hard'] for v in difficulty_data[d]['turns']]

    results['Overall'] = {
        'pass@1': all_correct / all_count * 100 if all_count > 0 else 0,
        'partial': all_partial / all_count * 100 if all_count > 0 else 0,
        'fcr': np.mean(all_fcr) if all_fcr else 0,
        'fcr_std': np.std(all_fcr) if all_fcr else 0,
        'hit_rate': np.mean(all_hit) if all_hit else 0,
        'hit_rate_std': np.std(all_hit) if all_hit else 0,
        'avg_turns': np.mean(all_turns) if all_turns else 0,
        'turns_std': np.std(all_turns) if all_turns else 0,
        'count': all_count
    }

    return results


def generate_markdown_table(model_results, model_labels):
    """
    生成 Markdown 格式的主表格

    Args:
        model_results: {model_name: {difficulty: metrics}}
        model_labels: [model_name1, model_name2, ...]

    Returns:
        str: Markdown 表格
    """
    lines = []

    # 表头
    lines.append("# Main Results Table")
    lines.append("")
    lines.append("## Difficulty Definition")
    lines.append("- 🟢 **Easy**: 1-5 atomic facts (568 samples)")
    lines.append("- 🟡 **Medium**: 6-10 atomic facts (619 samples)")
    lines.append("- 🔴 **Hard**: ≥11 atomic facts (421 samples)")
    lines.append("")
    lines.append("## Results by Difficulty")
    lines.append("")

    # 表格表头
    lines.append("| Model | Difficulty | Pass@1 (%) | FCR | Hit Rate (%) | Avg Turns | Samples |")
    lines.append("|-------|-----------|-----------|-----|-------------|-----------|---------|")

    # 表格内容
    for model_name in model_labels:
        if model_name not in model_results:
            continue

        results = model_results[model_name]

        for idx, difficulty in enumerate(['Easy', 'Medium', 'Hard', 'Overall']):
            if difficulty not in results:
                continue

            metrics = results[difficulty]

            # 第一行显示模型名，后续行为空
            model_col = model_name if idx == 0 else ""

            # 难度标识
            if difficulty == 'Easy':
                diff_col = "🟢 Easy"
            elif difficulty == 'Medium':
                diff_col = "🟡 Medium"
            elif difficulty == 'Hard':
                diff_col = "🔴 Hard"
            else:
                diff_col = "**Overall**"

            # 格式化指标
            pass1 = f"{metrics['pass@1']:.2f}"
            fcr = f"{metrics['fcr']:.3f}"
            hit_rate = f"{metrics['hit_rate']:.2f}"
            avg_turns = f"{metrics['avg_turns']:.2f}"
            count = metrics['count']

            lines.append(f"| {model_col:20s} | {diff_col:12s} | {pass1:>10s} | {fcr:>8s} | {hit_rate:>12s} | {avg_turns:>10s} | {count:>7d} |")

        # 模型之间添加分隔线
        if model_name != model_labels[-1]:
            lines.append("|-------|-----------|-----------|-----|-------------|-----------|---------|")

    return "\n".join(lines)


def generate_csv_table(model_results, model_labels, output_file):
    """生成 CSV 格式的表格"""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(['Model', 'Difficulty', 'Pass@1 (%)', 'FCR', 'Hit Rate (%)', 'Avg Turns', 'Samples'])

        # 数据
        for model_name in model_labels:
            if model_name not in model_results:
                continue

            results = model_results[model_name]

            for difficulty in ['Easy', 'Medium', 'Hard', 'Overall']:
                if difficulty not in results:
                    continue

                metrics = results[difficulty]
                writer.writerow([
                    model_name,
                    difficulty,
                    f"{metrics['pass@1']:.2f}",
                    f"{metrics['fcr']:.3f}",
                    f"{metrics['hit_rate']:.2f}",
                    f"{metrics['avg_turns']:.2f}",
                    metrics['count']
                ])


def main():
    parser = argparse.ArgumentParser(description="生成主表格统计数据")
    parser.add_argument('--inputs', '-i', type=str, nargs='+', required=True,
                       help='输入 evaluated_results.jsonl 文件列表')
    parser.add_argument('--labels', '-l', type=str, nargs='+', required=True,
                       help='模型标签（与 inputs 一一对应）')
    parser.add_argument('--output', '-o', type=str, default='main_table.md',
                       help='输出文件（.md 或 .csv）')

    args = parser.parse_args()

    # 验证输入
    if len(args.inputs) != len(args.labels):
        print("❌ 错误: inputs 和 labels 数量不匹配")
        return

    print("=" * 80)
    print("生成主表格统计数据")
    print("=" * 80)
    print()

    # 处理每个模型
    model_results = {}
    for input_file, label in zip(args.inputs, args.labels):
        if not os.path.exists(input_file):
            print(f"⚠️  跳过: {label} (文件不存在: {input_file})")
            continue

        print(f"📊 处理: {label}")
        print(f"   文件: {input_file}")

        results = calculate_metrics_by_difficulty(input_file)
        model_results[label] = results

        # 打印简要统计
        print(f"   样本数: {results['Overall']['count']}")
        print(f"   Overall Pass@1: {results['Overall']['pass@1']:.2f}%")
        print()

    if not model_results:
        print("❌ 没有可处理的数据")
        return

    # 生成表格
    print(f"💾 生成表格: {args.output}")

    if args.output.endswith('.csv'):
        generate_csv_table(model_results, args.labels, args.output)
    else:
        markdown_table = generate_markdown_table(model_results, args.labels)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(markdown_table)

    print()
    print("=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print()
    print(f"📁 输出文件: {args.output}")
    print()

    # 预览表格（如果是 markdown）
    if not args.output.endswith('.csv'):
        print("📋 表格预览:")
        print()
        with open(args.output, 'r', encoding='utf-8') as f:
            print(f.read())


if __name__ == '__main__':
    main()
