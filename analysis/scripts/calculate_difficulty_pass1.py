#!/usr/bin/env python3
"""
计算按难度分级的 Pass@1

适用于 Setting A/B/C 的 evaluated_results.jsonl 文件
根据原子事实数量自动分级：
- Easy: 1-5 个事实
- Medium: 6-10 个事实
- Hard: 11+ 个事实
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


def get_num_facts(sample: dict) -> int:
    """从样本中提取原子事实数量"""
    # Setting A: metrics.total_facts
    if 'metrics' in sample and 'total_facts' in sample['metrics']:
        return sample['metrics']['total_facts']

    # Setting B: trajectory_log.world_truth_info.atomic_facts
    if 'trajectory_log' in sample:
        world_truth = sample['trajectory_log'].get('world_truth_info', {})
        atomic_facts = world_truth.get('atomic_facts', {})
        if isinstance(atomic_facts, dict):
            return len(atomic_facts)
        elif isinstance(atomic_facts, list):
            return len(atomic_facts)

    # Setting C: data.world_truth_info.atomic_facts (字典)
    if 'data' in sample:
        world_truth = sample['data'].get('world_truth_info', {})
        atomic_facts = world_truth.get('atomic_facts', {})
        if isinstance(atomic_facts, dict):
            return len(atomic_facts)
        elif isinstance(atomic_facts, list):
            return len(atomic_facts)

    # 尝试直接从 metrics
    if 'metrics' in sample:
        total_facts = sample['metrics'].get('total_facts', 0)
        if total_facts > 0:
            return total_facts

    return 0


def get_difficulty(num_facts: int) -> str:
    """根据事实数量判断难度"""
    if num_facts <= 5:
        return 'easy'
    elif num_facts <= 10:
        return 'medium'
    else:
        return 'hard'


def get_judgment(sample: dict) -> str:
    """提取判断结果（兼容多种格式）"""
    # Setting A/B: answer 字段
    if 'answer' in sample:
        return sample['answer']

    # Setting C: evaluation.judgment 字段
    if 'evaluation' in sample:
        return sample['evaluation'].get('judgment', 'Unknown')

    return 'Unknown'


def calculate_difficulty_pass1(file_path: str, verbose: bool = False):
    """计算按难度分级的 Pass@1"""

    # 统计数据
    difficulty_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'partial': 0,
        'no_answer': 0,
        'unknown': 0
    })

    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)

            # 获取事实数量和难度
            num_facts = get_num_facts(sample)
            difficulty = get_difficulty(num_facts)

            # 获取判断结果
            judgment = get_judgment(sample)

            # 统计
            stats = difficulty_stats[difficulty]
            stats['total'] += 1

            if judgment == 'Correct':
                stats['correct'] += 1
            elif judgment == 'Incorrect':
                stats['incorrect'] += 1
            elif judgment == 'Partial':
                stats['partial'] += 1
            elif judgment == 'No Answer':
                stats['no_answer'] += 1
            else:
                stats['unknown'] += 1

            if verbose and stats['total'] <= 3:
                print(f"Sample {sample.get('index', '?')}: {num_facts} facts → {difficulty} → {judgment}")

    # 打印结果
    model_name = Path(file_path).parent.name
    print("=" * 80)
    print(f"按难度分级的 Pass@1 统计 - {model_name}")
    print("=" * 80)
    print()

    print(f"{'难度':<10} | {'样本数':>7} | {'正确':>7} | {'错误':>7} | {'部分正确':>9} | {'无答案':>7} | {'Pass@1':>8}")
    print("-" * 80)

    total_samples = 0
    total_correct = 0

    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty not in difficulty_stats:
            continue

        stats = difficulty_stats[difficulty]
        total = stats['total']
        correct = stats['correct']
        incorrect = stats['incorrect']
        partial = stats['partial']
        no_answer = stats['no_answer']

        if total > 0:
            pass_at_1 = correct / total * 100
        else:
            pass_at_1 = 0.0

        print(f"{difficulty:<10} | {total:>7} | {correct:>7} | {incorrect:>7} | "
              f"{partial:>9} | {no_answer:>7} | {pass_at_1:>7.2f}%")

        total_samples += total
        total_correct += correct

    print("-" * 80)
    overall_pass_at_1 = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':<10} | {total_samples:>7} | {total_correct:>7} | "
          f"{'-':>7} | {'-':>9} | {'-':>7} | {overall_pass_at_1:>7.2f}%")
    print()

    return difficulty_stats


def main():
    parser = argparse.ArgumentParser(description='计算按难度分级的 Pass@1')
    parser.add_argument('input', help='evaluated_results.jsonl 文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    parser.add_argument('--output', '-o', help='输出 JSON 文件路径（可选）')

    args = parser.parse_args()

    # 计算统计
    stats = calculate_difficulty_pass1(args.input, verbose=args.verbose)

    # 保存到 JSON（可选）
    if args.output:
        output_data = {
            'model': Path(args.input).parent.name,
            'difficulty_stats': dict(stats)
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 统计结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
