#!/usr/bin/env python3
"""
analyze_judge_results.py - 分析Setting A的LLM-as-Judge评估结果

生成详细的统计报告，包括：
- 整体准确率
- 分类别准确率
- 错误样本分析
"""

import json
import argparse
from collections import defaultdict


def load_results(file_path: str):
    """加载评估结果"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def analyze_results(results):
    """分析评估结果"""
    total = len(results)
    correct = sum(1 for r in results if r.get('verdict') == 'Correct')
    incorrect = sum(1 for r in results if r.get('verdict') == 'Incorrect')
    accuracy = correct / total * 100 if total > 0 else 0

    # 按category统计
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        cat = r.get('category', 'unknown')
        category_stats[cat]['total'] += 1
        if r.get('verdict') == 'Correct':
            category_stats[cat]['correct'] += 1

    # 计算每个category的准确率
    for cat, stats in category_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0

    # 找出错误样本
    incorrect_samples = [r for r in results if r.get('verdict') == 'Incorrect']

    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'accuracy': accuracy,
        'category_stats': category_stats,
        'incorrect_samples': incorrect_samples
    }


def print_report(analysis):
    """打印分析报告"""
    print("=" * 80)
    print("  Setting A: Oracle-Facts QA - 评估报告")
    print("=" * 80)
    print()

    # 整体统计
    print("📊 整体统计:")
    print(f"  总样本数: {analysis['total']}")
    print(f"  Correct: {analysis['correct']} ({analysis['correct']/analysis['total']*100:.2f}%)")
    print(f"  Incorrect: {analysis['incorrect']} ({analysis['incorrect']/analysis['total']*100:.2f}%)")
    print(f"  Accuracy: {analysis['accuracy']:.2f}%")
    print()

    # 分类别统计
    print("📂 分类别准确率:")
    sorted_cats = sorted(
        analysis['category_stats'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    for cat, stats in sorted_cats:
        print(f"  {cat:30s} | {stats['correct']:4d}/{stats['total']:4d} | {stats['accuracy']:6.2f}%")
    print()

    # 错误样本统计
    print(f"❌ 错误样本数: {len(analysis['incorrect_samples'])}")
    if analysis['incorrect_samples']:
        print(f"\n前10个错误样本:")
        for i, sample in enumerate(analysis['incorrect_samples'][:10], 1):
            print(f"\n  [{i}] Index: {sample.get('index')}, Category: {sample.get('category')}")
            print(f"      Question: {sample.get('question', '')[:80]}...")
            print(f"      Ground Truth: {sample.get('ground_truth', '')[:80]}...")
            print(f"      Prediction: {sample.get('prediction', '')[:80]}...")
            print(f"      Reasoning: {sample.get('reasoning', '')[:100]}...")


def save_incorrect_samples(incorrect_samples, output_file):
    """保存错误样本到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in incorrect_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"\n💾 错误样本已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析Setting A的LLM-as-Judge评估结果")
    parser.add_argument('--input', required=True, help='输入的evaluated_results.jsonl文件')
    parser.add_argument('--save-errors', default=None, help='保存错误样本的输出文件（可选）')

    args = parser.parse_args()

    # 加载结果
    results = load_results(args.input)

    # 分析结果
    analysis = analyze_results(results)

    # 打印报告
    print_report(analysis)

    # 保存错误样本（如果指定）
    if args.save_errors and analysis['incorrect_samples']:
        save_incorrect_samples(analysis['incorrect_samples'], args.save_errors)


if __name__ == '__main__':
    main()
