#!/usr/bin/env python3
"""
calculate_metrics.py - 离线指标计算脚本

从推理结果文件计算所有指标（FCR、hit_precision、成功率等）

使用方法：
    python calculate_metrics.py results_full/inference_results_full.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def calculate_fcr(hit_logs: List[Dict], world_truth_info: Dict) -> float:
    """计算事实覆盖率（FCR）"""
    if not hit_logs:
        return 0.0

    # 收集匹配的事实键
    unique_fact_keys = set()
    for log in hit_logs:
        keys = log.get('matched_fact_keys', [])
        unique_fact_keys.update([k.strip() for k in keys])

    # 获取所有原子事实
    atomic_facts = world_truth_info.get("atomic_facts", {})
    if isinstance(atomic_facts, str):
        try:
            atomic_facts = json.loads(atomic_facts)
        except:
            atomic_facts = {}

    atomic_facts_keys = set([k.strip() for k in atomic_facts.keys()])
    matched_facts = unique_fact_keys & atomic_facts_keys
    total_facts = len(atomic_facts_keys) if atomic_facts_keys else 1

    return len(matched_facts) / total_facts if total_facts > 0 else 0.0


def calculate_metrics_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """计算单个样本的所有指标"""
    trajectory_log = sample.get("trajectory_log", {})

    # 工具调用统计
    all_tool_calls_list = trajectory_log.get("tool_calls", [])
    hit_logs = trajectory_log.get("hit_logs", [])

    all_tool_calls = len(all_tool_calls_list)
    valid_tool_calls = len(hit_logs)
    format_error_calls = all_tool_calls - valid_tool_calls

    # 命中统计
    total_hits = sum([log.get('hit', 0) for log in hit_logs])

    # FCR
    extra_info = sample.get("data", {}).get("extra_info", {})
    world_truth_info = extra_info.get("world_truth_info", {})
    fcr = calculate_fcr(hit_logs, world_truth_info)

    # 命中精度
    hit_precision = total_hits / valid_tool_calls if valid_tool_calls > 0 else 0.0

    # 事实覆盖详情
    unique_fact_keys = set()
    for log in hit_logs:
        keys = log.get('matched_fact_keys', [])
        unique_fact_keys.update([k.strip() for k in keys])

    atomic_facts = world_truth_info.get("atomic_facts", {})
    if isinstance(atomic_facts, str):
        try:
            atomic_facts = json.loads(atomic_facts)
        except:
            atomic_facts = {}

    atomic_facts_keys = set([k.strip() for k in atomic_facts.keys()])
    matched_facts = unique_fact_keys & atomic_facts_keys

    return {
        "all_tool_calls": all_tool_calls,
        "valid_tool_calls": valid_tool_calls,
        "format_error_calls": format_error_calls,
        "total_hits": total_hits,
        "unique_facts_covered": len(matched_facts),
        "total_facts": len(atomic_facts_keys) if atomic_facts_keys else 1,
        "fcr": fcr,
        "hit_precision": hit_precision,
    }


def calculate_overall_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算整体统计"""
    total_samples = len(samples)

    # 状态统计
    status_counts = defaultdict(int)
    samples_with_answer = 0
    max_turns_reached_with_answer = 0

    for sample in samples:
        status = sample.get("status", "unknown")
        status_counts[status] += 1

        # 检查是否有答案
        has_answer = (
            sample.get("prediction") or  # 已提取的答案
            (
                '</answer>' in sample.get("full_response", "") and
                '<answer>' in sample.get("full_response", "")
            )  # 完整的答案标签
        )

        if has_answer:
            samples_with_answer += 1
            if status == "max_turns_reached":
                max_turns_reached_with_answer += 1

    # 实际成功率（基于答案）
    actual_success_rate = samples_with_answer / total_samples if total_samples > 0 else 0.0

    # 轮次统计
    turn_distribution = defaultdict(int)
    for sample in samples:
        turns = sample.get("actual_turns", 0)
        turn_distribution[turns] += 1

    # 工具调用统计
    total_all_calls = sum(s.get("metrics", {}).get("all_tool_calls", 0) for s in samples)
    total_valid_calls = sum(s.get("metrics", {}).get("valid_tool_calls", 0) for s in samples)
    total_format_errors = sum(s.get("metrics", {}).get("format_error_calls", 0) for s in samples)
    total_hits = sum(s.get("metrics", {}).get("total_hits", 0) for s in samples)

    avg_fcr = sum(s.get("metrics", {}).get("fcr", 0) for s in samples) / total_samples if total_samples > 0 else 0.0
    avg_hit_precision = (
        sum(s.get("metrics", {}).get("hit_precision", 0) for s in samples) / total_samples
        if total_samples > 0 else 0.0
    )

    return {
        "overall": {
            "total_samples": total_samples,
            "samples_with_answer": samples_with_answer,
            "actual_success_rate": actual_success_rate,
            "finished": status_counts.get("finished", 0),
            "max_turns_reached": status_counts.get("max_turns_reached", 0),
            "max_turns_reached_with_answer": max_turns_reached_with_answer,
            "api_error": status_counts.get("api_error", 0),
        },
        "tool_calls": {
            "total_all_calls": total_all_calls,
            "total_valid_calls": total_valid_calls,
            "total_format_errors": total_format_errors,
            "avg_calls_per_sample": total_all_calls / total_samples if total_samples > 0 else 0.0,
            "format_error_rate": total_format_errors / total_all_calls if total_all_calls > 0 else 0.0,
            "total_hits": total_hits,
        },
        "quality_metrics": {
            "avg_fcr": avg_fcr,
            "avg_hit_precision": avg_hit_precision,
        },
        "turn_distribution": dict(sorted(turn_distribution.items())),
    }


def calculate_category_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按类别统计"""
    category_data = defaultdict(list)

    for sample in samples:
        category = sample.get("category", "unknown")
        category_data[category].append(sample)

    category_stats = {}

    for category, category_samples in category_data.items():
        count = len(category_samples)

        # 成功统计
        finished = sum(1 for s in category_samples if s.get("status") == "finished")
        max_turns_reached = sum(1 for s in category_samples if s.get("status") == "max_turns_reached")
        success_rate = finished / count if count > 0 else 0.0

        # 工具调用统计
        total_tool_calls = sum(s.get("metrics", {}).get("all_tool_calls", 0) for s in category_samples)
        total_hits = sum(s.get("metrics", {}).get("total_hits", 0) for s in category_samples)
        total_valid_calls = sum(s.get("metrics", {}).get("valid_tool_calls", 0) for s in category_samples)
        avg_tool_calls = total_tool_calls / count if count > 0 else 0.0

        # FCR 和 hit_precision
        avg_fcr = sum(s.get("metrics", {}).get("fcr", 0) for s in category_samples) / count if count > 0 else 0.0
        hit_precision = total_hits / total_valid_calls if total_valid_calls > 0 else 0.0

        category_stats[category] = {
            "count": count,
            "success": finished,
            "max_turns_reached": max_turns_reached,
            "success_rate": success_rate,
            "avg_tool_calls": avg_tool_calls,
            "total_hits": total_hits,
            "hit_precision": hit_precision,
            "avg_fcr": avg_fcr,
        }

    return category_stats


def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from inference results")
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file (with metrics added)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    # 读取所有样本
    print(f"Reading {input_path}...")
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    # 计算每个样本的指标
    print("Calculating metrics for each sample...")
    for sample in samples:
        metrics = calculate_metrics_for_sample(sample)
        sample["metrics"] = metrics

    # 计算整体统计
    print("Calculating overall statistics...")
    overall_stats = calculate_overall_stats(samples)

    # 计算分类统计
    print("Calculating category statistics...")
    category_stats = calculate_category_stats(samples)

    # 输出结果
    stats = {
        "overall": overall_stats,
        "category_stats": category_stats,
    }

    # 保存带指标的样本
    if args.output:
        output_path = Path(args.output)
        print(f"Writing results with metrics to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 保存统计信息
    stats_path = input_path.parent / (input_path.stem + "_stats.json")
    print(f"Writing statistics to {stats_path}...")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(f"\nOverall:")
    print(f"  Total samples: {overall_stats['overall']['total_samples']}")
    print(f"  Samples with answer: {overall_stats['overall']['samples_with_answer']}")
    print(f"  Actual success rate: {overall_stats['overall']['actual_success_rate']:.2%}")
    print(f"  Finished: {overall_stats['overall']['finished']}")
    print(f"  Max turns reached: {overall_stats['overall']['max_turns_reached']}")
    print(f"    (with answer): {overall_stats['overall']['max_turns_reached_with_answer']}")

    print(f"\nTool Calls:")
    print(f"  Total all calls: {overall_stats['tool_calls']['total_all_calls']}")
    print(f"  Total valid calls: {overall_stats['tool_calls']['total_valid_calls']}")
    print(f"  Total format errors: {overall_stats['tool_calls']['total_format_errors']}")
    print(f"  Avg calls per sample: {overall_stats['tool_calls']['avg_calls_per_sample']:.2f}")
    print(f"  Format error rate: {overall_stats['tool_calls']['format_error_rate']:.2%}")
    print(f"  Total hits: {overall_stats['tool_calls']['total_hits']}")

    print(f"\nQuality Metrics:")
    print(f"  Avg FCR: {overall_stats['quality_metrics']['avg_fcr']:.4f}")
    print(f"  Avg hit precision: {overall_stats['quality_metrics']['avg_hit_precision']:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
