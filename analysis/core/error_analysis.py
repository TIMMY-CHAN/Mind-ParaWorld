#!/usr/bin/env python3
"""
analyze_inference_errors.py - 分析推理过程中的各类错误

统计以下指标：
1. Agent tool_call 格式错误次数
2. World Model JSON 解析失败次数
3. World Model API 调用失败次数
4. 各类错误对最终成功率的影响
"""

import json
from collections import defaultdict, Counter
from typing import Dict, Any, List


def analyze_errors(input_file: str) -> Dict[str, Any]:
    """分析推理结果中的错误"""

    stats = {
        "total_samples": 0,
        "successful_samples": 0,
        "failed_samples": 0,

        # Agent 相关
        "total_tool_calls": 0,
        "valid_tool_calls": 0,
        "agent_format_errors": 0,

        # World Model 相关
        "world_model_json_parse_errors": 0,
        "world_model_api_errors": 0,

        # 恢复能力
        "samples_with_format_errors": 0,
        "samples_recovered_after_format_error": 0,

        # 详细统计
        "format_error_samples": [],
        "failed_samples_list": [],
    }

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                result = json.loads(line)
                sample_idx = result.get('index', stats['total_samples'])
                messages = result.get('messages', [])

                stats['total_samples'] += 1

                # 分析这个样本
                sample_stats = analyze_sample_messages(messages)

                # 汇总
                stats['total_tool_calls'] += sample_stats['total_tool_calls']
                stats['valid_tool_calls'] += sample_stats['valid_tool_calls']
                stats['agent_format_errors'] += sample_stats['agent_format_errors']
                stats['world_model_json_parse_errors'] += sample_stats['world_model_json_parse_errors']
                stats['world_model_api_errors'] += sample_stats['world_model_api_errors']

                # 检查是否有最终答案
                has_answer = sample_stats['has_answer']
                if has_answer:
                    stats['successful_samples'] += 1
                else:
                    stats['failed_samples'] += 1
                    stats['failed_samples_list'].append(sample_idx)

                # 检查格式错误恢复情况
                if sample_stats['agent_format_errors'] > 0:
                    stats['samples_with_format_errors'] += 1
                    stats['format_error_samples'].append({
                        'index': sample_idx,
                        'format_errors': sample_stats['agent_format_errors'],
                        'recovered': has_answer
                    })

                    if has_answer:
                        stats['samples_recovered_after_format_error'] += 1

            except Exception as e:
                print(f"[WARN] 解析样本时出错: {e}")
                continue

    return stats


def analyze_sample_messages(messages: List[Dict]) -> Dict[str, Any]:
    """分析单个样本的消息序列"""

    sample_stats = {
        'total_tool_calls': 0,
        'valid_tool_calls': 0,
        'agent_format_errors': 0,
        'world_model_json_parse_errors': 0,
        'world_model_api_errors': 0,
        'has_answer': False,
    }

    for msg in messages:
        role = msg.get('role')
        content = str(msg.get('content', ''))

        # 统计 Agent tool_call
        if role == 'assistant' and '<tool_call>' in content:
            sample_stats['total_tool_calls'] += 1

        # 检查格式错误
        if role == 'user' and 'Invalid tool call format' in content:
            sample_stats['agent_format_errors'] += 1
            # tool_call 已经计数，但不是有效的
        elif role == 'user' and '<tool_response>' in content:
            # 有工具响应，说明前一个 tool_call 是有效的
            sample_stats['valid_tool_calls'] += 1

        # 检查 World Model 错误
        if 'JSON parse error' in content or 'JSON 解析失败' in content:
            sample_stats['world_model_json_parse_errors'] += 1

        if 'API Error' in content or 'Service Error' in content:
            sample_stats['world_model_api_errors'] += 1

        # 检查是否有最终答案
        if role == 'assistant' and '<answer>' in content:
            sample_stats['has_answer'] = True

    return sample_stats


def print_report(stats: Dict[str, Any]):
    """打印分析报告"""

    print("="*80)
    print("  推理错误分析报告")
    print("="*80)
    print()

    # 总体统计
    print("📊 总体统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  成功样本: {stats['successful_samples']} ({stats['successful_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  失败样本: {stats['failed_samples']} ({stats['failed_samples']/stats['total_samples']*100:.1f}%)")
    print()

    # Agent 工具调用统计
    print("🤖 Agent 工具调用统计:")
    print(f"  总工具调用次数: {stats['total_tool_calls']}")
    print(f"  有效工具调用: {stats['valid_tool_calls']}")
    print(f"  格式错误次数: {stats['agent_format_errors']}")
    if stats['total_tool_calls'] > 0:
        print(f"  格式错误率: {stats['agent_format_errors']/stats['total_tool_calls']*100:.2f}%")
    print()

    # World Model 统计
    print("🌍 World Model 统计:")
    print(f"  JSON 解析失败: {stats['world_model_json_parse_errors']} 次")
    print(f"  API 调用失败: {stats['world_model_api_errors']} 次")
    print()

    # 恢复能力统计
    print("🔄 错误恢复能力:")
    print(f"  有过格式错误的样本: {stats['samples_with_format_errors']}")
    print(f"  格式错误后恢复的: {stats['samples_recovered_after_format_error']}")
    if stats['samples_with_format_errors'] > 0:
        recovery_rate = stats['samples_recovered_after_format_error'] / stats['samples_with_format_errors']
        print(f"  恢复成功率: {recovery_rate*100:.1f}%")
    print()

    # 失败样本列表
    if stats['failed_samples'] > 0:
        print("❌ 失败样本索引:")
        print(f"  {stats['failed_samples_list'][:20]}")
        if len(stats['failed_samples_list']) > 20:
            print(f"  ... (共 {len(stats['failed_samples_list'])} 个)")
        print()

    # 格式错误详情（抽样）
    if stats['format_error_samples']:
        print("📋 格式错误样本详情（前 10 个）:")
        for sample in stats['format_error_samples'][:10]:
            status = "✅ 已恢复" if sample['recovered'] else "❌ 失败"
            print(f"  样本 #{sample['index']}: {sample['format_errors']} 次格式错误 - {status}")
        print()

    print("="*80)
    print()

    # 关键结论
    print("💡 关键结论:")
    print()

    if stats['failed_samples'] == 0:
        print("  ✅ 所有样本都成功完成了推理！")
    else:
        print(f"  ⚠️  有 {stats['failed_samples']} 个样本未能完成")

    if stats['agent_format_errors'] > 0:
        print(f"  📊 Agent 格式错误率: {stats['agent_format_errors']/stats['total_tool_calls']*100:.2f}%")
        if stats['samples_recovered_after_format_error'] == stats['samples_with_format_errors']:
            print("  ✅ 所有格式错误都被 Agent 自我纠正了")
        else:
            print(f"  ⚠️  部分格式错误未能恢复")

    if stats['world_model_json_parse_errors'] > 0:
        print(f"  ⚠️  World Model JSON 解析失败 {stats['world_model_json_parse_errors']} 次")

    if stats['world_model_api_errors'] > 0:
        print(f"  ⚠️  World Model API 调用失败 {stats['world_model_api_errors']} 次")

    print()


def save_detailed_report(stats: Dict[str, Any], output_file: str):
    """保存详细报告为 JSON"""

    # 移除列表（太长）
    report = {k: v for k, v in stats.items() if not isinstance(v, list)}
    report['format_error_sample_count'] = len(stats['format_error_samples'])
    report['failed_sample_count'] = len(stats['failed_samples_list'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📁 详细报告已保存: {output_file}")


if __name__ == "__main__":
    import sys

    input_file = "results_full/inference_results_full.jsonl"
    output_file = "results_full/error_analysis.json"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print(f"📋 分析推理结果: {input_file}\n")

    stats = analyze_errors(input_file)
    print_report(stats)
    save_detailed_report(stats, output_file)
