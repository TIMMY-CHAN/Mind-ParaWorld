#!/usr/bin/env python3
"""
多维度定量分析工具 - MPW-bench (Setting B/C 适配版本)

适配改动:
1. 从 trajectory_log 提取数据
2. 使用 answer 字段mei
3. 从 trajectory_log.world_truth_info.atomic_facts 获取难度信息

分析维度:
1. 难度相关性分析 (Pass@1, FCR, Hit Rate vs Difficulty)
2. 类别性能分析 (按类别的详细指标)
3. 工具使用效率分析 (Tool calls, Hit rate, Redundancy)
4. 轮数效率分析 (Turns vs Performance)
5. FCR-Pass@1 相关性分析
6. 误差分析 (Variance, Std, Outliers)
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics


class QuantitativeAnalyzerSettingB:
    """定量分析器 - Setting B 适配版本"""

    def __init__(self, evaluated_files: List[str]):
        self.models_data = {}
        for file_path in evaluated_files:
            model_name = Path(file_path).parent.name
            self.models_data[model_name] = self.load_data(file_path)

    def load_data(self, file_path: str) -> List[Dict]:
        """加载评估数据"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def extract_metrics(self, sample: Dict) -> Dict:
        """
        从 trajectory_log 提取指标

        Setting B 数据结构:
        - trajectory_log.tool_calls: 所有工具调用 (list of dicts with 'tool_name', 'query', 'hit')
        - trajectory_log.hit_logs: 命中记录 (list of dicts with 'query', 'hit', 'matched_fact_keys')
        - trajectory_log.world_truth_info.atomic_facts: 原子事实
        """
        trajectory = sample.get('trajectory_log', {})
        world_truth = trajectory.get('world_truth_info', {})
        atomic_facts = world_truth.get('atomic_facts', {})

        tool_calls = trajectory.get('tool_calls', [])
        hit_logs = trajectory.get('hit_logs', [])

        # 计算工具调用指标
        all_tool_calls = len(tool_calls)

        # 统计有效工具调用 (假设所有记录的工具调用都是有效的)
        valid_tool_calls = all_tool_calls

        # 统计格式错误 (如果工具调用中有错误标记)
        format_error_calls = sum(1 for call in tool_calls if call.get('error', False))

        # 统计命中 (从 tool_calls 的 hit 字段或 hit_logs 的 hit 字段)
        total_hits = sum(1 for call in tool_calls if call.get('hit', 0) > 0)

        # 计算覆盖的唯一事实 (从 hit_logs 的 matched_fact_keys)
        unique_facts_set = set()
        for hit_log in hit_logs:
            matched_keys = hit_log.get('matched_fact_keys', [])
            if isinstance(matched_keys, list):
                unique_facts_set.update(matched_keys)

        unique_facts_covered = len(unique_facts_set)

        # 总事实数
        if isinstance(atomic_facts, dict):
            total_facts = len(atomic_facts)
        elif isinstance(atomic_facts, list):
            total_facts = len(atomic_facts)
        else:
            total_facts = 0

        # 计算 FCR
        fcr = unique_facts_covered / total_facts if total_facts > 0 else 0

        # 计算命中率
        hit_rate = total_hits / valid_tool_calls if valid_tool_calls > 0 else 0

        return {
            'all_tool_calls': all_tool_calls,
            'valid_tool_calls': valid_tool_calls,
            'format_error_calls': format_error_calls,
            'total_hits': total_hits,
            'unique_facts_covered': unique_facts_covered,
            'total_facts': total_facts,
            'fcr': fcr,
            'hit_rate': hit_rate
        }

    def get_difficulty(self, sample: Dict) -> str:
        """获取样本难度"""
        trajectory = sample.get('trajectory_log', {})
        world_truth = trajectory.get('world_truth_info', {})
        atomic_facts = world_truth.get('atomic_facts', {})

        if isinstance(atomic_facts, dict):
            num_facts = len(atomic_facts)
        elif isinstance(atomic_facts, list):
            num_facts = len(atomic_facts)
        else:
            num_facts = 0

        if num_facts <= 5:
            return 'easy'
        elif num_facts <= 10:
            return 'medium'
        else:
            return 'hard'

    def get_judgment(self, sample: Dict) -> str:
        """
        获取判断结果

        Setting B 使用 answer 字段而非 evaluation.judgment
        """
        return sample.get('answer', 'Unknown')

    def analyze_difficulty_correlation(self, model_name: str) -> Dict:
        """分析难度相关性"""
        samples = self.models_data[model_name]

        difficulty_stats = defaultdict(lambda: {
            'samples': [],
            'correct': 0,
            'fcr_values': [],
            'hit_rate_values': [],
            'tool_calls': [],
            'turns': []
        })

        for sample in samples:
            difficulty = self.get_difficulty(sample)
            stats = difficulty_stats[difficulty]

            stats['samples'].append(sample)

            # 判断是否正确
            judgment = self.get_judgment(sample)
            if judgment == 'Correct':
                stats['correct'] += 1

            # 提取指标
            metrics = self.extract_metrics(sample)
            stats['fcr_values'].append(metrics.get('fcr', 0))
            stats['tool_calls'].append(metrics.get('all_tool_calls', 0))
            stats['turns'].append(sample.get('actual_turns', 0))

            # 修复：只要有工具调用就统计hit_rate，不管是否命中
            if metrics.get('valid_tool_calls', 0) > 0:
                stats['hit_rate_values'].append(metrics['hit_rate'])

        # 计算汇总统计
        result = {}
        for difficulty in ['easy', 'medium', 'hard']:
            stats = difficulty_stats[difficulty]
            count = len(stats['samples'])

            if count == 0:
                continue

            result[difficulty] = {
                'count': count,
                'pass@1': stats['correct'] / count * 100,
                'avg_fcr': statistics.mean(stats['fcr_values']) if stats['fcr_values'] else 0,
                'std_fcr': statistics.stdev(stats['fcr_values']) if len(stats['fcr_values']) > 1 else 0,
                'avg_hit_rate': statistics.mean(stats['hit_rate_values']) if stats['hit_rate_values'] else 0,
                'std_hit_rate': statistics.stdev(stats['hit_rate_values']) if len(stats['hit_rate_values']) > 1 else 0,
                'avg_tool_calls': statistics.mean(stats['tool_calls']) if stats['tool_calls'] else 0,
                'std_tool_calls': statistics.stdev(stats['tool_calls']) if len(stats['tool_calls']) > 1 else 0,
                'avg_turns': statistics.mean(stats['turns']) if stats['turns'] else 0,
                'std_turns': statistics.stdev(stats['turns']) if len(stats['turns']) > 1 else 0
            }

        return result

    def analyze_category_performance(self, model_name: str, top_k: int = 20) -> Dict:
        """分析类别性能"""
        samples = self.models_data[model_name]

        category_stats = defaultdict(lambda: {
            'count': 0,
            'correct': 0,
            'fcr_values': [],
            'hit_rate_values': [],
            'tool_calls': [],
            'turns': []
        })

        for sample in samples:
            category = sample.get('category', 'unknown')
            stats = category_stats[category]

            stats['count'] += 1

            # 判断是否正确
            judgment = self.get_judgment(sample)
            if judgment == 'Correct':
                stats['correct'] += 1

            # 收集指标
            metrics = self.extract_metrics(sample)
            stats['fcr_values'].append(metrics.get('fcr', 0))
            stats['tool_calls'].append(metrics.get('all_tool_calls', 0))
            stats['turns'].append(sample.get('actual_turns', 0))

            # 修复：只要有工具调用就统计hit_rate
            if metrics.get('valid_tool_calls', 0) > 0:
                stats['hit_rate_values'].append(metrics['hit_rate'])

        # 计算汇总统计
        result = {}
        for category, stats in category_stats.items():
            if stats['count'] == 0:
                continue

            result[category] = {
                'count': stats['count'],
                'pass@1': stats['correct'] / stats['count'] * 100,
                'avg_fcr': statistics.mean(stats['fcr_values']) if stats['fcr_values'] else 0,
                'avg_hit_rate': statistics.mean(stats['hit_rate_values']) if stats['hit_rate_values'] else 0,
                'avg_tool_calls': statistics.mean(stats['tool_calls']) if stats['tool_calls'] else 0,
                'avg_turns': statistics.mean(stats['turns']) if stats['turns'] else 0
            }

        # 按 Pass@1 排序，返回 top_k 最好和最差的
        sorted_categories = sorted(result.items(), key=lambda x: x[1]['pass@1'], reverse=True)

        return {
            'all': dict(sorted_categories),
            'best': dict(sorted_categories[:top_k]),
            'worst': dict(sorted_categories[-top_k:])
        }

    def analyze_tool_efficiency(self, model_name: str) -> Dict:
        """分析工具使用效率"""
        samples = self.models_data[model_name]

        # 按工具调用次数分组
        tool_bins = defaultdict(list)

        for sample in samples:
            metrics = self.extract_metrics(sample)
            tool_calls = metrics.get('all_tool_calls', 0)
            judgment = self.get_judgment(sample)

            # 确定分组
            if tool_calls == 0:
                bin_name = '0'
            elif tool_calls <= 2:
                bin_name = '1-2'
            elif tool_calls <= 5:
                bin_name = '3-5'
            elif tool_calls <= 10:
                bin_name = '6-10'
            else:
                bin_name = '10+'

            unique_facts = max(metrics.get('unique_facts_covered', 1), 1)

            tool_bins[bin_name].append({
                'correct': judgment == 'Correct',
                'fcr': metrics.get('fcr', 0),
                'hit_rate': metrics.get('hit_rate', 0),
                'format_errors': metrics.get('format_error_calls', 0),
                'redundancy_ratio': metrics.get('all_tool_calls', 1) / unique_facts
            })

        # 统计每个分组
        result = {}
        for bin_name in ['0', '1-2', '3-5', '6-10', '10+']:
            if bin_name not in tool_bins or len(tool_bins[bin_name]) == 0:
                continue

            samples_in_bin = tool_bins[bin_name]
            correct_count = sum(1 for s in samples_in_bin if s['correct'])

            result[bin_name] = {
                'count': len(samples_in_bin),
                'pass@1': correct_count / len(samples_in_bin) * 100,
                'avg_fcr': statistics.mean([s['fcr'] for s in samples_in_bin]),
                'avg_hit_rate': statistics.mean([s['hit_rate'] for s in samples_in_bin]),
                'avg_format_errors': statistics.mean([s['format_errors'] for s in samples_in_bin]),
                'avg_redundancy_ratio': statistics.mean([s['redundancy_ratio'] for s in samples_in_bin])
            }

        return result

    def analyze_turns_efficiency(self, model_name: str) -> Dict:
        """分析轮数效率"""
        samples = self.models_data[model_name]

        # 按轮数分组
        turn_bins = defaultdict(list)

        for sample in samples:
            turns = sample.get('actual_turns', 0)
            judgment = self.get_judgment(sample)
            metrics = self.extract_metrics(sample)

            # 确定分组
            if turns <= 2:
                bin_name = '1-2'
            elif turns <= 5:
                bin_name = '3-5'
            elif turns <= 10:
                bin_name = '6-10'
            elif turns <= 20:
                bin_name = '11-20'
            else:
                bin_name = '20+'

            turn_bins[bin_name].append({
                'correct': judgment == 'Correct',
                'fcr': metrics.get('fcr', 0),
                'tool_calls': metrics.get('all_tool_calls', 0)
            })

        # 统计每个分组
        result = {}
        for bin_name in ['1-2', '3-5', '6-10', '11-20', '20+']:
            if bin_name not in turn_bins or len(turn_bins[bin_name]) == 0:
                continue

            samples_in_bin = turn_bins[bin_name]
            correct_count = sum(1 for s in samples_in_bin if s['correct'])

            result[bin_name] = {
                'count': len(samples_in_bin),
                'pass@1': correct_count / len(samples_in_bin) * 100,
                'avg_fcr': statistics.mean([s['fcr'] for s in samples_in_bin]),
                'avg_tool_calls': statistics.mean([s['tool_calls'] for s in samples_in_bin])
            }

        return result

    def analyze_fcr_pass1_correlation(self, model_name: str) -> Dict:
        """分析 FCR 与 Pass@1 的相关性"""
        samples = self.models_data[model_name]

        # 按 FCR 区间分组
        fcr_bins = defaultdict(list)

        for sample in samples:
            metrics = self.extract_metrics(sample)
            fcr = metrics.get('fcr', 0)
            judgment = self.get_judgment(sample)

            # 确定分组
            if fcr == 0:
                bin_name = '0'
            elif fcr < 0.3:
                bin_name = '0-0.3'
            elif fcr < 0.5:
                bin_name = '0.3-0.5'
            elif fcr < 0.7:
                bin_name = '0.5-0.7'
            elif fcr < 0.8:
                bin_name = '0.7-0.8'
            else:
                bin_name = '0.8+'

            fcr_bins[bin_name].append(judgment == 'Correct')

        # 统计每个分组
        result = {}
        for bin_name in ['0', '0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.8', '0.8+']:
            if bin_name not in fcr_bins or len(fcr_bins[bin_name]) == 0:
                continue

            samples_in_bin = fcr_bins[bin_name]
            correct_count = sum(samples_in_bin)

            result[bin_name] = {
                'count': len(samples_in_bin),
                'pass@1': correct_count / len(samples_in_bin) * 100
            }

        # 计算相关系数 (Pearson)
        fcr_values = []
        correct_values = []
        for sample in samples:
            metrics = self.extract_metrics(sample)
            fcr = metrics.get('fcr', 0)
            judgment = self.get_judgment(sample)
            fcr_values.append(fcr)
            correct_values.append(1 if judgment == 'Correct' else 0)

        if len(fcr_values) > 1:
            correlation = np.corrcoef(fcr_values, correct_values)[0, 1]
        else:
            correlation = 0

        return {
            'bins': result,
            'pearson_correlation': correlation
        }

    def generate_report(self, output_dir: str):
        """生成定量分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name in self.models_data:
            print(f"\n{'='*80}")
            print(f"定量分析报告 (Setting B) - {model_name}")
            print(f"{'='*80}\n")

            # 1. 难度相关性
            print("## 1. 难度相关性分析\n")
            difficulty_analysis = self.analyze_difficulty_correlation(model_name)

            print(f"| {'难度':<10} | {'样本数':>7} | {'Pass@1':>8} | {'平均FCR':>9} | {'FCR标准差':>9} | {'平均命中率':>10} | {'平均调用':>9} | {'平均轮数':>9} |")
            print(f"|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*12}|{'-'*11}|{'-'*11}|")

            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in difficulty_analysis:
                    stats = difficulty_analysis[difficulty]
                    print(f"| {difficulty:<10} | {stats['count']:>7} | {stats['pass@1']:>7.2f}% | "
                          f"{stats['avg_fcr']:>9.3f} | {stats['std_fcr']:>9.3f} | "
                          f"{stats['avg_hit_rate']:>10.3f} | {stats['avg_tool_calls']:>9.2f} | "
                          f"{stats['avg_turns']:>9.2f} |")

            # 2. 工具使用效率
            print("\n## 2. 工具使用效率分析\n")
            tool_efficiency = self.analyze_tool_efficiency(model_name)

            print(f"| {'调用次数':>10} | {'样本数':>7} | {'Pass@1':>8} | {'平均FCR':>9} | {'平均命中率':>10} | {'平均格式错误':>12} | {'冗余比':>8} |")
            print(f"|{'-'*12}|{'-'*9}|{'-'*10}|{'-'*11}|{'-'*12}|{'-'*14}|{'-'*10}|")

            for bin_name in ['0', '1-2', '3-5', '6-10', '10+']:
                if bin_name in tool_efficiency:
                    stats = tool_efficiency[bin_name]
                    print(f"| {bin_name:>10} | {stats['count']:>7} | {stats['pass@1']:>7.2f}% | "
                          f"{stats['avg_fcr']:>9.3f} | {stats['avg_hit_rate']:>10.3f} | "
                          f"{stats['avg_format_errors']:>12.2f} | {stats['avg_redundancy_ratio']:>8.2f} |")

            # 3. FCR-Pass@1 相关性
            print("\n## 3. FCR-Pass@1 相关性分析\n")
            fcr_correlation = self.analyze_fcr_pass1_correlation(model_name)

            print(f"Pearson 相关系数: {fcr_correlation['pearson_correlation']:.4f}\n")

            print(f"| {'FCR区间':<10} | {'样本数':>7} | {'Pass@1':>8} |")
            print(f"|{'-'*12}|{'-'*9}|{'-'*10}|")

            for bin_name in ['0', '0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.8', '0.8+']:
                if bin_name in fcr_correlation['bins']:
                    stats = fcr_correlation['bins'][bin_name]
                    print(f"| {bin_name:<10} | {stats['count']:>7} | {stats['pass@1']:>7.2f}% |")

            # 保存完整数据
            report_data = {
                'model': model_name,
                'difficulty_analysis': difficulty_analysis,
                'category_analysis': self.analyze_category_performance(model_name),
                'tool_efficiency': tool_efficiency,
                'turns_efficiency': self.analyze_turns_efficiency(model_name),
                'fcr_correlation': fcr_correlation
            }

            json_path = output_path / f"quantitative_analysis_{model_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            print(f"\n详细数据已保存到: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='MPW-bench 定量分析 (Setting B)')
    parser.add_argument('--inputs', '-i', nargs='+', required=True,
                       help='评估结果文件路径列表')
    parser.add_argument('--output-dir', '-o', default='analysis_results/quantitative_settingB',
                       help='输出目录')

    args = parser.parse_args()

    analyzer = QuantitativeAnalyzerSettingB(args.inputs)
    analyzer.generate_report(args.output_dir)

    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
