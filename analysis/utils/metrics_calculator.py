#!/usr/bin/env python3
"""
metrics_calculator.py - 统一的指标计算模块

提供标准化的指标计算功能，被所有分析脚本使用
"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Optional
from pathlib import Path


class MetricsCalculator:
    """统一的指标计算器"""

    def __init__(self, samples: List[Dict[str, Any]]):
        """
        初始化

        Args:
            samples: 样本列表（从JSONL文件加载）
        """
        self.samples = samples

    @staticmethod
    def load_from_file(file_path: str) -> 'MetricsCalculator':
        """
        从JSONL文件加载数据并创建计算器

        Args:
            file_path: JSONL文件路径

        Returns:
            MetricsCalculator实例
        """
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return MetricsCalculator(samples)

    def calculate_overall_stats(self) -> Dict[str, Any]:
        """
        计算整体统计

        Returns:
            整体统计字典，包含:
            - total_samples: 总样本数
            - success_count: 成功样本数
            - failed_count: 失败样本数
            - success_rate: 成功率
            - avg_fcr: 平均FCR
            - avg_hit_rate: 平均命中率
            - avg_tool_calls: 平均工具调用数
            - avg_turns: 平均轮数
        """
        total = len(self.samples)
        if total == 0:
            return {}

        success_count = 0
        fcr_values = []
        hit_rate_values = []
        tool_calls_values = []
        turns_values = []

        for sample in self.samples:
            # 判断成功/失败
            status = sample.get('status', '')
            if status == 'finished':
                success_count += 1

            # 收集指标
            metrics = sample.get('metrics', {})
            fcr = metrics.get('fcr')
            if fcr is not None and fcr > 0:
                fcr_values.append(fcr)

            hit_precision = metrics.get('hit_precision')
            if hit_precision is not None and hit_precision > 0:
                hit_rate_values.append(hit_precision)

            tool_calls = metrics.get('all_tool_calls')
            if tool_calls is not None:
                tool_calls_values.append(tool_calls)

            actual_turns = metrics.get('actual_turns')
            if actual_turns is not None:
                turns_values.append(actual_turns)

        return {
            'total_samples': total,
            'success_count': success_count,
            'failed_count': total - success_count,
            'success_rate': success_count / total,
            'avg_fcr': sum(fcr_values) / len(fcr_values) if fcr_values else 0,
            'avg_hit_rate': sum(hit_rate_values) / len(hit_rate_values) if hit_rate_values else 0,
            'avg_tool_calls': sum(tool_calls_values) / len(tool_calls_values) if tool_calls_values else 0,
            'avg_turns': sum(turns_values) / len(turns_values) if turns_values else 0,
        }

    def calculate_category_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        按类别统计指标

        Returns:
            类别统计字典，格式:
            {
                'category_name': {
                    'count': 样本数,
                    'success_count': 成功数,
                    'success_rate': 成功率,
                    'avg_fcr': 平均FCR,
                    'avg_hit_rate': 平均命中率,
                    'avg_tool_calls': 平均工具调用数,
                    'format_error_rate': 格式错误率
                }
            }
        """
        category_data = defaultdict(lambda: {
            'count': 0,
            'success_count': 0,
            'fcr_values': [],
            'hit_rate_values': [],
            'tool_calls_values': [],
            'total_tool_calls': 0,
            'format_errors': 0,
        })

        for sample in self.samples:
            category = sample.get('category', 'unknown')
            stats = category_data[category]

            stats['count'] += 1

            # 成功统计
            status = sample.get('status', '')
            if status == 'finished':
                stats['success_count'] += 1

            # 指标统计
            metrics = sample.get('metrics', {})

            fcr = metrics.get('fcr')
            if fcr is not None and fcr > 0:
                stats['fcr_values'].append(fcr)

            hit_precision = metrics.get('hit_precision')
            if hit_precision is not None and hit_precision > 0:
                stats['hit_rate_values'].append(hit_precision)

            tool_calls = metrics.get('all_tool_calls', 0)
            stats['tool_calls_values'].append(tool_calls)
            stats['total_tool_calls'] += tool_calls

            format_errors = metrics.get('format_error_calls', 0)
            stats['format_errors'] += format_errors

        # 计算汇总统计
        result = {}
        for category, stats in category_data.items():
            if stats['count'] == 0:
                continue

            result[category] = {
                'count': stats['count'],
                'success_count': stats['success_count'],
                'success_rate': stats['success_count'] / stats['count'],
                'avg_fcr': sum(stats['fcr_values']) / len(stats['fcr_values']) if stats['fcr_values'] else 0,
                'avg_hit_rate': sum(stats['hit_rate_values']) / len(stats['hit_rate_values']) if stats['hit_rate_values'] else 0,
                'avg_tool_calls': sum(stats['tool_calls_values']) / len(stats['tool_calls_values']) if stats['tool_calls_values'] else 0,
                'format_error_rate': stats['format_errors'] / stats['total_tool_calls'] if stats['total_tool_calls'] > 0 else 0,
            }

        return result

    def calculate_difficulty_stats(self, difficulty_func) -> Dict[str, Dict[str, Any]]:
        """
        按难度级别统计指标

        Args:
            difficulty_func: 难度分级函数，接受sample返回difficulty字符串

        Returns:
            难度统计字典，格式:
            {
                'easy': {...},
                'medium': {...},
                'hard': {...}
            }
        """
        difficulty_data = defaultdict(lambda: {
            'count': 0,
            'success_count': 0,
            'fcr_values': [],
            'hit_rate_values': [],
            'tool_calls_values': [],
            'turns_values': [],
        })

        for sample in self.samples:
            difficulty = difficulty_func(sample)
            stats = difficulty_data[difficulty]

            stats['count'] += 1

            # 成功统计
            status = sample.get('status', '')
            if status == 'finished':
                stats['success_count'] += 1

            # 指标统计
            metrics = sample.get('metrics', {})

            fcr = metrics.get('fcr')
            if fcr is not None and fcr > 0:
                stats['fcr_values'].append(fcr)

            hit_precision = metrics.get('hit_precision')
            if hit_precision is not None and hit_precision > 0:
                stats['hit_rate_values'].append(hit_precision)

            tool_calls = metrics.get('all_tool_calls')
            if tool_calls is not None:
                stats['tool_calls_values'].append(tool_calls)

            actual_turns = metrics.get('actual_turns')
            if actual_turns is not None:
                stats['turns_values'].append(actual_turns)

        # 计算汇总统计
        result = {}
        for difficulty in ['easy', 'medium', 'hard']:
            stats = difficulty_data[difficulty]
            if stats['count'] == 0:
                continue

            result[difficulty] = {
                'count': stats['count'],
                'success_count': stats['success_count'],
                'success_rate': stats['success_count'] / stats['count'],
                'avg_fcr': sum(stats['fcr_values']) / len(stats['fcr_values']) if stats['fcr_values'] else 0,
                'avg_hit_rate': sum(stats['hit_rate_values']) / len(stats['hit_rate_values']) if stats['hit_rate_values'] else 0,
                'avg_tool_calls': sum(stats['tool_calls_values']) / len(stats['tool_calls_values']) if stats['tool_calls_values'] else 0,
                'avg_turns': sum(stats['turns_values']) / len(stats['turns_values']) if stats['turns_values'] else 0,
            }

        return result

    def get_tool_call_stats(self) -> Dict[str, Any]:
        """
        获取工具调用统计

        Returns:
            工具调用统计字典
        """
        total_tool_calls = 0
        valid_tool_calls = 0
        format_errors = 0
        total_hits = 0

        for sample in self.samples:
            metrics = sample.get('metrics', {})
            total_tool_calls += metrics.get('all_tool_calls', 0)
            valid_tool_calls += metrics.get('valid_tool_calls', 0)
            format_errors += metrics.get('format_error_calls', 0)
            total_hits += metrics.get('total_hits', 0)

        return {
            'total_tool_calls': total_tool_calls,
            'valid_tool_calls': valid_tool_calls,
            'format_error_calls': format_errors,
            'total_hits': total_hits,
            'format_error_rate': format_errors / total_tool_calls if total_tool_calls > 0 else 0,
            'hit_rate': total_hits / valid_tool_calls if valid_tool_calls > 0 else 0,
        }

    def filter_by_category(self, category: str) -> 'MetricsCalculator':
        """
        按类别过滤样本

        Args:
            category: 类别名称

        Returns:
            新的MetricsCalculator实例
        """
        filtered = [s for s in self.samples if s.get('category') == category]
        return MetricsCalculator(filtered)

    def filter_by_difficulty(self, difficulty: str, difficulty_func) -> 'MetricsCalculator':
        """
        按难度过滤样本

        Args:
            difficulty: 难度级别
            difficulty_func: 难度分级函数

        Returns:
            新的MetricsCalculator实例
        """
        filtered = [s for s in self.samples if difficulty_func(s) == difficulty]
        return MetricsCalculator(filtered)

    def filter_by_status(self, status: str) -> 'MetricsCalculator':
        """
        按状态过滤样本

        Args:
            status: 状态字符串 (如 'finished', 'failed')

        Returns:
            新的MetricsCalculator实例
        """
        filtered = [s for s in self.samples if s.get('status') == status]
        return MetricsCalculator(filtered)
