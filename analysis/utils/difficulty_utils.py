#!/usr/bin/env python3
"""
difficulty_utils.py - 统一的难度分级工具

提供标准化的难度分级逻辑，被所有分析脚本使用
"""

from typing import Dict, Any


# 难度级别定义
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

DIFFICULTY_LABELS = {
    'easy': '🟢 Easy',
    'medium': '🟡 Medium',
    'hard': '🔴 Hard'
}

DIFFICULTY_THRESHOLDS = {
    'easy': (1, 5),      # 1-5 atomic facts
    'medium': (6, 10),   # 6-10 atomic facts
    'hard': (11, 999)    # ≥11 atomic facts
}


def get_difficulty(sample: Dict[str, Any]) -> str:
    """
    获取样本的难度级别

    Args:
        sample: 样本数据，需要包含atomic_facts信息

    Returns:
        难度级别字符串: 'easy', 'medium', 'hard'

    难度定义:
        - Easy: 1-5 个原子事实
        - Medium: 6-10 个原子事实
        - Hard: ≥11 个原子事实
    """
    # 从不同的数据结构中提取atomic_facts
    atomic_facts = None

    # 方式1: 从data.world_truth_info获取（evaluated_results格式）
    if 'data' in sample:
        world_truth_info = sample.get('data', {}).get('world_truth_info', {})
        atomic_facts = world_truth_info.get('atomic_facts', {})

    # 方式2: 从trajectory_log获取（inference_results格式）
    elif 'trajectory_log' in sample:
        world_truth_info = sample.get('trajectory_log', {}).get('world_truth_info', {})
        atomic_facts = world_truth_info.get('atomic_facts', {})

    # 方式3: 直接从metrics获取total_facts（如果有）
    elif 'metrics' in sample:
        total_facts = sample.get('metrics', {}).get('total_facts', 0)
        if total_facts > 0:
            num_facts = total_facts
        else:
            num_facts = 0

    # 计算原子事实数量
    if atomic_facts is not None:
        if isinstance(atomic_facts, dict):
            num_facts = len(atomic_facts)
        elif isinstance(atomic_facts, list):
            num_facts = len(atomic_facts)
        else:
            num_facts = 0
    else:
        num_facts = 0

    # 根据数量分配难度
    if num_facts <= 5:
        return 'easy'
    elif num_facts <= 10:
        return 'medium'
    else:
        return 'hard'


def get_difficulty_label(difficulty: str) -> str:
    """
    获取难度的显示标签（带emoji）

    Args:
        difficulty: 难度级别 ('easy', 'medium', 'hard')

    Returns:
        带emoji的难度标签
    """
    return DIFFICULTY_LABELS.get(difficulty, difficulty)


def get_num_facts_from_sample(sample: Dict[str, Any]) -> int:
    """
    从样本中提取原子事实数量

    Args:
        sample: 样本数据

    Returns:
        原子事实数量
    """
    # 方式1: 从data.world_truth_info获取
    if 'data' in sample:
        world_truth_info = sample.get('data', {}).get('world_truth_info', {})
        atomic_facts = world_truth_info.get('atomic_facts', {})
        if isinstance(atomic_facts, (dict, list)):
            return len(atomic_facts)

    # 方式2: 从trajectory_log获取
    if 'trajectory_log' in sample:
        world_truth_info = sample.get('trajectory_log', {}).get('world_truth_info', {})
        atomic_facts = world_truth_info.get('atomic_facts', {})
        if isinstance(atomic_facts, (dict, list)):
            return len(atomic_facts)

    # 方式3: 从metrics获取
    if 'metrics' in sample:
        total_facts = sample.get('metrics', {}).get('total_facts', 0)
        if total_facts > 0:
            return total_facts

    return 0


def get_difficulty_range(difficulty: str) -> tuple:
    """
    获取难度对应的事实数量范围

    Args:
        difficulty: 难度级别

    Returns:
        (min_facts, max_facts) 元组
    """
    return DIFFICULTY_THRESHOLDS.get(difficulty, (0, 0))
