#!/usr/bin/env python3
"""
生成 Setting C 完整结果分析总表
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def load_model_data(json_path: str) -> Dict:
    """加载模型分析数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_overall(difficulty_analysis: Dict) -> Dict:
    """计算 Overall 指标"""
    total_count = 0
    total_pass1 = 0
    total_fcr = 0
    total_hr = 0
    total_tool_calls = 0
    
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in difficulty_analysis:
            stats = difficulty_analysis[difficulty]
            count = stats['count']
            total_count += count
            total_pass1 += stats['pass@1'] * count
            total_fcr += stats['avg_fcr'] * count
            total_hr += stats['avg_hit_rate'] * count
            total_tool_calls += stats['avg_tool_calls'] * count
    
    if total_count == 0:
        return {}
    
    return {
        'count': total_count,
        'pass@1': total_pass1 / total_count,
        'avg_fcr': total_fcr / total_count,
        'avg_hit_rate': total_hr / total_count,
        'avg_tool_calls': total_tool_calls / total_count
    }

def generate_summary_table(results_dir: str, output_file: str):
    """生成汇总表格"""
    results_path = Path(results_dir)
    
    # 收集所有模型数据
    models_data = {}
    for json_file in sorted(results_path.glob('quantitative_analysis_*.json')):
        model_name = json_file.stem.replace('quantitative_analysis_', '')
        models_data[model_name] = load_model_data(str(json_file))
    
    # 生成 Markdown 表格
    lines = []
    lines.append("# Setting C 完整结果分析总表\n")
    lines.append(f"生成时间: 2025-02-24\n")
    lines.append("## 说明")
    lines.append("- **难度**: E=Easy, M=Medium, H=Hard, O=Overall")
    lines.append("- **指标**: Pass@1 (%), FCR (事实覆盖率), HR (命中率), Tool Calls (平均调用次数)")
    lines.append("- **Hit Rate定义**: 只计算有工具调用的轨迹，HR = 命中query数 / 总query数")
    lines.append("- **数据文件**: re_evaluated_results.jsonl\n")
    
    lines.append("| 模型 | 难度 | Pass@1 (%) | FCR | Hit Rate | Tool Calls |")
    lines.append("|------|------|------------|-----|----------|------------|")
    
    # 按模型名排序
    for model_name in sorted(models_data.keys()):
        data = models_data[model_name]
        diff_analysis = data['difficulty_analysis']
        
        # 添加 Easy/Medium/Hard
        for idx, difficulty in enumerate(['easy', 'medium', 'hard']):
            if difficulty not in diff_analysis:
                continue
            
            stats = diff_analysis[difficulty]
            difficulty_label = {'easy': 'E', 'medium': 'M', 'hard': 'H'}[difficulty]
            
            if idx == 0:
                lines.append(f"| {model_name} | {difficulty_label} | {stats['pass@1']:.2f} | "
                           f"{stats['avg_fcr']:.3f} | {stats['avg_hit_rate']:.3f} | "
                           f"{stats['avg_tool_calls']:.2f} |")
            else:
                lines.append(f"|  | {difficulty_label} | {stats['pass@1']:.2f} | "
                           f"{stats['avg_fcr']:.3f} | {stats['avg_hit_rate']:.3f} | "
                           f"{stats['avg_tool_calls']:.2f} |")
        
        # 添加 Overall
        overall = calculate_overall(diff_analysis)
        if overall:
            lines.append(f"|  | O | {overall['pass@1']:.2f} | "
                       f"{overall['avg_fcr']:.3f} | {overall['avg_hit_rate']:.3f} | "
                       f"{overall['avg_tool_calls']:.2f} |")
        
        lines.append("|------|------|------------|-----|----------|------------|")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"汇总表格已生成: {output_file}")
    print(f"包含 {len(models_data)} 个模型")

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'analysis_results/quantitative_settingC'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'analysis_results/setting_c_summary.md'
    
    generate_summary_table(results_dir, output_file)
