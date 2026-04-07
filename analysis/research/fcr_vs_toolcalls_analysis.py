#!/usr/bin/env python3
"""
【Setting C】工具调用次数 vs FCR（消息约束口径）统计脚本

统计对象
- 输入：evaluation pipeline 产出的 Setting C re_evaluated_results.jsonl（逐行 JSON）。
- 仅统计 status ∈ {finished, max_turns_reached} 且包含 trajectory_log/world_truth_info 的样本。

关键口径（非常重要）
- "有效工具调用次数" effective_calls：从修复后的 sample["messages"] 中统计。
  *规则*：每个 assistant turn 至多计 1 次工具调用；如果该 turn 的 content 字符串包含 '<tool_call>' 则计 1。
- 为对齐该口径，会将 trajectory_log.hit_logs 裁剪为 hit_logs[:effective_calls]。
  目的：避免历史 bug 导致 hit_logs 长度虚高，从而在后处理里错误地把“并不存在的调用”也计入 FCR。

FCR 定义
- world_truth_info.atomic_facts 给出该样本的原子事实集合。
- hit_logs[*].matched_fact_keys 给出该次 tool call 覆盖到的事实 key。
- 累计 FCR(k) = 截止第 k 次有效 tool call，累计覆盖到的 unique fact keys / total_facts。

输出
- 生成 /.../fcr_vs_toolcalls_data.json
  - fcr_curve：每 2 次调用为 1 个刻度（interval），统计该刻度末尾的平均累计 FCR 与方差（注意：每个样本在每个刻度只贡献 1 次，避免长轨迹样本被过度加权）。
  - per_sample_series：每个样本的逐 call 序列 fcr_by_call / new_facts_by_call，用于 cohort/bins/个例轨迹可视化。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / 'setting_c_results'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'fcr_vs_toolcalls_data.json'


def analyze_model(model_name, file_path, max_effective_calls: int = 32):
    """
    分析单个模型的工具调用与 FCR 关系

    Setting C 口径对齐：以修复后的 sample["messages"] 约束有效 tool_call 次数，
    并裁剪 trajectory_log.hit_logs 到有效长度后再计算累计 FCR。

    Returns:
        dict: {
            'call_interval_fcr': {1: [fcr1, fcr2, ...], 2: [...], ...},
            'samples_processed': int,
            'samples_with_tools': int,
            'avg_total_calls': float,
            'avg_final_fcr': float
        }
    """

    def _count_effective_tool_calls_from_messages(sample: dict) -> int:
        """每个 assistant turn 至多计 1 次 tool_call。"""
        messages = sample.get('messages', []) or []
        cnt = 0
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            content = msg.get('content', '')
            if isinstance(content, str) and '<tool_call>' in content:
                cnt += 1
        return cnt

    print(f"\n{'='*60}")
    print(f"分析模型: {model_name}")
    print(f"{'='*60}")

    # 存储：刻度 -> 该刻度下的 FCR 列表（每个样本在每个刻度最多贡献 1 次，避免长轨迹样本被过度加权）
    call_interval_fcr = defaultdict(list)

    # 额外导出：每个样本的序列（用于 cohort/bins 和个例可视化）
    # 仅保留必要字段，避免文件过大。
    per_sample_series = []

    samples_processed = 0
    samples_with_tools = 0
    total_calls = 0
    final_fcrs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                samples_processed += 1

                # 只处理完成的样本（finished 或 max_turns_reached）
                status = sample.get('status', '')
                if status not in ['finished', 'max_turns_reached']:
                    continue

                # 获取 trajectory_log
                traj = sample.get('trajectory_log', {})
                if not traj:
                    continue

                hit_logs = traj.get('hit_logs', [])
                world_truth = traj.get('world_truth_info', {})
                atomic_facts = world_truth.get('atomic_facts', {})

                effective_calls = _count_effective_tool_calls_from_messages(sample)
                hit_logs_eff = hit_logs[:effective_calls] if effective_calls > 0 else []

                if not hit_logs_eff or not atomic_facts:
                    continue

                samples_with_tools += 1
                total_calls += len(hit_logs_eff)

                # 计算累计 FCR
                total_facts = len(atomic_facts)
                covered_facts = set()  # 累计覆盖的事实

                # per-call 序列（长度 = effective_calls），用于后续 cohort/bins/个例图
                fcr_by_call = []
                new_facts_by_call = []
                hit_by_call = []  # trajectory_log.hit_logs[i].hit (0/1)

                last_interval = 0
                for call_idx, hit_log in enumerate(hit_logs_eff[:max_effective_calls]):
                    matched_keys = hit_log.get('matched_fact_keys', [])
                    hit_by_call.append(1 if hit_log.get('hit', 0) else 0)
                    before = len(covered_facts)
                    if matched_keys:
                        covered_facts.update(matched_keys)
                    after = len(covered_facts)

                    new_facts = after - before
                    current_fcr = after / total_facts if total_facts > 0 else 0

                    fcr_by_call.append(current_fcr)
                    new_facts_by_call.append(new_facts)

                    # 映射到刻度（每 2 次调用为 1 个刻度）
                    interval = (call_idx // 2) + 1  # 刻度从 1 开始

                    # 关键：同一个样本在同一刻度只记录一次（该刻度最后一次调用后的累计 FCR）
                    if interval != last_interval:
                        call_interval_fcr[interval].append(current_fcr)
                        last_interval = interval
                    else:
                        call_interval_fcr[interval][-1] = current_fcr

                # 记录最终 FCR（在有效调用裁剪范围内）
                final_fcr = len(covered_facts) / total_facts if total_facts > 0 else 0
                final_fcrs.append(final_fcr)

                per_sample_series.append({
                    'index': sample.get('index'),
                    'category': sample.get('category', 'unknown'),
                    'effective_calls': len(fcr_by_call),
                    'total_facts': total_facts,
                    'final_fcr': final_fcr,
                    'fcr_by_call': fcr_by_call,
                    'new_facts_by_call': new_facts_by_call,
                    'hit_by_call': hit_by_call,
                })

                # 进度显示
                if samples_with_tools % 100 == 0:
                    print(f"  已处理 {samples_with_tools} 个有效样本...")

            except Exception as e:
                print(f"  ⚠️  行 {line_num} 解析错误: {e}")
                continue

    # 计算统计信息
    avg_total_calls = total_calls / samples_with_tools if samples_with_tools > 0 else 0
    avg_final_fcr = sum(final_fcrs) / len(final_fcrs) if final_fcrs else 0

    print(f"\n统计结果:")
    print(f"  总样本数: {samples_processed}")
    print(f"  有效样本数（有工具调用且完成）: {samples_with_tools}")
    print(f"  平均工具调用次数: {avg_total_calls:.2f}")
    print(f"  平均最终 FCR: {avg_final_fcr:.4f}")
    print(f"  刻度数量: {len(call_interval_fcr)}")

    return {
        'model_name': model_name,
        'call_interval_fcr': dict(call_interval_fcr),
        'per_sample_series': per_sample_series,
        'samples_processed': samples_processed,
        'samples_with_tools': samples_with_tools,
        'avg_total_calls': avg_total_calls,
        'avg_final_fcr': avg_final_fcr
    }


def compute_statistics(model_data):
    """
    计算每个刻度的统计信息

    Returns:
        list: [(刻度, 平均FCR, 标准差, 样本数), ...]
    """
    call_interval_fcr = model_data['call_interval_fcr']

    stats = []
    for interval in sorted(call_interval_fcr.keys()):
        fcrs = call_interval_fcr[interval]
        if not fcrs:
            continue

        avg_fcr = sum(fcrs) / len(fcrs)

        # 计算标准差
        if len(fcrs) > 1:
            variance = sum((x - avg_fcr) ** 2 for x in fcrs) / len(fcrs)
            std_fcr = variance ** 0.5
        else:
            std_fcr = 0

        stats.append((interval, avg_fcr, std_fcr, len(fcrs)))

    return stats


def find_saturation_point(stats, threshold=0.01):
    """
    找到 FCR 饱和点

    Args:
        stats: [(刻度, 平均FCR, 标准差, 样本数), ...]
        threshold: FCR 增长率阈值

    Returns:
        int: 饱和刻度（从该刻度开始，增长率 < threshold）
    """
    if len(stats) < 2:
        return None

    for i in range(1, len(stats)):
        prev_interval, prev_fcr, _, _ = stats[i-1]
        curr_interval, curr_fcr, _, _ = stats[i]

        # 计算增长率
        if prev_fcr > 0:
            growth_rate = (curr_fcr - prev_fcr) / prev_fcr
        else:
            growth_rate = curr_fcr

        # 如果增长率低于阈值，认为已饱和
        if growth_rate < threshold:
            return curr_interval

    return None


def compute_marginal_gain(stats):
    """
    计算边际收益

    Returns:
        list: [(刻度, 边际FCR增益), ...]
    """
    if len(stats) < 2:
        return []

    gains = []
    for i in range(1, len(stats)):
        prev_interval, prev_fcr, _, _ = stats[i-1]
        curr_interval, curr_fcr, _, _ = stats[i]

        marginal_gain = curr_fcr - prev_fcr
        gains.append((curr_interval, marginal_gain))

    return gains


def main():
    # 支持从命令行指定模型名列表：--models a,b,c
    # 默认基目录：--models-dir /.../setting_c_results
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        required=True,
        help='Comma-separated model names under models-dir, e.g. MindWatcher_round2,minimax-m2_1',
    )
    parser.add_argument(
        '--models-dir',
        default=str(DEFAULT_MODELS_DIR),
        help='Base dir containing <model>/re_evaluated_results.jsonl',
    )
    parser.add_argument(
        '--max-effective-calls',
        type=int,
        default=32,
        help='Cap effective tool calls for per-sample series',
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT_PATH),
        help='Output JSON file path',
    )
    args = parser.parse_args()

    base_dir = Path(args.models_dir)
    model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    if not model_names:
        raise SystemExit('Empty --models')

    # 定义模型：在 setting_c_results/<model>/re_evaluated_results.jsonl
    models = []
    for m in model_names:
        fp = base_dir / m / 're_evaluated_results.jsonl'
        models.append((m, str(fp)))

    # 分析每个模型
    all_model_stats = {}

    for model_name, file_path in models:
        model_data = analyze_model(model_name, file_path, max_effective_calls=args.max_effective_calls)
        stats = compute_statistics(model_data)
        saturation_point = find_saturation_point(stats)
        marginal_gains = compute_marginal_gain(stats)

        all_model_stats[model_name] = {
            'model_data': {
                'model_name': model_name,
                'samples_processed': model_data['samples_processed'],
                'samples_with_tools': model_data['samples_with_tools'],
                'avg_total_calls': model_data['avg_total_calls'],
                'avg_final_fcr': model_data['avg_final_fcr'],
                'per_sample_series': model_data.get('per_sample_series', []),
            },
            'stats': stats,
            'saturation_point': saturation_point,
            'marginal_gains': marginal_gains
        }

    # 打印汇总结果
    print(f"\n\n{'='*80}")
    print("汇总分析")
    print(f"{'='*80}\n")

    for model_name, data in all_model_stats.items():
        print(f"\n{model_name}:")
        print(f"  有效样本数: {data['model_data']['samples_with_tools']}")
        print(f"  平均工具调用: {data['model_data']['avg_total_calls']:.2f}")
        print(f"  平均最终 FCR: {data['model_data']['avg_final_fcr']:.4f}")
        print(f"  饱和刻度: {data['saturation_point'] if data['saturation_point'] else '未饱和'}")

        # 打印前 10 个刻度的 FCR
        print(f"\n  FCR 曲线（前 10 个刻度）:")
        for interval, avg_fcr, std_fcr, sample_count in data['stats'][:10]:
            print(f"    刻度 {interval:2d}: 平均 FCR={avg_fcr:.4f} (±{std_fcr:.4f}), 样本数={sample_count}")

        # 打印边际收益
        print(f"\n  边际收益（前 10 个刻度）:")
        for interval, gain in data['marginal_gains'][:10]:
            print(f"    刻度 {interval:2d}: 边际 FCR 增益={gain:+.4f}")

    # 保存详细数据用于绘图
    output_file = args.output
    output_data = {}

    for model_name, data in all_model_stats.items():
        output_data[model_name] = {
            'model_data': data['model_data'],
            'saturation_point': data['saturation_point'],
            'fcr_curve': [
                {
                    'interval': interval,
                    'avg_fcr': avg_fcr,
                    'std_fcr': std_fcr,
                    'sample_count': sample_count
                }
                for interval, avg_fcr, std_fcr, sample_count in data['stats']
            ],
            'marginal_gains': [
                {
                    'interval': interval,
                    'marginal_fcr_gain': gain
                }
                for interval, gain in data['marginal_gains']
            ]
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n\n✅ 统计数据已保存到: {output_file}")


if __name__ == '__main__':
    main()
