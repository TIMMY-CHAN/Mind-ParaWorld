#!/usr/bin/env python3
"""
sample_state.py - 样本状态管理

用于在异步 turn 之间持久化样本状态
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from copy import deepcopy


@dataclass
class SampleState:
    """
    单个样本的完整状态

    在每个 turn 完成后，状态会被更新并传递到下一个 turn
    """

    # 基础信息
    index: int                          # 样本索引
    data: Dict[str, Any]                # 原始数据（prompt, images, answer等）

    # Agent 状态
    messages: List[Dict]                # 对话历史（可能被截断，用于推理）
    full_messages: List[Dict] = field(default_factory=list)  # 完整对话历史（不截断，用于保存）
    turn: int = 0                       # 当前轮数
    max_turns: int = 8                  # 最大轮数

    # 执行状态
    status: str = "running"             # "running" | "finished" | "max_turns_reached" | "api_error"
    final_answer: str = ""              # 最终答案
    full_response_log: str = ""         # 完整响应日志
    error_info: Dict[str, Any] = field(default_factory=dict)  # 错误信息（仅当 status="api_error" 时）

    # 评测数据
    trajectory_log: Dict[str, Any] = field(default_factory=dict)  # 轨迹日志
    ground_truth: str = ""              # 真实答案
    extra_info: Dict[str, Any] = field(default_factory=dict)  # 额外信息（world_truth等）

    # Agent 实例（需要保持状态）
    agent: Any = None  # AgentEval 实例

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        # 计算实际对话轮次（仅统计 assistant 消息）
        actual_turns = len([msg for msg in self.messages if msg.get('role') == 'assistant']) if self.messages else 0

        return {
            "index": self.index,
            "turn": actual_turns,  # ✅ 修复：使用实际对话轮次而非累计调用次数
            "status": self.status,
            "final_answer": self.final_answer,
            "trajectory_log": self.trajectory_log,
            "messages_count": len(self.messages),
        }

    def clone(self) -> 'SampleState':
        """深拷贝状态（除了 agent）"""
        return SampleState(
            index=self.index,
            data=deepcopy(self.data),
            messages=deepcopy(self.messages),
            full_messages=deepcopy(self.full_messages),  # 也拷贝完整消息
            turn=self.turn,
            max_turns=self.max_turns,
            status=self.status,
            final_answer=self.final_answer,
            full_response_log=self.full_response_log,
            error_info=deepcopy(self.error_info),  # 拷贝错误信息
            trajectory_log=deepcopy(self.trajectory_log),
            ground_truth=self.ground_truth,
            extra_info=deepcopy(self.extra_info),
            agent=self.agent  # Agent 不拷贝，保持引用
        )


class StateManager:
    """
    全局状态管理器

    负责：
    1. 初始化所有样本的状态
    2. 跟踪样本完成情况
    3. 生成最终结果
    """

    def __init__(self):
        self.states: Dict[int, SampleState] = {}  # index -> SampleState
        self.completed: List[int] = []            # 已完成的样本索引

    def add_state(self, state: SampleState):
        """添加一个样本状态"""
        self.states[state.index] = state

    def get_state(self, index: int) -> Optional[SampleState]:
        """获取样本状态"""
        return self.states.get(index)

    def mark_completed(self, index: int):
        """标记样本完成"""
        if index not in self.completed:
            self.completed.append(index)

    def is_completed(self, index: int) -> bool:
        """检查样本是否完成"""
        return index in self.completed

    def get_completion_rate(self) -> float:
        """获取完成率"""
        if not self.states:
            return 0.0
        return len(self.completed) / len(self.states)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.states)
        completed = len(self.completed)
        running = total - completed

        # 各状态统计
        status_counts = {}
        for state in self.states.values():
            status = state.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total": total,
            "completed": completed,
            "running": running,
            "completion_rate": self.get_completion_rate(),
            "status_breakdown": status_counts
        }

    def export_results(self) -> List[Dict[str, Any]]:
        """导出所有样本的结果"""
        results = []
        for state in self.states.values():
            results.append(self._state_to_result(state))
        return sorted(results, key=lambda x: x["index"])

    def _state_to_result(self, state: SampleState) -> Dict[str, Any]:
        """将状态转换为结果格式（仅保存原始数据，不计算指标）

        核心原则：只保存推理过程中产生的原始数据
        - 完整轨迹（full_messages）
        - 预测答案（final_answer）
        - 状态（status）
        - 轨迹日志（trajectory_log）
        - 错误信息（error_info，仅当 api_error 时）

        指标计算（如 FCR、hit_precision 等）由离线脚本处理
        """
        result = {
            "index": state.index,
            "category": state.data.get("category"),
            "messages": self._simplify_messages(state.full_messages if state.full_messages else state.messages),
            "ground_truth": state.ground_truth,
            "prediction": state.final_answer,
            "full_response": state.full_response_log,
            "status": state.status,
            "trajectory_log": state.trajectory_log,
            # 添加实际对话轮次（从 full_messages 统计）
            "actual_turns": len([msg for msg in (state.full_messages or state.messages) if msg.get('role') == 'assistant']),
            # 添加最大轮数限制（用于离线分析）
            "max_turns": state.max_turns,
        }

        # 如果是 api_error，添加错误信息
        if state.status == "api_error" and state.error_info:
            result["error_info"] = state.error_info

        return result

    def _simplify_messages(self, messages: List[Dict]) -> List[Dict]:
        """简化消息（移除图片 base64）"""
        import copy
        simple = copy.deepcopy(messages)
        for msg in simple:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        item['image_url']['url'] = "[IMAGE_BASE64_REMOVED]"
        return simple


if __name__ == "__main__":
    # 单元测试
    print("=" * 80)
    print("  测试 SampleState 和 StateManager")
    print("=" * 80)
    print()

    # 创建状态管理器
    manager = StateManager()

    # 添加样本
    for i in range(5):
        state = SampleState(
            index=i,
            data={"prompt": [{"role": "user", "content": f"Question {i}"}]},
            messages=[{"role": "system", "content": "System prompt"}],
            ground_truth=f"Answer {i}",
            extra_info={"world_truth_info": {"atomic_facts": {"fact1": "value1"}}}
        )
        manager.add_state(state)

    print(f"✅ 添加了 5 个样本")
    print(f"   Stats: {manager.get_stats()}")
    print()

    # 模拟完成一些样本
    manager.states[0].status = "finished"
    manager.states[0].final_answer = "Answer 0"
    manager.mark_completed(0)

    manager.states[1].status = "finished"
    manager.states[1].final_answer = "Answer 1"
    manager.mark_completed(1)

    print(f"✅ 完成了 2 个样本")
    print(f"   Stats: {manager.get_stats()}")
    print()

    # 导出结果
    results = manager.export_results()
    print(f"✅ 导出结果: {len(results)} 条")
    print(f"   第一条: index={results[0]['index']}, status={results[0]['status']}")
    print()

    print("=" * 80)
    print("  测试通过！")
    print("=" * 80)
