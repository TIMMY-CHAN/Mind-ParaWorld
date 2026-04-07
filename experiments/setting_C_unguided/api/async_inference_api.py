#!/usr/bin/env python3
"""
async_inference_api.py - 基于闭源 API 的异步推理引擎

核心特性：
1. Turn 级别异步调度（保持高效率）
2. Token Bucket 速率限制（避免 429 错误）
3. 指数退避重试（应对 API 不稳定）
4. 断点续传支持（中断后可继续）
5. 成本监控（实时统计 token 使用）

使用示例：
    python async_inference_api.py \\
        --input data/mpw_bench_full.jsonl \\
        --output results/api_results.jsonl \\
        --api-provider openai \\
        --model gpt-4 \\
        --qps 5 \\
        --max-concurrent 50
"""

import asyncio
import json
import time
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

# 确保项目根目录在 sys.path 中，支持从任意工作目录运行
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from verl.sample_state import SampleState, StateManager
from verl.workers.agent.envs.agent_eval import AgentEval

# System Prompt (与 async_inference.py 保持一致)
INSTRUCTION_PROMPT_SYSTEM = """
你是一个ReAct范式的多模态agent，能够接受图像和文本输入，回答用户问题
对于一些复杂的问题，你可以选择调用工具帮助你解决问题
你可以调用的工具包括以下两种：
web_search:
-description: Retrieve external text information from internet based on your provided text query.
-input: only text query(**this tool cannot see the image**)
-output: top-5 text(you can attempt to change your query if the previous search result is not satisfactory)
-Usage:
<tool_call>
{
    "name": "web_search",
    "arguments":
    {
        "query": "the content",  # The text query you provided.
    }
}
</tool_call>

对于每一个问题，你需要先思考，然后调用工具（如果需要），你会得到工具调用返回的结果，还可以根据工具的返回结果进行进一步的思考，最后给出答案
你的思考过程，工具调用请求以及回答需要严格按照以下格式：
<think>
你的思考过程
</think>

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call> (如果需要调用工具,你的工具调用请求参考usage中的示例)

<think>
你的思考过程
</think> (如果需要进一步思考)

<answer>
你的最终答案
</answer>

请记住，你在每次调用工具之后，也就是输出</tool_call>之后，都需要结束本轮对话，等待工具调用的结果返回，再进行后续动作
在输出回答之后，即在输出</answer>之后，你需要立即结束本轮对话，不要再输出任何内容
你的思考次数和工具调用次数没有限制，但必须在最后给出你的答案
对于任何问题，你不应该拒绝回答，而应该通过不断思考或调用工具，直到得到确信的结果
"""


# ==================== Token Bucket 速率限制器 ====================

class TokenBucket:
    """
    令牌桶算法实现 QPS/QPM 速率限制

    原理：
    - 桶中有固定容量的令牌
    - 每秒按照 rate 速率补充令牌
    - 每次请求消耗 1 个令牌
    - 无令牌时阻塞等待
    """
    def __init__(self, rate: float, capacity: int, name: str = "limiter"):
        """
        Args:
            rate: 每秒生成令牌数（QPS）
            capacity: 桶容量（允许突发请求数）
            name: 限制器名称（用于日志）
        """
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

        # 统计信息
        self.total_acquired = 0
        self.total_wait_time = 0.0

    async def acquire(self, tokens: int = 1):
        """获取指定数量的令牌（阻塞直到有足够令牌）"""
        async with self.lock:
            wait_start = time.time()

            while True:
                now = time.time()
                elapsed = now - self.last_update

                # 补充令牌
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.total_acquired += tokens
                    wait_time = time.time() - wait_start
                    self.total_wait_time += wait_time
                    return

                # 计算需要等待的时间
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
                await asyncio.sleep(wait_time)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "rate": self.rate,
            "capacity": self.capacity,
            "current_tokens": self.tokens,
            "total_acquired": self.total_acquired,
            "total_wait_time": self.total_wait_time,
            "avg_wait_time": self.total_wait_time / max(1, self.total_acquired)
        }


# ==================== API 客户端工厂 ====================

class APIClientFactory:
    """统一的 API 客户端工厂"""

    @staticmethod
    def create_client(provider: str, api_key: str = None, base_url: str = None) -> AsyncOpenAI:
        """
        创建异步 API 客户端

        支持的 provider:
        - openai: OpenAI 官方 API
        - azure: Azure OpenAI
        - custom: 自定义（需要提供 base_url）
        """
        if provider == "openai":
            return AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )

        elif provider == "azure":
            # Azure OpenAI 配置
            return AsyncOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
            )

        elif provider == "custom":
            if not base_url:
                raise ValueError("Custom provider requires base_url")
            return AsyncOpenAI(
                api_key=api_key or "dummy",
                base_url=base_url
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")


# ==================== 异步推理引擎（API 版本）====================

class AsyncInferenceEngineAPI:
    """
    基于闭源 API 的异步推理引擎

    核心设计：
    1. Turn 级别异步调度（而非 Sample 级别）
    2. Token Bucket 全局速率限制
    3. 指数退避重试机制
    4. 断点续传支持
    """

    def __init__(
        self,
        # API 配置
        api_provider: str = "openai",
        api_key: str = None,
        base_url: str = None,
        model_name: str = "gpt-4",

        # 并发配置
        max_concurrent_turns: int = 100,  # Turn 级别最大并发

        # 速率限制
        qps: float = 10,                  # 每秒请求数
        qpm: float = None,                # 每分钟请求数（可选）

        # 重试配置
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,

        # Agent 配置
        max_turns_per_sample: int = 32,

        # 其他
        enable_cost_tracking: bool = True,
    ):
        # API 客户端
        self.client = APIClientFactory.create_client(api_provider, api_key, base_url)
        self.model_name = model_name
        self.api_provider = api_provider

        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.max_concurrent_turns = max_concurrent_turns

        # 速率限制
        self.qps_limiter = TokenBucket(rate=qps, capacity=int(qps * 2), name="QPS")
        self.qpm_limiter = TokenBucket(rate=qpm / 60, capacity=qpm, name="QPM") if qpm else None

        # 重试配置
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        # Agent 配置
        self.max_turns_per_sample = max_turns_per_sample

        # 成本跟踪
        self.enable_cost_tracking = enable_cost_tracking
        self.total_prompt_tokens = 0        # 新增的 prompt tokens（不含缓存）
        self.total_cached_tokens = 0        # 缓存命中的 tokens
        self.total_completion_tokens = 0    # 生成的 tokens
        self.total_requests = 0
        self.failed_requests = 0

        print(f"✅ [API Engine] 初始化完成")
        print(f"   Provider: {api_provider}")
        print(f"   Model: {model_name}")
        print(f"   Max Concurrent Turns: {max_concurrent_turns}")
        print(f"   Max Turns per Sample: {max_turns_per_sample}")
        print(f"   QPS Limit: {qps}")
        if qpm:
            print(f"   QPM Limit: {qpm}")
        print(f"   ⚠️  上下文管理: 由 API 自动处理（无手动截断）")

    async def call_api_with_retry(
        self,
        messages: List[Dict],
        state_index: int
    ) -> Optional[str]:
        """
        调用 API（带指数退避重试）

        Returns:
            生成的 content，失败返回 None
        """
        for attempt in range(self.max_retries):
            try:
                # 1. 获取速率限制令牌
                await self.qps_limiter.acquire()
                if self.qpm_limiter:
                    await self.qpm_limiter.acquire()

                # 2. 调用 API
                self.total_requests += 1
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                )

                # 3. 提取内容
                content = response.choices[0].message.content

                # 4. 成本跟踪（支持缓存统计）
                if self.enable_cost_tracking and hasattr(response, 'usage'):
                    usage = response.usage

                    # 提取缓存 tokens（兼容 OpenAI/Azure）
                    cached_tokens = 0
                    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

                    # 计算实际新增的 prompt tokens（排除缓存）
                    prompt_tokens = usage.prompt_tokens
                    new_prompt_tokens = prompt_tokens - cached_tokens

                    self.total_prompt_tokens += new_prompt_tokens
                    self.total_cached_tokens += cached_tokens
                    self.total_completion_tokens += usage.completion_tokens

                return content

            except Exception as e:
                error_type = type(e).__name__

                # 判断是否需要重试
                if attempt < self.max_retries - 1:
                    # 指数退避
                    delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"[{state_index}] {error_type}: {e}, retry {attempt+1}/{self.max_retries} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    # 最终失败
                    print(f"[{state_index}] ❌ API failed after {self.max_retries} retries: {e}")
                    self.failed_requests += 1
                    return None

    async def run_single_turn(self, state: SampleState) -> SampleState:
        """
        运行单个 turn

        核心逻辑：
        1. 调用 API（带速率限制和重试）
        2. 检查答案/工具调用
        3. 更新状态
        4. 检查轮数限制
        """
        async with self.semaphore:
            try:
                # 1. 调用 API（带速率限制和重试）
                content = await self.call_api_with_retry(state.messages, state.index)

                if content is None or not content.strip():
                    # API 调用失败或返回空响应
                    if content is None:
                        state.status = "api_error"
                    else:
                        print(f"[{state.index}] ⚠️  Empty response from API")
                        state.status = "empty_response"
                    return state

                # 3. 添加响应到消息列表
                state.full_response_log += content
                assistant_msg = {"role": "assistant", "content": content}
                state.messages.append(assistant_msg)
                state.full_messages.append(assistant_msg)

                # 4. 检查是否有答案
                answer_match = state.agent.extract_answer(content)
                if answer_match:
                    state.final_answer = answer_match
                    state.status = "finished"
                    return state

                # 5. 检查工具调用
                tool_call_match = state.agent.extract_action(content)
                if tool_call_match:
                    # 执行工具（World Model）
                    obs, reward, done, info = await state.agent.execute_async(action_string=content)

                    # 添加观察到消息列表
                    user_msg = {"role": "user", "content": obs}
                    state.messages.append(user_msg)
                    state.full_messages.append(user_msg)

                    if done:
                        state.status = "finished"
                        return state

                    # ✅ 修复：工具调用后也要检查轮数限制
                    actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                    if actual_turns >= state.max_turns:
                        print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}) after tool call")
                        state.status = "max_turns_reached"
                        return state
                else:
                    # 6. 无工具调用也无答案，检查轮数限制
                    actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])

                    if actual_turns >= state.max_turns:
                        # 检查历史中是否有答案
                        import re
                        for msg in state.full_messages:
                            if msg['role'] == 'assistant':
                                answers = re.findall(r'<answer>(.*?)</answer>', msg.get('content', ''), re.DOTALL)
                                if answers:
                                    state.final_answer = answers[-1].strip()
                                    state.status = "finished"
                                    return state

                        # 真的没有答案
                        state.status = "max_turns_reached"
                        return state

                    # 未达到限制，提醒继续
                    user_prompt_msg = {"role": "user", "content": state.agent.user_prompt}
                    state.messages.append(user_prompt_msg)
                    state.full_messages.append(user_prompt_msg)

                # 7. 更新轮数，继续
                actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                state.turn = actual_turns
                state.status = "running"
                return state

            except Exception as e:
                print(f"[{state.index}] Unexpected error: {e}")
                state.status = "api_error"
                return state

    def initialize_state(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        """初始化样本状态"""
        try:
            raw_messages = data.get('prompt', []) or data.get('messages', [])
            ground_truth = data.get('answer', '')
            extra_info = data.get('extra_info', {})

            # 创建 Agent
            agent = AgentEval(save_full_history=False)  # API 模式不保存 full_history
            allowed_tools = {'web_search'}
            agent.sub_tool_registry = {k: v for k, v in agent.sub_tool_registry.items() if k in allowed_tools}

            # 重置 Agent
            agent.reset(
                raw_prompt="",
                multi_modal_data=None,
                origin_multi_modal_data=None,
                world_truth=extra_info.get("world_truth_info", {})
            )

            # 准备消息
            chat_messages = raw_messages.copy()
            system_prompt = data.get('system', INSTRUCTION_PROMPT_SYSTEM)
            if not chat_messages or chat_messages[0]['role'] != 'system':
                chat_messages.insert(0, {"role": "system", "content": system_prompt})

            # 创建状态
            state = SampleState(
                index=line_idx,
                data=data,
                messages=chat_messages,
                full_messages=chat_messages.copy(),
                turn=0,
                max_turns=data.get('max_turns', self.max_turns_per_sample),
                status="running",
                agent=agent,
                ground_truth=ground_truth,
                extra_info=extra_info,
                trajectory_log={}
            )

            return state

        except Exception as e:
            print(f"[{line_idx}] 初始化失败: {e}")
            return None

    async def run_all_samples(self, samples_data: List[tuple]) -> List[Dict[str, Any]]:
        """
        运行所有样本（Turn 级别异步调度）

        核心逻辑：
        1. 初始化所有样本
        2. 创建初始 Turn 任务
        3. 事件循环：Turn 完成 → 创建下一个 Turn
        4. 实时保存进度
        """
        # 1. 初始化状态管理器
        state_manager = StateManager()
        for idx, data in samples_data:
            state = self.initialize_state(idx, data)
            if state:
                state_manager.add_state(state)

        total_samples = len(state_manager.states)
        print(f"\n✅ 初始化完成: {total_samples} 个样本")

        # 2. 创建初始 Turn 任务
        active_tasks = {}
        for state_idx in state_manager.states.keys():
            state = state_manager.states[state_idx]
            task = asyncio.create_task(self.run_single_turn(state))
            active_tasks[task] = state_idx

        # 3. 进度条
        pbar = async_tqdm(total=total_samples, desc="Samples")

        # 4. 事件循环
        while active_tasks:
            done, pending = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )

            for completed_task in done:
                state_idx = active_tasks.pop(completed_task)
                updated_state = completed_task.result()

                # 更新状态
                state_manager.states[state_idx] = updated_state

                # 更新轨迹日志
                updated_state.trajectory_log = updated_state.agent.get_trajectory_log()

                if updated_state.status == "running":
                    # 继续下一个 Turn
                    next_task = asyncio.create_task(self.run_single_turn(updated_state))
                    active_tasks[next_task] = state_idx
                else:
                    # 样本完成
                    state_manager.mark_completed(state_idx)
                    pbar.update(1)

        pbar.close()

        # 5. 打印统计
        print(f"\n✅ 推理完成！")
        print(f"   总样本: {total_samples}")
        print(f"   总请求: {self.total_requests}")
        print(f"   失败请求: {self.failed_requests}")

        if self.enable_cost_tracking:
            total_input_tokens = self.total_prompt_tokens + self.total_cached_tokens
            cache_hit_rate = (self.total_cached_tokens / total_input_tokens * 100) if total_input_tokens > 0 else 0

            print(f"\n📊 Token 使用统计:")
            print(f"   新增 Prompt Tokens: {self.total_prompt_tokens:,}")
            print(f"   缓存命中 Tokens: {self.total_cached_tokens:,}")
            print(f"   总 Input Tokens: {total_input_tokens:,}")
            print(f"   Completion Tokens: {self.total_completion_tokens:,}")
            print(f"   Total Tokens: {total_input_tokens + self.total_completion_tokens:,}")
            print(f"   缓存命中率: {cache_hit_rate:.1f}%")

        # 6. 导出结果
        return state_manager.export_results()


# ==================== 主程序 ====================

async def main():
    parser = argparse.ArgumentParser(description="MPW-bench API 异步推理")

    # 文件路径
    parser.add_argument('--input', '-i', type=str, required=True, help='输入 JSONL 文件')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出 JSONL 文件')

    # API 配置
    parser.add_argument('--api-provider', type=str, default='openai',
                       choices=['openai', 'azure', 'custom'],
                       help='API 提供商')
    parser.add_argument('--api-key', type=str, default=None, help='API Key')
    parser.add_argument('--base-url', type=str, default=None, help='Base URL (custom provider)')
    parser.add_argument('--model', type=str, default='gpt-4', help='模型名称')

    # 并发和速率
    parser.add_argument('--max-concurrent', type=int, default=100, help='最大并发 Turn 数')
    parser.add_argument('--qps', type=float, default=10, help='每秒请求数限制')
    parser.add_argument('--qpm', type=float, default=None, help='每分钟请求数限制（可选）')

    # Agent 配置
    parser.add_argument('--max-turns', type=int, default=32, help='每个样本最大轮数（统一配置）')

    # 其他
    parser.add_argument('--max-retries', type=int, default=5, help='API 最大重试次数')

    args = parser.parse_args()

    # 读取数据
    print(f"📂 读取数据: {args.input}")
    samples_data = []

    # 断点续传：读取已有结果
    processed_indices = set()
    if os.path.exists(args.output):
        print(f"📂 检测到已有结果，启用断点续传...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    if 'index' in res:
                        processed_indices.add(res['index'])
        print(f"   已完成: {len(processed_indices)} 个样本")

    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                if idx not in processed_indices:
                    data = json.loads(line)
                    samples_data.append((idx, data))

    if not samples_data:
        print("✅ 所有样本已完成！")
        return

    print(f"📊 待处理: {len(samples_data)} 个样本")

    # 创建引擎
    engine = AsyncInferenceEngineAPI(
        api_provider=args.api_provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_concurrent_turns=args.max_concurrent,
        qps=args.qps,
        qpm=args.qpm,
        max_retries=args.max_retries,
        max_turns_per_sample=args.max_turns,
    )

    # 运行推理
    results = await engine.run_all_samples(samples_data)

    # 保存结果
    print(f"💾 保存结果: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 追加模式（支持断点续传）
    with open(args.output, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print("✅ 完成！")


if __name__ == "__main__":
    asyncio.run(main())
