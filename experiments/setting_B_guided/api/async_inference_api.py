#!/usr/bin/env python3
"""
async_inference_api.py - Setting B Guided Search API 推理（基础版）

核心特性：
1. Turn 级别异步调度（FIRST_COMPLETED 事件循环）
2. Token Bucket 速率限制（QPS / QPM）
3. 指数退避重试
4. 断点续传
5. 成本跟踪

与 Setting C 的区别：
- 默认使用 guidance_prompt（原子化查询指南）
- 支持通过 --world-model-vllm-url 配置世界模型节点
- initialize_sample() 可被子类重写以替换 system prompt
  （async_inference_api_fewshot.py 即通过此机制切换为 fewshot_prompt）

使用方法：
    python experiments/setting_B_guided/api/async_inference_api.py \\
        --provider openai \\
        --model gpt-4o \\
        --input data/mpw_bench_full.jsonl \\
        --output results/setting_B/api_results.jsonl \\
        --qps 10 \\
        --max-concurrent-turns 100 \\
        --world-model-vllm-url http://localhost:8000/v1
"""

import asyncio
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

# 确保项目根目录在 sys.path 中，支持从任意工作目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from verl.sample_state import SampleState, StateManager
from verl.workers.agent.envs.agent_eval import AgentEval
from verl.workers.agent.envs.tools.world_model_web_search_tool import WorldModelWebSearchTool
from experiments.setting_B_guided.prompt import guidance_prompt

# 默认 system prompt：原子化查询指南（guidance_prompt）
# fewshot 变体通过重写 initialize_sample() 切换为 fewshot_prompt
INSTRUCTION_PROMPT_SYSTEM = guidance_prompt


# ==================== Token Bucket 速率限制器 ====================

class TokenBucket:
    """令牌桶算法实现 QPS/QPM 速率限制。"""

    def __init__(self, rate: float, capacity: int, name: str = "limiter"):
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.total_acquired = 0
        self.total_wait_time = 0.0

    async def acquire(self, tokens: int = 1):
        async with self.lock:
            wait_start = time.time()
            while True:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.total_acquired += tokens
                    self.total_wait_time += time.time() - wait_start
                    return
                await asyncio.sleep((tokens - self.tokens) / self.rate)


# ==================== API 客户端工厂 ====================

class APIClientFactory:
    @staticmethod
    def create_client(provider: str, api_key: str = None, base_url: str = None) -> AsyncOpenAI:
        if provider == "openai":
            return AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "azure":
            return AsyncOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=base_url or os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        elif provider == "custom":
            if not base_url:
                raise ValueError("Custom provider requires --base-url")
            return AsyncOpenAI(api_key=api_key or "dummy", base_url=base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# ==================== 异步推理引擎 ====================

class AsyncInferenceEngine:
    """
    Setting B API 推理引擎（基础版）。

    fewshot 变体（AsyncInferenceEngine_FewShot）通过重写
    initialize_sample() 来替换 system prompt，其余逻辑完全复用本类。
    """

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: str = None,
        base_url: str = None,
        model_name: str = "gpt-4",
        max_concurrent_turns: int = 100,
        qps: float = 10.0,
        qpm: float = None,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        max_turns_per_sample: int = 32,
        enable_cost_tracking: bool = True,
        # 世界模型配置
        world_model_provider: str = "vllm",
        world_model_name: str = None,
        world_model_vllm_url: str = None,
    ):
        self.client = APIClientFactory.create_client(api_provider, api_key, base_url)
        self.model_name = model_name

        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.qps_limiter = TokenBucket(rate=qps, capacity=int(qps * 2), name="QPS")
        self.qpm_limiter = (
            TokenBucket(rate=qpm / 60, capacity=int(qpm), name="QPM") if qpm else None
        )

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_turns_per_sample = max_turns_per_sample

        # 世界模型
        self.world_model_provider = world_model_provider
        self.world_model_name = world_model_name
        self.world_model_vllm_url = world_model_vllm_url

        # 成本跟踪
        self.enable_cost_tracking = enable_cost_tracking
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

        print(f"[API Engine] Initialized")
        print(f"  Provider   : {api_provider}")
        print(f"  Model      : {model_name}")
        print(f"  QPS        : {qps}")
        print(f"  Max Turns  : {max_turns_per_sample}")
        print(f"  World Model: {world_model_vllm_url or 'default (env/localhost)'}")

    # ── 世界模型 Agent 工厂 ──────────────────────────────────────────────────

    def choose_world_model_agent(
        self, world_truth_info: Dict, category: str = "default"
    ) -> AgentEval:
        """
        创建并配置 AgentEval 实例。

        世界模型节点优先使用 --world-model-vllm-url，
        未指定时回退到环境变量 WORLD_MODEL_ENDPOINTS 或 localhost:8000。
        """
        web_search_tool = WorldModelWebSearchTool(endpoints=self.world_model_vllm_url)
        agent = AgentEval(save_full_history=False, tools=[rag_engine])
        agent.reset(
            raw_prompt="",
            multi_modal_data=None,
            origin_multi_modal_data=None,
            world_truth=world_truth_info,
        )
        return agent

    # ── 样本初始化（可被子类重写以替换 prompt）───────────────────────────────

    def initialize_sample(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        """
        初始化单个样本状态。

        子类（如 AsyncInferenceEngine_FewShot）可重写此方法以使用不同 system prompt，
        其余字段构造逻辑保持不变。
        """
        try:
            raw_messages = data.get("prompt", []) or data.get("messages", [])
            ground_truth = data.get("answer", "")
            extra_info = data.get("extra_info", {})
            world_truth_info = extra_info.get("world_truth_info", {})
            category = extra_info.get("category", "default")

            agent = self.choose_world_model_agent(world_truth_info, category)

            chat_messages = raw_messages.copy()
            system_prompt = data.get("system", INSTRUCTION_PROMPT_SYSTEM)
            if not chat_messages or chat_messages[0]["role"] != "system":
                chat_messages.insert(0, {"role": "system", "content": system_prompt})

            return SampleState(
                index=line_idx,
                data=data,
                messages=chat_messages,
                full_messages=chat_messages.copy(),
                turn=0,
                max_turns=data.get("max_turns", self.max_turns_per_sample),
                status="running",
                agent=agent,
                ground_truth=ground_truth,
                extra_info=extra_info,
                trajectory_log={},
            )

        except Exception as e:
            print(f"[{line_idx}] 初始化失败: {e}")
            return None

    # ── API 调用（带重试）────────────────────────────────────────────────────

    async def call_api_with_retry(
        self, messages: List[Dict], state_index: int
    ) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                await self.qps_limiter.acquire()
                if self.qpm_limiter:
                    await self.qpm_limiter.acquire()

                self.total_requests += 1
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192,
                )

                if self.enable_cost_tracking and hasattr(response, "usage"):
                    usage = response.usage
                    cached = 0
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        cached = usage.prompt_tokens_details.cached_tokens or 0
                    self.total_prompt_tokens += usage.prompt_tokens - cached
                    self.total_cached_tokens += cached
                    self.total_completion_tokens += usage.completion_tokens

                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"[{state_index}] {type(e).__name__}: {e}, retry {attempt+1}/{self.max_retries} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"[{state_index}] API failed after {self.max_retries} retries: {e}")
                    self.failed_requests += 1
                    return None

    # ── 单轮执行（agent loop 核心）────────────────────────────────────────────

    async def run_single_turn(self, state: SampleState) -> SampleState:
        async with self.semaphore:
            try:
                content = await self.call_api_with_retry(state.messages, state.index)

                if content is None:
                    state.status = "api_error"
                    return state
                if not content.strip():
                    state.status = "empty_response"
                    return state

                state.full_response_log += content
                assistant_msg = {"role": "assistant", "content": content}
                state.messages.append(assistant_msg)
                state.full_messages.append(assistant_msg)

                # 检查最终答案
                answer = state.agent.extract_answer(content)
                if answer:
                    state.final_answer = answer
                    state.status = "finished"
                    return state

                # 检查工具调用
                action = state.agent.extract_action(content)
                if action:
                    obs, _reward, done, _info = await state.agent.execute_async(
                        action_string=content
                    )
                    user_msg = {"role": "user", "content": obs}
                    state.messages.append(user_msg)
                    state.full_messages.append(user_msg)

                    if done:
                        state.status = "finished"
                        return state

                    actual_turns = sum(
                        1 for m in state.full_messages if m["role"] == "assistant"
                    )
                    if actual_turns >= state.max_turns:
                        print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}) after tool call")
                        state.status = "max_turns_reached"
                        return state
                else:
                    # 无工具调用也无答案
                    actual_turns = sum(
                        1 for m in state.full_messages if m["role"] == "assistant"
                    )
                    if actual_turns >= state.max_turns:
                        # 历史中回溯最后一个答案
                        for msg in reversed(state.full_messages):
                            if msg["role"] == "assistant":
                                answers = re.findall(
                                    r"<answer>(.*?)</answer>",
                                    msg.get("content", ""),
                                    re.DOTALL,
                                )
                                if answers:
                                    state.final_answer = answers[-1].strip()
                                    state.status = "finished"
                                    return state
                        state.status = "max_turns_reached"
                        return state

                    user_msg = {"role": "user", "content": state.agent.user_prompt}
                    state.messages.append(user_msg)
                    state.full_messages.append(user_msg)

                actual_turns = sum(
                    1 for m in state.full_messages if m["role"] == "assistant"
                )
                state.turn = actual_turns
                state.status = "running"
                return state

            except Exception as e:
                print(f"[{state.index}] Unexpected error: {e}")
                state.status = "api_error"
                return state

    # ── 全量运行（Turn 级别异步调度）────────────────────────────────────────

    async def run_all_samples(self, samples_data: List[tuple]) -> List[Dict[str, Any]]:
        state_manager = StateManager()
        for idx, data in samples_data:
            state = self.initialize_sample(idx, data)
            if state:
                state_manager.add_state(state)

        total = len(state_manager.states)
        print(f"\n初始化完成: {total} 个样本")

        active_tasks: Dict[asyncio.Task, int] = {}
        for sid in state_manager.states:
            task = asyncio.create_task(self.run_single_turn(state_manager.states[sid]))
            active_tasks[task] = sid

        pbar = async_tqdm(total=total, desc="Samples")

        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for completed in done:
                sid = active_tasks.pop(completed)
                updated = completed.result()
                state_manager.states[sid] = updated
                updated.trajectory_log = updated.agent.get_trajectory_log()

                if updated.status == "running":
                    next_task = asyncio.create_task(self.run_single_turn(updated))
                    active_tasks[next_task] = sid
                else:
                    state_manager.mark_completed(sid)
                    pbar.update(1)

        pbar.close()

        print(f"\n推理完成")
        print(f"  总样本  : {total}")
        print(f"  总请求  : {self.total_requests}")
        print(f"  失败请求: {self.failed_requests}")

        if self.enable_cost_tracking:
            total_input = self.total_prompt_tokens + self.total_cached_tokens
            cache_rate = (
                self.total_cached_tokens / total_input * 100 if total_input > 0 else 0
            )
            print(f"\nToken 统计:")
            print(f"  新增 Prompt  : {self.total_prompt_tokens:,}")
            print(f"  缓存命中     : {self.total_cached_tokens:,} ({cache_rate:.1f}%)")
            print(f"  Completion   : {self.total_completion_tokens:,}")
            print(f"  Total        : {total_input + self.total_completion_tokens:,}")

        return state_manager.export_results()

    # ── 推理入口（读取 JSONL → 运行 → 追加写出）────────────────────────────

    async def run_inference(
        self, input_file: str, output_file: str, resume: bool = False
    ):
        processed = set()
        if resume and os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if "index" in r:
                            processed.add(r["index"])
                    except Exception:
                        pass
            print(f"断点续传：已跳过 {len(processed)} 个样本")

        samples_data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip() and idx not in processed:
                    try:
                        samples_data.append((idx, json.loads(line)))
                    except Exception:
                        continue

        if not samples_data:
            print("无待处理样本，退出。")
            return

        print(f"待处理: {len(samples_data)} 个样本")
        results = await self.run_all_samples(samples_data)

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"结果已保存: {output_file}")


# ==================== 主程序 ====================

async def main():
    parser = argparse.ArgumentParser(
        description="Setting B Guided Search API 推理（guidance_prompt 版）"
    )

    # 文件路径
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件")

    # API 配置
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "azure", "custom"], help="API 提供商")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--api-key", type=str, default=None, help="API Key（可选）")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL（custom provider 必填）")

    # 并发和速率
    parser.add_argument("--qps", type=float, default=10.0, help="每秒请求数限制")
    parser.add_argument("--qpm", type=float, default=None, help="每分钟请求数限制（可选）")
    parser.add_argument("--max-concurrent-turns", type=int, default=100,
                        help="最大并发 Turn 数")
    parser.add_argument("--max-turns-per-sample", type=int, default=32,
                        help="每个样本最大轮次")

    # 世界模型配置
    parser.add_argument("--world-model-provider", type=str, default="vllm",
                        choices=["openai", "azure", "custom", "vllm"],
                        help="世界模型 API 提供商（默认: vllm）")
    parser.add_argument("--world-model-name", type=str, default=None,
                        help="世界模型名称（vllm 时自动探测）")
    parser.add_argument("--world-model-vllm-url", type=str, default=None,
                        help="世界模型 vLLM 节点地址，多节点用逗号分隔")

    # 其他
    parser.add_argument("--max-retries", type=int, default=5, help="API 最大重试次数")
    parser.add_argument("--resume", action="store_true", help="断点续传")

    args = parser.parse_args()

    engine = AsyncInferenceEngine(
        api_provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        max_concurrent_turns=args.max_concurrent_turns,
        qps=args.qps,
        qpm=args.qpm,
        max_retries=args.max_retries,
        max_turns_per_sample=args.max_turns_per_sample,
        world_model_provider=args.world_model_provider,
        world_model_name=args.world_model_name,
        world_model_vllm_url=args.world_model_vllm_url,
    )

    await engine.run_inference(args.input, args.output, resume=args.resume)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
