#!/usr/bin/env python3
"""
async_inference.py - Custom Prompt Evaluation (API Version)

完整 agent loop 版本，与 vllm/async_inference.py 功能对等：
  - Turn 级别异步调度
  - Token Bucket 速率限制
  - 指数退避重试
  - 断点续传
  - 成本跟踪
  - 可插拔 tool call parser（--parser），支持 default / deepseek / glm4 / minimax / qwen35 / kimi_k2
  - 自定义 System Prompt（--prompt-file）
  - 并行工具调用（--parallel-tool-calls，仅 --tool-mode text 下生效）

支持两种工具调用模式（--tool-mode）：

  text（默认）：
    通过 system prompt 约定格式，由 parser 从模型文本输出中提取工具调用。
    与 vLLM 版本保持一致，适合跨部署方式的对比评估。

  native：
    使用 API 原生 function calling（传入 tools 参数，从 tool_calls 字段读取结果）。
    由 API provider 完成解析，适合评估商业模型的真实工具调用能力上限。
    模型直接输出文本时即视为最终答案，无需 <answer> 标签。

使用方法：
    # 文本模式（与 vLLM 对比）
    python experiments/custom/api/async_inference.py \\
        --provider openai --model gpt-4o \\
        --input data.jsonl --output results.jsonl \\
        --prompt-file prompts/my_prompt.py \\
        --tool-mode text --parser default

    # 原生模式（商业 API 能力评估）
    python experiments/custom/api/async_inference.py \\
        --provider openai --model gpt-4o \\
        --input data.jsonl --output results.jsonl \\
        --prompt-file prompts/my_prompt.py \\
        --tool-mode native
"""

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from verl.sample_state import SampleState, StateManager
from verl.workers.agent.envs.agent_eval import AgentEval


# ================= Prompt 加载 =================

def load_prompt_from_file(prompt_file: str) -> str:
    """从 Python 文件加载 system_prompt 变量。"""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    spec = importlib.util.spec_from_file_location("custom_prompt", prompt_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "system_prompt"):
        raise ValueError(
            f"Prompt file must define a 'system_prompt' variable: {prompt_file}"
        )
    return module.system_prompt


# ================= 速率限制器 =================

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


# ================= API 客户端工厂 =================

class APIClientFactory:
    @staticmethod
    def create_client(
        provider: str,
        api_key: str = None,
        base_url: str = None,
    ) -> AsyncOpenAI:
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


# ================= 工具函数 =================

def build_tool_definitions(agent: AgentEval) -> List[Dict]:
    """
    从 agent.sub_tool_registry 构建 OpenAI 格式的 tool definitions。

    每个工具需具有 name、description、parameters 属性。
    """
    definitions = []
    for tool_name, tool in agent.sub_tool_registry.items():
        definitions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": getattr(tool, "description", f"Tool: {tool_name}"),
                "parameters": getattr(tool, "parameters", {
                    "type": "object",
                    "properties": {},
                }),
            },
        })
    return definitions


def strip_chattml_obs(obs: str) -> str:
    """
    从 ChatML 格式的工具观察中提取原始内容。

    WorldModelWebSearchTool 返回的 obs 格式为：
      \\n<|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n
    """
    match = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', obs, re.DOTALL)
    if match:
        return match.group(1).strip()
    return obs.strip()


# ================= 推理引擎 =================

class AsyncInferenceEngine:
    """
    API 版本推理引擎，与 vllm/async_inference.py 功能对等。

    核心设计：
      1. Turn 级别异步调度（FIRST_COMPLETED 事件循环）
      2. Token Bucket 全局速率限制
      3. 指数退避重试
      4. 断点续传
      5. 成本跟踪
      6. 双工具模式：text（文本解析）/ native（原生 function calling）
    """

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: str = None,
        base_url: str = None,
        model_name: str = "gpt-4o",
        system_prompt: str = "",
        tool_call_parser: str = "default",
        tool_mode: str = "text",
        max_concurrent_turns: int = 100,
        qps: float = 10.0,
        qpm: float = None,
        max_retries: int = 5,
        initial_retry_delay: float = 1.0,
        max_turns_per_sample: int = 32,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        enable_cost_tracking: bool = True,
        parallel_tool_calls: bool = False,
    ):
        if tool_mode not in ("text", "native"):
            raise ValueError(f"tool_mode must be 'text' or 'native', got '{tool_mode}'")

        self.client = APIClientFactory.create_client(api_provider, api_key, base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tool_call_parser = tool_call_parser
        self.tool_mode = tool_mode
        self.parallel_tool_calls = parallel_tool_calls

        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.qps_limiter = TokenBucket(rate=qps, capacity=int(qps * 2), name="QPS")
        self.qpm_limiter = (
            TokenBucket(rate=qpm / 60, capacity=int(qpm), name="QPM") if qpm else None
        )

        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_turns_per_sample = max_turns_per_sample
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.enable_cost_tracking = enable_cost_tracking
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.failed_requests = 0

        print(f"[Engine] Initialized")
        print(f"  Provider   : {api_provider}")
        print(f"  Model      : {model_name}")
        print(f"  Tool Mode  : {tool_mode}")
        if tool_mode == "text":
            print(f"  Parser     : {tool_call_parser}")
            print(f"  Parallel   : {parallel_tool_calls}")
        print(f"  Prompt     : {len(system_prompt)} chars")
        print(f"  QPS        : {qps}")
        print(f"  Max Turns  : {max_turns_per_sample}")

    # ── 通用：成本跟踪 ────────────────────────────────────────────────────────

    def _track_usage(self, usage):
        if not self.enable_cost_tracking or usage is None:
            return
        cached = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = usage.prompt_tokens_details.cached_tokens or 0
        self.total_prompt_tokens += usage.prompt_tokens - cached
        self.total_cached_tokens += cached
        self.total_completion_tokens += usage.completion_tokens

    # ── 文本模式：API 调用 ────────────────────────────────────────────────────

    async def _call_text(
        self, messages: List[Dict], state_index: int
    ) -> Optional[str]:
        """文本模式 API 调用，返回 content 字符串，失败返回 None。"""
        for attempt in range(self.max_retries):
            try:
                await self.qps_limiter.acquire()
                if self.qpm_limiter:
                    await self.qpm_limiter.acquire()

                self.total_requests += 1
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self._track_usage(getattr(response, "usage", None))
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

    # ── 原生模式：API 调用 ────────────────────────────────────────────────────

    async def _call_native(
        self, messages: List[Dict], tool_defs: List[Dict], state_index: int
    ) -> Optional[Any]:
        """
        原生模式 API 调用，传入 tools 参数，返回 message 对象，失败返回 None。

        调用方通过 message.content 和 message.tool_calls 分别读取文本和工具调用。
        """
        for attempt in range(self.max_retries):
            try:
                await self.qps_limiter.acquire()
                if self.qpm_limiter:
                    await self.qpm_limiter.acquire()

                self.total_requests += 1
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                self._track_usage(getattr(response, "usage", None))
                return response.choices[0].message

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.initial_retry_delay * (2 ** attempt)
                    print(f"[{state_index}] {type(e).__name__}: {e}, retry {attempt+1}/{self.max_retries} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    print(f"[{state_index}] API failed after {self.max_retries} retries: {e}")
                    self.failed_requests += 1
                    return None

    # ── 原生模式：单个工具执行 ────────────────────────────────────────────────

    async def _execute_native_tool(
        self, agent: AgentEval, tool_name: str, tool_args: Dict
    ) -> tuple:
        """
        直接调用 sub_tool 并维护 agent 轨迹日志，绕过文本提取步骤。

        Returns:
            (raw_content, reward, done, info)
            raw_content: 已剥离 ChatML 包装的纯文本观察
        """
        tool = agent.sub_tool_registry.get(tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_name}'", 0.0, False, {}

        action_json = json.dumps({"name": tool_name, "arguments": tool_args})
        obs, reward, done, info = await tool.execute_async(
            action_string=action_json,
            agent_trajectory=agent.chatml_history,
            world_truth=agent.world_truth,
        )

        # 同步轨迹日志（与 AgentEval.execute_async 保持一致）
        if "hit_log" in info:
            agent.trajectory_log["hit_logs"].append(info["hit_log"])
        agent.trajectory_log["tool_calls"].append({
            "tool_name": tool_name,
            "query": tool_args.get("query", ""),
            "hit": info.get("hit_log", {}).get("hit", 0),
        })
        if not done:
            agent.chatml_history += action_json + obs

        raw_content = strip_chattml_obs(obs)
        return raw_content, reward, done, info

    # ── 状态初始化 ────────────────────────────────────────────────────────────

    def initialize_state(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        try:
            raw_messages = data.get("prompt", []) or data.get("messages", [])
            ground_truth = data.get("answer", "")
            extra_info = data.get("extra_info", {})
            world_truth = extra_info.get("world_truth_info", {})

            agent = AgentEval(
                save_full_history=False,
                parser=self.tool_call_parser,
            )
            agent.reset(
                raw_prompt="",
                multi_modal_data=None,
                origin_multi_modal_data=None,
                world_truth=world_truth,
            )

            chat_messages = raw_messages.copy()
            system_prompt = data.get("system", self.system_prompt)
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

    # ── 文本模式：单轮执行 ────────────────────────────────────────────────────

    async def _run_turn_text(self, state: SampleState) -> SampleState:
        content = await self._call_text(state.messages, state.index)

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
            if self.parallel_tool_calls:
                obs, _reward, done, _info = await state.agent.execute_parallel_async(
                    action_string=content
                )
            else:
                obs, _reward, done, _info = await state.agent.execute_async(
                    action_string=content
                )
            user_msg = {"role": "user", "content": strip_chattml_obs(obs)}
            state.messages.append(user_msg)
            state.full_messages.append(user_msg)
            if done:
                state.status = "finished"
                return state
        else:
            # 无工具调用也无答案：提示继续
            user_msg = {"role": "user", "content": state.agent.user_prompt}
            state.messages.append(user_msg)
            state.full_messages.append(user_msg)

        assistant_turns = sum(1 for m in state.full_messages if m["role"] == "assistant")
        state.turn = assistant_turns
        state.status = "max_turns_reached" if assistant_turns >= state.max_turns else "running"
        return state

    # ── 原生模式：单轮执行 ────────────────────────────────────────────────────

    async def _run_turn_native(self, state: SampleState) -> SampleState:
        tool_defs = build_tool_definitions(state.agent)
        message = await self._call_native(state.messages, tool_defs, state.index)

        if message is None:
            state.status = "api_error"
            return state

        text_content = message.content or ""
        tool_calls = message.tool_calls  # None 或 list

        if tool_calls:
            # 构建带 tool_calls 字段的 assistant 消息
            assistant_msg = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            state.messages.append(assistant_msg)
            state.full_messages.append(assistant_msg)
            state.full_response_log += text_content

            # 逐个执行工具调用
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                raw_content, _reward, done, _info = await self._execute_native_tool(
                    state.agent, tool_name, tool_args
                )

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": raw_content,
                }
                state.messages.append(tool_msg)
                state.full_messages.append(tool_msg)

                if done:
                    state.status = "finished"
                    return state

            assistant_turns = sum(1 for m in state.full_messages if m["role"] == "assistant")
            state.turn = assistant_turns
            state.status = "max_turns_reached" if assistant_turns >= state.max_turns else "running"

        else:
            # 无工具调用：模型完成输出，文本内容即为最终答案
            if not text_content.strip():
                state.status = "empty_response"
                return state

            state.full_response_log += text_content
            assistant_msg = {"role": "assistant", "content": text_content}
            state.messages.append(assistant_msg)
            state.full_messages.append(assistant_msg)

            # 兼容：先检查 <answer> 标签，没有则直接取全文
            answer = state.agent.extract_answer(text_content)
            state.final_answer = answer if answer else text_content.strip()
            state.status = "finished"

        return state

    # ── 统一入口 ──────────────────────────────────────────────────────────────

    async def run_single_turn(self, state: SampleState) -> SampleState:
        async with self.semaphore:
            try:
                if self.tool_mode == "native":
                    return await self._run_turn_native(state)
                else:
                    return await self._run_turn_text(state)
            except Exception as e:
                print(f"[{state.index}] Unexpected error: {e}")
                state.status = "api_error"
                return state

    # ── 全量运行 ──────────────────────────────────────────────────────────────

    async def run_all_samples(self, samples_data: List[tuple]) -> List[Dict[str, Any]]:
        """Turn 级别异步调度，所有样本并发推进。"""
        state_manager = StateManager()
        for idx, data in samples_data:
            state = self.initialize_state(idx, data)
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

    async def run_inference(
        self, input_file: str, output_file: str, resume: bool = False
    ):
        """入口：读取 JSONL，运行推理，追加写出结果。"""
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
                        data = json.loads(line)
                        samples_data.append((idx, data))
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


# ================= 主函数 =================

async def main():
    parser = argparse.ArgumentParser(
        description="Custom Prompt Evaluation (API Version)"
    )

    # 必需参数
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "azure", "custom"],
                        help="API 提供商")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件")
    parser.add_argument("--prompt-file", type=str, required=True,
                        help="System prompt 文件（.py），须定义 system_prompt 变量")

    # API 配置
    parser.add_argument("--api-key", type=str, default=None, help="API Key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL（custom provider 必填）")

    # 工具调用模式
    parser.add_argument("--tool-mode", type=str, default="text",
                        choices=["text", "native"],
                        help=(
                            "工具调用模式: "
                            "text=文本解析（与 vLLM 对齐，默认）, "
                            "native=API 原生 function calling（商业 API 推荐）"
                        ))

    # 文本模式专用：parser
    parser.add_argument("--parser", type=str, default="default",
                        choices=["default", "deepseek", "glm4",
                                 "minimax", "qwen35", "kimi_k2"],
                        help="Tool call 文本解析器，仅 --tool-mode text 时生效（默认: default）")
    parser.add_argument("--parallel-tool-calls", action="store_true",
                        help="在文本模式下，并行执行同一轮中的多个 tool_call，结果合并为单条消息"
                             "（仅 --tool-mode text 且 --parser default 时生效）")

    # 性能参数
    parser.add_argument("--qps", type=float, default=10.0, help="每秒请求数限制")
    parser.add_argument("--qpm", type=float, default=None, help="每分钟请求数限制（可选）")
    parser.add_argument("--max-concurrent-turns", type=int, default=100,
                        help="最大并发 Turn 数")
    parser.add_argument("--max-turns-per-sample", type=int, default=32,
                        help="每个样本最大轮次")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="每次生成最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")

    # 其他
    parser.add_argument("--max-retries", type=int, default=5, help="API 最大重试次数")
    parser.add_argument("--resume", action="store_true", help="断点续传")

    args = parser.parse_args()

    print(f"加载 Prompt: {args.prompt_file}")
    system_prompt = load_prompt_from_file(args.prompt_file)
    print(f"Prompt 加载完成: {len(system_prompt)} 字符")

    engine = AsyncInferenceEngine(
        api_provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        system_prompt=system_prompt,
        tool_call_parser=args.parser,
        tool_mode=args.tool_mode,
        max_concurrent_turns=args.max_concurrent_turns,
        qps=args.qps,
        qpm=args.qpm,
        max_retries=args.max_retries,
        max_turns_per_sample=args.max_turns_per_sample,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        parallel_tool_calls=args.parallel_tool_calls,
    )

    await engine.run_inference(args.input, args.output, resume=args.resume)
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
