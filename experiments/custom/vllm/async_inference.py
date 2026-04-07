#!/usr/bin/env python3
"""
async_inference.py - Custom Prompt Evaluation (vLLM Version)

核心特点：
1. 支持用户自定义 System Prompt（通过 --prompt-file 参数）
2. Turn 级别的细粒度异步
3. 可插拔 tool call parser（--parser），支持 default / deepseek / glm4 / minimax / qwen35 / kimi_k2
4. 并行工具调用（--parallel-tool-calls）：单轮多个 tool_call 并行执行后合并为一条消息

使用方法：
    python async_inference.py \
        --input data.jsonl \
        --output results.jsonl \
        --prompt-file my_prompt.py \
        --model models \
        --api-base http://localhost:8000/v1 \
        --parser default \
        --parallel-tool-calls
"""

import argparse
import asyncio
import json
import os
import sys
import base64
import re
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import httpx

# 确保项目根目录在 sys.path 中，支持从任意工作目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入状态管理
from verl.sample_state import SampleState, StateManager

# 导入评测专用 Agent
from verl.workers.agent.envs.agent_eval import AgentEval

# ================= 全局配置 =================

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


def load_prompt_from_file(prompt_file: str) -> str:
    """
    从 Python 文件加载 system_prompt 变量

    Args:
        prompt_file: Python 文件路径，必须定义 system_prompt 变量

    Returns:
        system_prompt 字符串
    """
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("custom_prompt", prompt_file)
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)

    # 检查是否定义了 system_prompt
    if not hasattr(prompt_module, 'system_prompt'):
        raise ValueError(f"Prompt file must define a 'system_prompt' variable: {prompt_file}")

    return prompt_module.system_prompt


# ================= 核心异步推理类 =================

class AsyncInferenceEngine:
    """
    异步推理引擎 - 支持自定义 System Prompt
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        system_prompt: str,
        max_concurrent_turns: int = 32,
        max_turns_per_sample: int = 8,
        max_context_chars: int = 60000,
        thinking_mode: Optional[bool] = None,
        max_retries: int = 5,
        tool_call_parser: str = "default",
        parallel_tool_calls: bool = False,
    ):
        """
        初始化推理引擎

        Args:
            model_name: 模型名称
            api_base: API 地址
            api_key: API 密钥
            system_prompt: 自定义 System Prompt
            max_concurrent_turns: Turn 级别的最大并发数
            max_turns_per_sample: 每个样本的最大轮数
            max_context_chars: 上下文最大字符数
            thinking_mode: 控制模型内置 thinking 模式
            max_retries: API 调用失败时的最大重试次数
            tool_call_parser: tool call 解析器名称，内置选项："default"、"deepseek"、"glm4"
            parallel_tool_calls: 是否启用并行工具调用（一次输出多个 tool_call 时同时执行）
        """
        timeout_config = httpx.Timeout(connect=100.0, read=600.0, write=100.0, pool=100.0)
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_config)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.max_turns_per_sample = max_turns_per_sample
        self.max_context_chars = max_context_chars
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.tool_call_parser = tool_call_parser
        self.parallel_tool_calls = parallel_tool_calls

        # 检测是否为 GLM-4 系列模型
        self.is_glm4 = 'glm' in model_name.lower() or 'GLM' in model_name

        print(f"[Engine] Initialized:")
        print(f"   - Model: {model_name}")
        print(f"   - System Prompt: {len(system_prompt)} chars")
        print(f"   - MaxConcurrentTurns: {max_concurrent_turns}")
        print(f"   - MaxTurnsPerSample: {max_turns_per_sample}")
        print(f"   - ThinkingMode: {thinking_mode}")
        print(f"   - Max Retries: {max_retries}")
        print(f"   - Parallel Tool Calls: {parallel_tool_calls}")
        if self.is_glm4:
            print(f"   - GLM-4 Compatible: Enabled")

    def _load_pil_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载 PIL 图片"""
        pil_images = []
        for img_src in image_paths:
            try:
                if os.path.exists(img_src):
                    img = Image.open(img_src).convert("RGB")
                else:
                    if "," in img_src:
                        img_src = img_src.split(",")[1]
                    img_data = base64.b64decode(img_src)
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                pil_images.append(img)
            except Exception as e:
                print(f"[Image Load Error] {e}")
        return pil_images

    def _encode_image_to_base64(self, image_source: str) -> str:
        """编码图片为 base64"""
        try:
            if os.path.exists(image_source):
                with open(image_source, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                if "," in image_source:
                    return image_source.split(",")[1]
                return image_source
        except Exception:
            return ""

    def _clean_observation(self, obs: str) -> str:
        """清洗 observation"""
        match = re.search(r'<\|im_start\|>observation(.*?)<\|im_end\|>', obs, re.DOTALL)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r'<\|im_start\|>\w+|<\|im_end\|>', '', obs)
        return cleaned.strip()

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """滑动窗口截断消息"""
        if not messages:
            return messages

        system_msg = None
        conversation = messages

        if messages[0].get('role') == 'system':
            system_msg = messages[0]
            conversation = messages[1:]

        def count_message_chars(msg: Dict) -> int:
            content = msg.get('content', '')
            if isinstance(content, str):
                return len(content)
            elif isinstance(content, list):
                total = 0
                for item in content:
                    if item.get('type') == 'text':
                        total += len(item.get('text', ''))
                    elif item.get('type') == 'image_url':
                        total += 1000
                return total
            return 0

        system_chars = count_message_chars(system_msg) if system_msg else 0
        conversation_chars = sum(count_message_chars(msg) for msg in conversation)
        total_chars = system_chars + conversation_chars

        if total_chars <= self.max_context_chars:
            return messages

        print(f"[Context Truncate] Total chars {total_chars} exceeds limit {self.max_context_chars}, truncating...")

        truncated_conversation = []
        accumulated_chars = system_chars

        for msg in reversed(conversation):
            msg_chars = count_message_chars(msg)
            if accumulated_chars + msg_chars <= self.max_context_chars:
                truncated_conversation.insert(0, msg)
                accumulated_chars += msg_chars
            else:
                break

        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(truncated_conversation)

        removed_count = len(conversation) - len(truncated_conversation)
        if removed_count > 0:
            print(f"[Context Truncate] Removed {removed_count} earliest messages, kept {len(truncated_conversation)}")

        return result

    def _prepare_messages(self, raw_messages: List[Dict], image_paths: List[str]) -> List[Dict]:
        """准备消息（处理多模态）"""
        import copy
        messages = copy.deepcopy(raw_messages)
        images_attached = False
        for msg in messages:
            if msg['role'] == 'user':
                content = msg['content']
                if isinstance(content, str):
                    if '<image>' in content or (not images_attached and image_paths):
                        new_content = []
                        for img_path in image_paths:
                            b64 = self._encode_image_to_base64(img_path)
                            if b64:
                                new_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                                })
                        text_part = content.replace('<image>', '').strip()
                        if text_part:
                            new_content.append({"type": "text", "text": text_part})
                        msg['content'] = new_content
                        images_attached = True
        return messages

    async def run_single_turn(self, state: SampleState) -> SampleState:
        """运行单个 turn（带重试机制）"""
        async with self.semaphore:
            max_retries = self.max_retries
            for attempt in range(max_retries):
                try:
                    state.messages = self._truncate_messages(state.messages)

                    stop_tokens = ["<|im_end|>"]
                    if self.is_glm4:
                        stop_tokens.extend(["<|user|>", "<|observation|>"])

                    request_params = {
                        "model": self.model_name,
                        "messages": state.messages,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_tokens": 8192,
                        "stop": stop_tokens,
                        "extra_body": {
                            'skip_special_tokens': False,
                            'include_stop_str_in_output': True,
                        }
                    }

                    if self.thinking_mode is not None:
                        request_params["extra_body"]['chat_template_kwargs'] = {
                            'enable_thinking': self.thinking_mode
                        }

                    response = await self.client.chat.completions.create(**request_params)

                    content = response.choices[0].message.content

                    if content is None or not str(content).strip():
                        print(f"[{state.index}] Empty response detected, stopping rollout")
                        state.status = "empty_response"
                        return state

                    state.full_response_log += content

                    # GLM-4 兼容性处理
                    if self.is_glm4 and '<tool_call>' in content and '</tool_call>' not in content:
                        json_match = re.search(r'\s*(\{.*?\})\s*(?:<\|observation\|>|<\|user\|>)?$', content, re.DOTALL)
                        if json_match:
                            content = re.sub(r'<\|observation\|>|<\|user\|>', '', content).rstrip() + '\n</tool_call>\n'

                    assistant_msg = {"role": "assistant", "content": content}
                    state.messages.append(assistant_msg)
                    state.full_messages.append(assistant_msg)

                    # 检查是否有答案
                    answer_match = state.agent.extract_answer(content)
                    if answer_match:
                        state.final_answer = answer_match
                        state.status = "finished"
                        return state

                    # 检查工具调用
                    tool_call_match = state.agent.extract_action(content)
                    if tool_call_match:
                        if self.parallel_tool_calls:
                            obs, reward, done, info = await state.agent.execute_parallel_async(action_string=content)
                        else:
                            obs, reward, done, info = await state.agent.execute_async(action_string=content)
                        clean_obs = self._clean_observation(obs)

                        user_msg = {"role": "user", "content": clean_obs}
                        state.messages.append(user_msg)
                        state.full_messages.append(user_msg)

                        if done:
                            state.status = "finished"
                            return state

                        actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                        if actual_turns >= state.max_turns:
                            print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}) after tool call")
                            state.status = "max_turns_reached"
                            return state
                    else:
                        actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])

                        if actual_turns >= state.max_turns:
                            has_answer_in_history = False
                            answer_content = None

                            for msg in state.full_messages:
                                if msg['role'] == 'assistant':
                                    msg_content = msg.get('content', '')
                                    if isinstance(msg_content, str):
                                        answers = re.findall(r'<answer>(.*?)</answer>', msg_content, re.DOTALL)
                                        if answers:
                                            has_answer_in_history = True
                                            answer_content = answers[-1].strip()
                                            break

                            if has_answer_in_history:
                                state.final_answer = answer_content
                                state.status = "finished"
                                print(f"[{state.index}] Found answer in history at turn {actual_turns}")
                                return state

                            if '<answer>' in content and '</answer>' not in content:
                                state.status = "max_turns_reached"
                                print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}): answer truncated")
                                return state

                            state.status = "max_turns_reached"
                            print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}): no answer found")
                            return state

                        user_prompt_msg = {"role": "user", "content": state.agent.user_prompt}
                        state.messages.append(user_prompt_msg)
                        state.full_messages.append(user_prompt_msg)

                    actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                    state.turn = actual_turns
                    state.status = "running"
                    return state

                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
                    error_type = type(e).__name__
                    wait_time = 2 ** attempt
                    if attempt < max_retries - 1:
                        print(f"[{state.index}] Warning: {error_type} (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[{state.index}] Error: {error_type} after {max_retries} retries, giving up")
                        state.status = "api_error"
                        state.error_info = {"error_type": error_type, "error_message": str(e), "retries": max_retries}
                        return state

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    wait_time = 2 ** attempt
                    if status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                        print(f"[{state.index}] Warning: HTTP {status_code} (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[{state.index}] Error: HTTP {status_code}, non-retryable or max retries reached")
                        state.status = "api_error"
                        state.error_info = {
                            "error_type": "HTTPStatusError",
                            "error_message": f"HTTP {status_code}: {e.response.text[:200]}",
                            "status_code": status_code,
                        }
                        return state

                except Exception as e:
                    import traceback
                    error_type = type(e).__name__
                    print(f"[{state.index}] Error: Unexpected error ({error_type}): {e}")
                    print(f"    Traceback:\n{traceback.format_exc()}")
                    state.status = "api_error"
                    state.error_info = {"error_type": error_type, "error_message": str(e)}
                    return state

            state.status = "api_error"
            return state

    def initialize_state(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        """初始化单个样本的状态"""
        try:
            raw_messages = data.get('prompt', []) or data.get('messages', [])
            image_paths = data.get('images', [])
            ground_truth = data.get('answer', '')
            extra_info = data.get('extra_info', {})

            agent = AgentEval(parser=self.tool_call_parser)
            allowed_tools = {'web_search'}
            agent.sub_tool_registry = {k: v for k, v in agent.sub_tool_registry.items() if k in allowed_tools}

            pil_images = self._load_pil_images(image_paths)
            mm_data = {"image": pil_images} if pil_images else None

            agent.reset(
                raw_prompt="",
                multi_modal_data=mm_data,
                origin_multi_modal_data=mm_data,
                world_truth=extra_info.get("world_truth_info", {})
            )

            chat_messages = self._prepare_messages(raw_messages, image_paths)

            # 使用自定义 System Prompt
            system_prompt = data.get('system', self.system_prompt)
            if not chat_messages or chat_messages[0]['role'] != 'system':
                chat_messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                chat_messages[0]['content'] = system_prompt

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
            print(f"[{line_idx}] Initialization failed: {e}")
            return None

    async def run_all_samples(self, samples_data: List[tuple]) -> List[Dict[str, Any]]:
        """运行所有样本（核心调度逻辑）"""
        state_manager = StateManager()

        print(f"Initializing {len(samples_data)} samples...")

        for idx, data in samples_data:
            state = self.initialize_state(idx, data)
            if state:
                state_manager.add_state(state)

        active_tasks = {}
        for state in state_manager.states.values():
            task = asyncio.create_task(self.run_single_turn(state))
            active_tasks[task] = state.index

        print(f"Starting async inference, initial tasks: {len(active_tasks)}")

        with tqdm(total=len(samples_data), desc="Async Inference") as pbar:
            while active_tasks:
                done, pending = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

                for completed_task in done:
                    state_idx = active_tasks.pop(completed_task)
                    updated_state = completed_task.result()

                    state_manager.states[state_idx] = updated_state
                    updated_state.trajectory_log = updated_state.agent.get_trajectory_log()

                    if updated_state.status in ["finished", "max_turns_reached", "api_error"]:
                        state_manager.mark_completed(state_idx)
                        pbar.update(1)
                    elif updated_state.status == "running":
                        next_task = asyncio.create_task(self.run_single_turn(updated_state))
                        active_tasks[next_task] = state_idx

        print(f"\nAll samples completed!")
        print(f"   Stats: {state_manager.get_stats()}")

        return state_manager.export_results()


# ================= 主函数 =================

async def main():
    parser = argparse.ArgumentParser(description='Custom Prompt Evaluation')

    # 必需参数
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file')

    # Prompt 配置
    parser.add_argument('--prompt-file', '-p', type=str, required=True,
                        help='Path to custom prompt file (.py). Must define system_prompt variable.')

    # 模型配置
    parser.add_argument('--model', type=str, default="models", help='Model name')
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE, help='API base URL')

    # 性能参数
    parser.add_argument('--max-concurrent-turns', type=int, default=32, help='Max concurrent turns')
    parser.add_argument('--max-turns-per-sample', type=int, default=8, help='Max turns per sample')
    parser.add_argument('--max-context-chars', type=int, default=60000, help='Max context chars')

    # Thinking 模式
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument('--enable-thinking', action='store_true',
                        help='Force enable thinking mode')
    thinking_group.add_argument('--disable-thinking', action='store_true',
                        help='Force disable thinking mode')

    parser.add_argument('--max-retries', type=int, default=5, help='Max retries for API calls')
    parser.add_argument('--parser', type=str, default='default',
                        choices=['default', 'deepseek', 'glm4', 'minimax', 'qwen35', 'kimi_k2'],
                        help='Tool call parser for the target model (default: "default")')
    parser.add_argument('--parallel-tool-calls', action='store_true',
                        help='Execute multiple tool_call blocks in a single response in parallel, '
                             'combining results into one user message (only supported by "default" parser)')

    args = parser.parse_args()

    # 加载自定义 Prompt
    print(f"Loading prompt from: {args.prompt_file}")
    system_prompt = load_prompt_from_file(args.prompt_file)
    print(f"Prompt loaded: {len(system_prompt)} characters")

    # 解析 thinking_mode
    thinking_mode = None
    if args.enable_thinking:
        thinking_mode = True
    elif args.disable_thinking:
        thinking_mode = False

    # 初始化引擎
    engine = AsyncInferenceEngine(
        model_name=args.model,
        api_base=args.api_base,
        api_key=DEFAULT_API_KEY,
        system_prompt=system_prompt,
        max_concurrent_turns=args.max_concurrent_turns,
        max_turns_per_sample=args.max_turns_per_sample,
        max_context_chars=args.max_context_chars,
        thinking_mode=thinking_mode,
        max_retries=args.max_retries,
        tool_call_parser=args.parser,
        parallel_tool_calls=args.parallel_tool_calls,
    )

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存运行配置
    import time
    log_file = os.path.join(output_dir, "run_config.log") if output_dir else "run_config.log"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input File: {args.input}\n")
        f.write(f"Output File: {args.output}\n")
        f.write(f"Prompt File: {args.prompt_file}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"API Base: {args.api_base}\n")
        f.write(f"Max Concurrent Turns: {args.max_concurrent_turns}\n")
        f.write(f"Max Turns Per Sample: {args.max_turns_per_sample}\n")
        f.write(f"Thinking Mode: {thinking_mode}\n")
        f.write(f"Max Retries: {args.max_retries}\n")
        f.write("=" * 80 + "\n\n")
    print(f"Config saved to: {log_file}")

    # 读取已处理记录
    processed_indices = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if 'index' in res:
                        processed_indices.add(res['index'])
                except:
                    pass
        print(f"Already processed: {len(processed_indices)} samples")

    # 读取待处理数据
    tasks_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in processed_indices:
                continue
            try:
                data = json.loads(line)
                tasks_data.append((idx, data))
            except:
                continue

    print(f"Remaining tasks: {len(tasks_data)}")

    # 运行推理
    results = await engine.run_all_samples(tasks_data)

    # 保存结果
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nInference completed! Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
