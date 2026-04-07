#!/usr/bin/env python3
"""
async_inference.py - Setting B: Guided Search (Configurable Prompts) - vLLM版本

核心特点：
1. Turn 级别的细粒度异步
2. 支持选择不同的System Prompt (fewshot_prompt 或 guidance_prompt)
3. 指导模型进行原子化查询分解

基于async_inference.py修改：
- 主要改动：System Prompt可配置
- 其他逻辑完全相同
"""

import argparse
import asyncio
import json
import os
import sys
import base64
import re
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

# 导入Prompts
from experiments.setting_B_guided.prompt import fewshot_prompt, guidance_prompt

# ================= 全局配置 =================

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

# Prompt 映射字典
PROMPT_TEMPLATES = {
    "fewshot_prompt": fewshot_prompt,
    "guidance_prompt": guidance_prompt
}


# ================= 核心异步推理类 =================

class AsyncInferenceEngineV2:
    """
    真正的异步推理引擎 - V2 (支持可配置Prompt)

    核心改进：Turn 级别异步调度 + 可配置System Prompt
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        prompt_type: str = "fewshot_prompt",  # 新增参数
        max_concurrent_turns: int = 32,  # Turn 级别并发数
        max_turns_per_sample: int = 8,
        max_context_chars: int = 60000,
        thinking_mode: Optional[bool] = None, # thinking 模式：True=启用, False=禁用, None=不干预
        max_retries: int = 5,                 # API 调用失败时的最大重试次数
    ):
        """
        初始化推理引擎

        Args:
            model_name: 模型名称
            api_base: API 地址
            api_key: API 密钥
            prompt_type: Prompt类型 ("fewshot_prompt" 或 "guidance_prompt")
            max_concurrent_turns: Turn 级别的最大并发数（不是样本级别）
            max_turns_per_sample: 每个样本的最大轮数
            max_context_chars: 上下文最大字符数
            thinking_mode: 控制模型内置 thinking 模式。
                None（默认）：不传 chat_template_kwargs，完全使用模型自身的默认行为。
                True：强制传 enable_thinking=True，开启内置 thinking。
                False：强制传 enable_thinking=False，禁用内置 thinking。
                    ⚠️ 注意：对 DeepResearch-30B-A3B 传 False 会导致输出严重乱码，慎用。
        """
        timeout_config = httpx.Timeout(connect=100.0, read=600.0, write=100.0, pool=100.0)
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_config)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.max_turns_per_sample = max_turns_per_sample
        self.max_context_chars = max_context_chars
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries

        # 设置System Prompt
        if prompt_type not in PROMPT_TEMPLATES:
            raise ValueError(f"Invalid prompt_type: {prompt_type}. Must be one of {list(PROMPT_TEMPLATES.keys())}")

        self.system_prompt = PROMPT_TEMPLATES[prompt_type]
        self.prompt_type = prompt_type

        # 检测是否为 GLM-4 系列模型
        self.is_glm4 = 'glm' in model_name.lower() or 'GLM' in model_name

        print(f"✅ [V2] 初始化推理引擎:")
        print(f"   - Model: {model_name}")
        print(f"   - Prompt Type: {prompt_type}")
        print(f"   - MaxConcurrentTurns: {max_concurrent_turns}")
        if self.is_glm4:
            print(f"   - GLM-4 兼容模式: 启用")
        print(f"   - MaxTurnsPerSample: {max_turns_per_sample}")
        print(f"   - ThinkingMode: {thinking_mode}")
        print(f"   - Max Retries: {max_retries}")

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
        match = re.search(r'<tool_response>(.*?)</tool_response>', obs, re.DOTALL)
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

        print(f"[Context Truncate] 总字符数 {total_chars} 超过限制 {self.max_context_chars}，开始截断...")

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
            print(f"[Context Truncate] 删除了最早的 {removed_count} 条消息，保留 {len(truncated_conversation)} 条")

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
        """
        运行单个 turn（带重试机制）

        这是异步调度的最小单元

        核心原则：
        1. 完整轨迹保存：所有消息必须同步保存到 full_messages
        2. 轮数限制限制：基于 full_messages 的实际对话轮次判断
        3. 连接/超时错误指数退避重试
        """
        async with self.semaphore:
            # 重试机制：使用 self.max_retries（可通过构造函数配置）
            max_retries = self.max_retries
            for attempt in range(max_retries):
                try:
                    # 1. 截断消息用于推理（避免超过上下文窗口）
                    # full_messages 保持完整轨迹，不在这里修改
                    state.messages = self._truncate_messages(state.messages)

                    # 2. 调用 Agent API（使用截断的 messages）
                    # GLM-4 兼容：添加 <|observation|> 和 <|user|> 到 stop tokens
                    stop_tokens = ["<|im_end|>"]
                    if self.is_glm4:
                        stop_tokens.extend(["<|observation|>", "<|user|>"])

                    # 构建请求参数
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
                    # 仅在用户显式指定时才传 chat_template_kwargs，
                    # 不传时使用模型默认行为（DeepResearch-30B-A3B 默认已正确输出 <think>）
                    if self.thinking_mode is not None:
                        request_params["extra_body"]['chat_template_kwargs'] = {
                            'enable_thinking': self.thinking_mode
                        }

                    # 如果显式启用 thinking（不传 False）则不覆盖，此处已在上方 chat_template_kwargs 统一处理

                    response = await self.client.chat.completions.create(**request_params)

                    content = response.choices[0].message.content

                    # ✅ Bug #1 修复：检查空响应（Doubao Bug）
                    if content is None or not str(content).strip():
                        print(f"[{state.index}] Empty response detected, stopping rollout")
                        state.status = "empty_response"
                        return state

                    state.full_response_log += content

                    # GLM-4 兼容性处理：如果模型在 <tool_call> 后被 stop token 截断，补全 </tool_call>
                    if self.is_glm4 and '<tool_call>' in content and '</tool_call>' not in content:
                        # 检查是否有完整的 JSON（可能以 <|observation|> 或 <|user|> 结尾）
                        import re
                        json_match = re.search(r'<tool_call>\s*(\{.*?\})\s*(?:<\|observation\|>|<\|user\|>)?$', content, re.DOTALL)
                        if json_match:
                            # 移除 special tokens 并补全 </tool_call> 标签
                            content = re.sub(r'<\|observation\|>|<\|user\|>', '', content).rstrip() + '\n</tool_call>'
                            print(f"[{state.index}] GLM-4 格式修正：补全 </tool_call> 标签")

                    # 3. 核心修复：同时添加到 messages 和 full_messages（使用修正后的 content）
                    assistant_msg = {"role": "assistant", "content": content}
                    state.messages.append(assistant_msg)
                    state.full_messages.append(assistant_msg)

                    # 4. 检查是否有答案（最高优先级）
                    answer_match = state.agent.extract_answer(content)
                    if answer_match:
                        state.final_answer = answer_match
                        state.status = "finished"
                        return state

                    # 5. 检查工具调用
                    tool_call_match = state.agent.extract_action(content)
                    if tool_call_match:
                        # 执行工具（World Model 异步调用）
                        obs, reward, done, info = await state.agent.execute_async(action_string=content)
                        clean_obs = self._clean_observation(obs)

                        # 同时添加到两个消息列表
                        user_msg = {"role": "user", "content": clean_obs}
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
                        # 6. 没有工具调用也没有答案，检查轮数限制
                        actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])

                        if actual_turns >= state.max_turns:
                            # 已经达到轮数限制，检查所有 assistant 消息中是否有 answer
                            import re
                            has_answer_in_history = False
                            answer_content = None

                            # ✅ 检查所有 assistant 消息
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
                                # 在历史消息中找到了答案
                                state.final_answer = answer_content
                                state.status = "finished"
                                print(f"[{state.index}] Found answer in history at turn {actual_turns}")
                                return state

                            # 检查当前 content 是否被截断
                            if '<answer>' in content and '</answer>' not in content:
                                # 答案被截断
                                state.status = "max_turns_reached"
                                print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}): answer truncated")
                                return state

                            # 确实没有答案
                            state.status = "max_turns_reached"
                            print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}): no answer found")
                            return state

                        # 未达到限制，提醒继续
                        user_prompt_msg = {"role": "user", "content": state.agent.user_prompt}
                        state.messages.append(user_prompt_msg)
                        state.full_messages.append(user_prompt_msg)

                    # 7. 继续下一轮
                    actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                    state.turn = actual_turns  # 更新为实际轮次
                    state.status = "running"
                    return state

                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
                    # ✅ 修复 api_error：超时类错误全部走指数退避重试
                    error_type = type(e).__name__
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, ...
                    if attempt < max_retries - 1:
                        print(f"[{state.index}] ⚠️ {error_type} (attempt {attempt+1}/{max_retries}), "
                              f"retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[{state.index}] ❌ {error_type} after {max_retries} retries, giving up")
                        state.status = "api_error"
                        state.error_info = {"error_type": error_type, "error_message": str(e), "retries": max_retries}
                        return state

                except httpx.HTTPStatusError as e:
                    # ✅ 修复 api_error：429 / 5xx 走指数退避重试，其他不重试
                    status_code = e.response.status_code
                    wait_time = 2 ** attempt
                    if status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                        print(f"[{state.index}] ⚠️ HTTP {status_code} (attempt {attempt+1}/{max_retries}), "
                              f"retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"[{state.index}] ❌ HTTP {status_code}, non-retryable or max retries reached")
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
                    print(f"[{state.index}] ❌ Unexpected error ({error_type}): {e}")
                    print(f"    Traceback:\n{traceback.format_exc()}")
                    state.status = "api_error"
                    state.error_info = {"error_type": error_type, "error_message": str(e)}
                    return state

            # 不应该到这里，但以防万一
            state.status = "api_error"
            return state

    def initialize_state(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        """初始化单个样本的状态"""
        try:
            raw_messages = data.get('prompt', []) or data.get('messages', [])
            image_paths = data.get('images', [])
            ground_truth = data.get('answer', '')
            extra_info = data.get('extra_info', {})

            # 创建 Agent
            agent = AgentEval()
            allowed_tools = {'web_search'}
            agent.sub_tool_registry = {k: v for k, v in agent.sub_tool_registry.items() if k in allowed_tools}

            # 准备多模态数据
            pil_images = self._load_pil_images(image_paths)
            mm_data = {"image": pil_images} if pil_images else None

            # 重置 Agent
            agent.reset(
                raw_prompt="",
                multi_modal_data=mm_data,
                origin_multi_modal_data=mm_data,
                world_truth=extra_info.get("world_truth_info", {})
            )

            # 准备消息
            chat_messages = self._prepare_messages(raw_messages, image_paths)

            # 设置 System Prompt (使用配置的prompt)
            system_prompt = data.get('system', self.system_prompt)
            if not chat_messages or chat_messages[0]['role'] != 'system':
                chat_messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                chat_messages[0]['content'] = system_prompt

            # 创建状态
            state = SampleState(
                index=line_idx,
                data=data,
                messages=chat_messages,
                full_messages=chat_messages.copy(),  # 初始化时也复制一份完整消息
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
        运行所有样本（核心调度逻辑）

        Args:
            samples_data: [(idx, data), ...]

        Returns:
            results: 所有样本的结果
        """
        # 1. 初始化状态管理器
        state_manager = StateManager()

        print(f"📋 [V2] 初始化 {len(samples_data)} 个样本...")

        for idx, data in samples_data:
            state = self.initialize_state(idx, data)
            if state:
                state_manager.add_state(state)

        # 2. 创建所有样本的第一个 turn 任务
        active_tasks = {}  # task -> state.index
        for state in state_manager.states.values():
            task = asyncio.create_task(self.run_single_turn(state))
            active_tasks[task] = state.index

        print(f"🚀 [V2] 开始异步调度，初始任务数: {len(active_tasks)}")

        # 3. 全局异步调度循环
        with tqdm(total=len(samples_data), desc="🚀 Async Inference V2") as pbar:
            while active_tasks:
                # 等待任意一个任务完成
                done, pending = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

                for completed_task in done:
                    # 获取完成的状态
                    state_idx = active_tasks.pop(completed_task)
                    updated_state = completed_task.result()

                    # 更新状态管理器
                    state_manager.states[state_idx] = updated_state

                    # 更新轨迹日志
                    updated_state.trajectory_log = updated_state.agent.get_trajectory_log()

                    # 检查状态
                    if updated_state.status in ["finished", "max_turns_reached", "api_error"]:
                        # 样本完成
                        state_manager.mark_completed(state_idx)
                        pbar.update(1)
                    elif updated_state.status == "running":
                        # 需要继续，创建下一个 turn 任务
                        next_task = asyncio.create_task(self.run_single_turn(updated_state))
                        active_tasks[next_task] = state_idx

        print(f"\n✅ [V2] 所有样本完成！")
        print(f"   Stats: {state_manager.get_stats()}")

        # 4. 导出结果
        return state_manager.export_results()


# ================= 主函数 =================

async def main():
    parser = argparse.ArgumentParser(description='Async Inference V2 - Setting B (可配置Prompt)')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--model', type=str, default="models", help='Model name')
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE, help='API base URL')
    parser.add_argument('--prompt-type', type=str, default="fewshot_prompt",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help='Prompt类型: fewshot_prompt 或 guidance_prompt')
    parser.add_argument('--max-concurrent-turns', type=int, default=32, help='最大并发 turn 数')
    parser.add_argument('--max-turns-per-sample', type=int, default=8, help='每个样本的最大轮数')
    parser.add_argument('--max-context-chars', type=int, default=60000, help='上下文最大字符数')
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument('--enable-thinking', action='store_true',
                        help='强制启用模型内置 thinking 模式（传 enable_thinking=True）')
    thinking_group.add_argument('--disable-thinking', action='store_true',
                        help='强制禁用模型内置 thinking 模式（传 enable_thinking=False）。'
                             '⚠️ 对 DeepResearch-30B-A3B 等模型传 False 会导致输出严重乱码，慎用。')
    parser.add_argument('--max-retries', type=int, default=5,
                        help='API 调用失败时的最大重试次数（指数退避），默认 5')

    args = parser.parse_args()

    # 解析 thinking_mode：None=不干预，True=强制开启，False=强制关闭
    thinking_mode = None
    if args.enable_thinking:
        thinking_mode = True
    elif args.disable_thinking:
        thinking_mode = False

    # 初始化引擎
    engine = AsyncInferenceEngineV2(
        model_name=args.model,
        api_base=args.api_base,
        api_key=DEFAULT_API_KEY,
        prompt_type=args.prompt_type,
        max_concurrent_turns=args.max_concurrent_turns,
        max_turns_per_sample=args.max_turns_per_sample,
        max_context_chars=args.max_context_chars,
        thinking_mode=thinking_mode,
        max_retries=args.max_retries,
    )

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存运行配置到 .log 文件
    import time
    log_file = os.path.join(output_dir, "run_config.log")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Input File: {args.input}\n")
        f.write(f"Output File: {args.output}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"API Base: {args.api_base}\n")
        f.write(f"Prompt Type: {args.prompt_type}\n")
        f.write(f"Max Concurrent Turns: {args.max_concurrent_turns}\n")
        f.write(f"Max Turns Per Sample: {args.max_turns_per_sample}\n")
        f.write(f"Max Context Chars: {args.max_context_chars}\n")
        f.write(f"Enable Thinking: {thinking_mode}\n")
        f.write(f"Max Retries: {args.max_retries}\n")
        f.write("=" * 80 + "\n\n")
    print(f"✅ 运行配置已保存到: {log_file}")

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
        print(f"✅ 已处理记录数: {len(processed_indices)}")

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

    print(f"📋 剩余待处理任务数: {len(tasks_data)}")

    # 运行推理
    results = await engine.run_all_samples(tasks_data)

    # 保存结果
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n✅ 推理完成！结果已保存到: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
