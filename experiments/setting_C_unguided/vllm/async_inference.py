#!/usr/bin/env python3
"""
async_inference.py - 真正的异步推理引擎

核心改进：
1. Turn 级别的细粒度异步
2. 全局任务队列 + 动态调度
3. 100% 资源利用率

性能：6-9倍提升
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

# ================= 全局配置 =================

DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

# System Prompt
INSTRUCTION_PROMPT_SYSTEM = """
你是一个ReAct范式的agent，能够接受文本输入，回答用户问题
对于一些复杂的问题，你可以选择调用工具帮助你解决问题
你可以调用的工具包括以下两种：
web_search:
-description: Retrieve external text information from internet based on your provided text query.
-input: only text query(**this tool cannot see the image**)
-output: top-4 text(you can attempt to change your query if the previous search result is not satisfactory)
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


# ================= 核心异步推理类 =================

class AsyncInferenceEngineV2:
    """
    真正的异步推理引擎 - V2

    核心改进：Turn 级别异步调度
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        max_concurrent_turns: int = 32,  # Turn 级别并发数
        max_turns_per_sample: int = 8,
        max_context_chars: int = 60000,
        read_timeout: float = 600.0,          # 增加读取超时时间（秒）
        thinking_mode: Optional[bool] = None, # thinking 模式：True=启用, False=禁用, None=不干预
        max_retries: int = 5,                 # API 调用失败时的最大重试次数
    ):
        """
        初始化推理引擎

        Args:
            model_name: 模型名称
            api_base: API 地址
            api_key: API 密钥
            max_concurrent_turns: Turn 级别的最大并发数（不是样本级别）
            max_turns_per_sample: 每个样本的最大轮数
            max_context_chars: 上下文最大字符数
            read_timeout: API 读取超时时间（秒），默认 600s (10分钟)
            thinking_mode: 控制模型内置 thinking 模式。
                None（默认）：不传 chat_template_kwargs，完全使用模型自身的默认行为。
                    测试证明 DeepResearch-30B-A3B 不传时已能正确输出 <think>...</think>。
                True：强制传 enable_thinking=True，开启内置 thinking。
                False：强制传 enable_thinking=False，禁用内置 thinking。
                    ⚠️ 注意：对 DeepResearch-30B-A3B 传 False 会导致输出严重乱码，
                    请勿对不确定的模型使用此选项。
            max_retries: API 调用遇到网络错误时的最大重试次数（指数退避）。
                覆盖 ReadTimeout、ConnectTimeout、HTTP 429/5xx 等可重试错误。
        """
        timeout_config = httpx.Timeout(
            connect=30.0,      # 连接超时 30s
            read=read_timeout, # 读取超时由参数控制
            write=30.0,        # 写入超时 30s
            pool=30.0          # 连接池超时 30s
        )
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_config)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_turns)
        self.max_turns_per_sample = max_turns_per_sample
        self.max_context_chars = max_context_chars
        self.read_timeout = read_timeout
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries

        # 检测 GLM-4 系列：需要额外 stop tokens 和 tool_call 格式修正
        self.is_glm4 = 'glm' in model_name.lower()

        print(f"✅ [V2] 初始化推理引擎: Model={model_name}, MaxConcurrentTurns={max_concurrent_turns}, "
              f"MaxTurnsPerSample={max_turns_per_sample}, ReadTimeout={read_timeout}s, "
              f"ThinkingMode={thinking_mode}, MaxRetries={max_retries}, GLM4={self.is_glm4}")

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

    def _estimate_tokens(self, messages: List[Dict]) -> tuple:
        """
        估算消息的 token 数

        Returns:
            (estimated_tokens, search_result_chars, total_chars)
        """
        total_chars = 0
        search_result_chars = 0
        image_count = 0

        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', '')

            if isinstance(content, str):
                total_chars += len(content)
                if role == 'user' and 'search_result' in content:
                    search_result_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        total_chars += len(text)
                        if role == 'user' and 'search_result' in text:
                            search_result_chars += len(text)
                    elif item.get('type') == 'image_url':
                        image_count += 1

        # Token 估算：
        # - 搜索结果（重复词汇多）：约 3 字符/token
        # - 普通文本（中英混合）：约 2.5 字符/token
        # - 图片（Qwen2-VL）：约 512 token
        estimated_tokens = (
            search_result_chars / 3.0 +
            (total_chars - search_result_chars) / 2.5 +
            image_count * 512
        )

        return int(estimated_tokens), search_result_chars, total_chars

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
        运行单个 turn（带指数退避重试）

        修复内容：
        1. API 通信错误指数退避重试（ReadTimeout/ConnectTimeout/429/5xx），
           不再因单次超时永久终止 rollout
        2. thinking_mode=None 时不干预 chat_template_kwargs，使用模型默认行为；
           测试证明 DeepResearch-30B-A3B 默认已能正确输出 <think>...</think>，
           强制传 False 反而导致输出严重乱码
        3. GLM-4 stop token 兼容 + tool_call 格式修正
        """
        async with self.semaphore:
            actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])

            for attempt in range(self.max_retries):
                try:
                    # 1. 截断消息用于推理（避免超过上下文窗口）
                    state.messages = self._truncate_messages(state.messages)

                    estimated_tokens, search_result_chars, total_chars = self._estimate_tokens(state.messages)
                    print(f"[{state.index}] Turn {actual_turns+1} (attempt {attempt+1}/{self.max_retries}): "
                          f"estimated_tokens={estimated_tokens}, chars={total_chars}, "
                          f"search_chars={search_result_chars}")

                    # 2. 构建请求参数
                    # GLM-4 需要额外 stop tokens，否则 tool_call 会被截断
                    stop_tokens = ["<|im_end|>"]
                    if self.is_glm4:
                        stop_tokens.extend(["<|observation|>", "<|user|>"])

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

                    # 3. 调用 API
                    response = await self.client.chat.completions.create(**request_params)
                    content = response.choices[0].message.content

                    # ✅ 检查空响应（Doubao Bug）
                    if content is None or not str(content).strip():
                        print(f"[{state.index}] Empty response detected, stopping rollout")
                        state.status = "empty_response"
                        return state

                    # GLM-4 兼容：补全被 stop token 截断的 </tool_call>
                    if self.is_glm4 and '<tool_call>' in content and '</tool_call>' not in content:
                        json_match = re.search(
                            r'<tool_call>\s*(\{.*?\})\s*(?:<\|observation\|>|<\|user\|>)?$',
                            content, re.DOTALL
                        )
                        if json_match:
                            content = re.sub(r'<\|observation\|>|<\|user\|>', '', content).rstrip() + '\n</tool_call>'
                            print(f"[{state.index}] GLM-4 格式修正：补全 </tool_call> 标签")

                    state.full_response_log += content

                    # 4. 同时添加到 messages 和 full_messages
                    assistant_msg = {"role": "assistant", "content": content}
                    state.messages.append(assistant_msg)
                    state.full_messages.append(assistant_msg)

                    # 5. 检查是否有答案（最高优先级）
                    answer_match = state.agent.extract_answer(content)
                    if answer_match:
                        state.final_answer = answer_match
                        state.status = "finished"
                        return state

                    # 6. 检查工具调用
                    tool_call_match = state.agent.extract_action(content)
                    if tool_call_match:
                        obs, reward, done, info = await state.agent.execute_async(action_string=content)
                        clean_obs = self._clean_observation(obs)

                        user_msg = {"role": "user", "content": clean_obs}
                        state.messages.append(user_msg)
                        state.full_messages.append(user_msg)

                        if done:
                            state.status = "finished"
                            return state

                        # 工具调用后也检查轮数限制
                        actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                        if actual_turns >= state.max_turns:
                            print(f"[{state.index}] Max turns reached ({actual_turns}/{state.max_turns}) after tool call")
                            state.status = "max_turns_reached"
                            return state
                    else:
                        # 7. 没有工具调用也没有答案，检查轮数限制
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

                    # 8. 继续下一轮
                    actual_turns = len([msg for msg in state.full_messages if msg['role'] == 'assistant'])
                    state.turn = actual_turns
                    state.status = "running"
                    return state

                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
                    # ✅ 修复 api_error：超时类错误全部走指数退避重试
                    error_type = type(e).__name__
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                    if attempt < self.max_retries - 1:
                        print(f"[{state.index}] ⚠️ {error_type} (attempt {attempt+1}/{self.max_retries}), "
                              f"retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        estimated_tokens, _, _ = self._estimate_tokens(state.messages)
                        print(f"[{state.index}] ❌ {error_type} after {self.max_retries} retries, giving up")
                        state.status = "api_error"
                        state.error_info = {
                            "error_type": error_type,
                            "error_message": str(e),
                            "estimated_tokens": estimated_tokens,
                            "retries": self.max_retries,
                            "turn": actual_turns + 1
                        }
                        return state

                except httpx.HTTPStatusError as e:
                    # ✅ 修复 api_error：429 / 5xx 走指数退避重试，其他 HTTP 错误不重试
                    status_code = e.response.status_code
                    wait_time = 2 ** attempt
                    if status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                        print(f"[{state.index}] ⚠️ HTTP {status_code} (attempt {attempt+1}/{self.max_retries}), "
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
                            "retries": attempt + 1,
                            "turn": actual_turns + 1
                        }
                        return state

                except Exception as e:
                    import traceback
                    error_type = type(e).__name__
                    error_traceback = traceback.format_exc()
                    estimated_tokens, _, _ = self._estimate_tokens(state.messages)
                    print(f"[{state.index}] ❌ Unexpected error ({error_type}): {e}")
                    print(f"    Traceback:\n{error_traceback}")
                    state.status = "api_error"
                    state.error_info = {
                        "error_type": error_type,
                        "error_message": str(e),
                        "estimated_tokens": estimated_tokens,
                        "turn": actual_turns + 1,
                        "traceback": error_traceback
                    }
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

            # 设置 System Prompt
            system_prompt = data.get('system', INSTRUCTION_PROMPT_SYSTEM)
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
    parser = argparse.ArgumentParser(description='Async Inference V2 - 真正的异步推理')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--model', type=str, default="models", help='Model name')
    parser.add_argument('--api-base', type=str, default=DEFAULT_API_BASE, help='API base URL')
    parser.add_argument('--api-key', type=str, default=DEFAULT_API_KEY, help='API key')
    parser.add_argument('--max-concurrent-turns', type=int, default=32, help='最大并发 turn 数')
    parser.add_argument('--max-turns-per-sample', type=int, default=8, help='每个样本的最大轮数')
    parser.add_argument('--max-context-chars', type=int, default=60000, help='上下文最大字符数')
    parser.add_argument('--read-timeout', type=float, default=600.0, help='API 读取超时时间（秒），默认 600s (10分钟)')
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
        api_key=args.api_key,
        max_concurrent_turns=args.max_concurrent_turns,
        max_turns_per_sample=args.max_turns_per_sample,
        max_context_chars=args.max_context_chars,
        read_timeout=args.read_timeout,
        thinking_mode=thinking_mode,
        max_retries=args.max_retries,
    )

    # 创建输出目录
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

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
