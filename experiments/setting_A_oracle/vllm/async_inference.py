#!/usr/bin/env python3
"""
async_inference.py - Setting A: Oracle-Facts QA (vLLM版本)

核心特点:
1. 无需工具调用，所有atomic facts直接提供给模型
2. 单轮对话即可完成
3. 测试模型的信息整合和推理能力

基于 async_inference.py 修改:
- 删除AgentEval和工具调用逻辑
- 修改为单轮推理
- 在System Prompt中嵌入atomic facts
"""

import argparse
import asyncio
import json
import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import httpx


# ==================== Prompt 模板 ====================

# System Prompt - 极简化，只定义角色和格式
SYSTEM_PROMPT_ORACLE = """你是一个专业的QA agent，能够根据提供的信息回答问题。

你必须严格按照以下格式回答，不能跳过任何部分：

格式示例：
<think>
你的推理过程
</think>
<answer>
你的最终答案
</answer>
"""

# User Prompt 模板 - facts在前，问题在后
USER_PROMPT_ORACLE_TEMPLATE = """<facts>
{formatted_atomic_facts}
</facts>

结合以上给定的事实信息，回答下列问题：

<question>
{question}
</question>"""


# ==================== 工具函数 ====================

def format_atomic_facts(atomic_facts: Dict[str, str]) -> str:
    """
    将atomic_facts字典格式化为编号列表（保留键值对完整信息）

    Args:
        atomic_facts: 原子事实字典（key是事实描述，value是结果）

    Returns:
        格式化后的事实列表字符串
    """
    if not atomic_facts:
        return "（无相关事实）"

    # 按key排序确保顺序一致
    sorted_keys = sorted(atomic_facts.keys())

    # 格式化为编号列表，保留键值对完整信息
    formatted_facts = []
    for i, key in enumerate(sorted_keys, 1):
        value = atomic_facts[key]
        # 格式：编号. 事实描述：结果值
        formatted_facts.append(f"{i}. {key}：{value}")

    formatted = "\n".join(formatted_facts)

    return formatted


def build_oracle_user_prompt(atomic_facts: Dict[str, str], question: str) -> str:
    """
    构建Oracle模式的User Prompt（facts在前，问题在后）

    Args:
        atomic_facts: 原子事实字典
        question: 用户问题

    Returns:
        完整的User Prompt
    """
    formatted_facts = format_atomic_facts(atomic_facts)
    return USER_PROMPT_ORACLE_TEMPLATE.format(
        formatted_atomic_facts=formatted_facts,
        question=question
    )


def extract_answer(response_text: str) -> str:
    """从模型回复中提取答案"""
    # 处理 None 或非字符串情况
    if response_text is None or not isinstance(response_text, str):
        return ""

    # 尝试匹配<answer>标签
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果没有找到标签，返回全部内容
    return response_text.strip()


def load_samples(input_file: str) -> List[Dict]:
    """加载输入样本"""
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            # 添加索引字段
            if 'index' not in data:
                data['index'] = idx
            samples.append(data)
    return samples


def save_result(result: Dict, output_file: str):
    """实时保存结果（追加模式）"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def encode_image_to_base64(image_source: str) -> str:
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


# ==================== Oracle 推理引擎 ====================

class OracleInferenceEngine:
    """
    Oracle模式推理引擎（vLLM版本）

    核心特点：
    1. 无工具调用
    2. 单轮推理
    3. atomic facts直接嵌入prompt
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "EMPTY",
        max_concurrent: int = 64,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        max_retries: int = 3
    ):
        """
        初始化推理引擎

        Args:
            model_name: 模型名称
            api_base: API 地址
            api_key: API 密钥
            max_concurrent: 最大并发样本数（因为每个样本只有1轮）
            max_tokens: 最大生成token数
            temperature: 采样温度
            enable_thinking: 是否启用thinking模式（适用于Qwen等支持的模型）
            max_retries: 失败重试次数（默认3次）
        """
        timeout_config = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_config)
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries

        print(f"✅ [Oracle vLLM] 初始化推理引擎")
        print(f"   Model: {model_name}")
        print(f"   API Base: {api_base}")
        print(f"   Max Concurrent: {max_concurrent}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"   Enable Thinking: {enable_thinking}")
        print(f"   Max Retries: {max_retries}")

    async def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本（Oracle模式，带重试）

        Args:
            sample: 输入样本

        Returns:
            处理结果
        """
        async with self.semaphore:
            index = sample.get('index', -1)

            # 重试循环
            for attempt in range(self.max_retries):
                try:
                    # 1. 提取数据
                    # 从 prompt 中提取问题和图像（如果有）
                    raw_messages = sample.get('prompt', [])
                    question = ""
                    image = None

                    # 解析 prompt 字段
                    if raw_messages:
                        for msg in raw_messages:
                            if msg.get('role') == 'user':
                                content = msg.get('content', '')
                                if isinstance(content, str):
                                    question = content
                                elif isinstance(content, list):
                                    # 多模态内容
                                    for item in content:
                                        if item.get('type') == 'text':
                                            question = item.get('text', '')
                                        elif item.get('type') == 'image_url':
                                            image_url = item.get('image_url', {}).get('url', '')
                                            # 提取base64部分
                                            if 'base64,' in image_url:
                                                image = image_url.split('base64,')[1]

                    ground_truth = sample.get('answer', '')
                    category = sample.get('category', 'unknown')

                    # 提取 atomic_facts
                    atomic_facts = sample.get('extra_info', {}).get('world_truth_info', {}).get('atomic_facts', {})

                    if not atomic_facts:
                        raise ValueError(f"Sample {index}: atomic_facts not found")

                    # 2. 构建User Prompt（包含facts和question）
                    user_prompt = build_oracle_user_prompt(atomic_facts, question)

                    # 3. 构建messages（System极简，User包含facts+问题）
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_ORACLE},
                        {"role": "user", "content": user_prompt}
                    ]

                    # 4. 调用vLLM（单次）
                    # 构建请求参数
                    request_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }

                    # 如果启用thinking，通过extra_body传递给chat_template_kwargs
                    if self.enable_thinking:
                        request_params["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }

                    response = await self.client.chat.completions.create(**request_params)

                    # 5. 提取答案
                    assistant_message = response.choices[0].message.content
                    prediction = extract_answer(assistant_message)

                    # 6. 添加assistant消息到历史
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })

                    # 7. 构建结果
                    result = {
                        "index": index,
                        "category": category,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "status": "finished",
                        "messages": messages,
                        "metrics": {
                            "total_facts": len(atomic_facts),
                            "actual_turns": 1,
                            "tool_calls": 0,
                            "format_errors": 0,
                            "retry_count": attempt  # 记录重试次数
                        }
                    }

                    return result

                except Exception as e:
                    # 如果还有重试机会，继续重试
                    if attempt < self.max_retries - 1:
                        print(f"[{index}] Attempt {attempt + 1}/{self.max_retries} failed: {e}, retrying...")
                        await asyncio.sleep(3)  # 等待1秒后重试
                        continue
                    else:
                        # 最后一次重试也失败，记录错误
                        print(f"[{index}] All {self.max_retries} attempts failed: {e}")
                        return {
                            "index": index,
                            "category": sample.get('category', 'unknown'),
                            "question": "",
                            "ground_truth": sample.get('answer', ''),
                            "prediction": "",
                            "status": "error",
                            "error_message": str(e),
                            "metrics": {
                                "total_facts": 0,
                                "actual_turns": 0,
                                "tool_calls": 0,
                                "format_errors": 0,
                                "retry_count": self.max_retries
                            }
                        }

    async def run(self, input_file: str, output_file: str):
        """
        运行Oracle推理

        Args:
            input_file: 输入JSONL文件
            output_file: 输出JSONL文件
        """
        # 1. 加载数据
        print(f"📂 加载数据: {input_file}")
        samples = load_samples(input_file)
        print(f"✅ 加载完成，共 {len(samples)} 条样本")

        # 2. 检查断点续传
        completed_indices = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    completed_indices.add(result['index'])
            print(f"🔄 检测到已完成 {len(completed_indices)} 条样本，继续处理剩余样本")

        # 3. 过滤未完成的样本
        samples_to_process = [s for s in samples if s['index'] not in completed_indices]
        print(f"📊 待处理样本数: {len(samples_to_process)}")

        if not samples_to_process:
            print("✅ 所有样本已完成！")
            return

        # 4. 创建任务
        tasks = [
            self.process_single_sample(sample)
            for sample in samples_to_process
        ]

        # 5. 执行并发任务（带进度条）
        print(f"\n🚀 开始推理...")
        print()

        results = []
        completed = 0
        pbar = async_tqdm(total=len(tasks), desc="Oracle推理")

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            # 实时保存结果
            save_result(result, output_file)
            pbar.update(1)

        pbar.close()

        # 6. 统计结果
        finished = sum(1 for r in results if r['status'] == 'finished')
        errors = sum(1 for r in results if r['status'] == 'error')

        print(f"\n✅ 推理完成！")
        print(f"   总样本数: {len(results)}")
        print(f"   成功: {finished}")
        print(f"   错误: {errors}")
        print(f"   输出文件: {output_file}")


# ==================== 主函数 ====================

async def main():
    parser = argparse.ArgumentParser(description="Setting A: Oracle-Facts QA (vLLM版本)")
    parser.add_argument('--input', required=True, help='输入JSONL文件')
    parser.add_argument('--output', required=True, help='输出JSONL文件')
    parser.add_argument('--model', default='models', help='模型名称（默认models）')
    parser.add_argument('--api-base', default="http://localhost:8000/v1", help='vLLM服务地址')
    parser.add_argument('--api-key', default='EMPTY', help='API密钥（vLLM通常为EMPTY）')
    parser.add_argument('--max-concurrent', type=int, default=64, help='最大并发样本数（默认64）')
    parser.add_argument('--max-tokens', type=int, default=2048, help='最大生成token数（默认2048）')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度（默认0.0）')
    parser.add_argument('--enable-thinking', action='store_true', help='启用thinking模式（适用于Qwen等模型）')
    parser.add_argument('--max-retries', type=int, default=3, help='失败重试次数（默认3次）')

    args = parser.parse_args()

    print("=" * 80)
    print("  Setting A: Oracle-Facts QA - vLLM推理")
    print("=" * 80)
    print()

    # 创建推理引擎
    engine = OracleInferenceEngine(
        args.model,
        args.api_base,
        args.api_key,
        args.max_concurrent,
        args.max_tokens,
        args.temperature,
        args.enable_thinking,
        args.max_retries
    )

    # 运行推理
    await engine.run(args.input, args.output)


if __name__ == '__main__':
    asyncio.run(main())
