#!/usr/bin/env python3
"""
async_inference_oracle_api.py - Setting A: Oracle-Facts QA (API版本)

核心特点:
1. 无需工具调用，所有atomic facts直接提供给模型
2. 单轮对话即可完成
3. 测试模型的信息整合和推理能力

使用示例:
    python experiments/setting_A_oracle/api/async_inference_oracle_api.py \
        --provider openai \
        --model qwen-plus \
        --input data/mpw_bench_full.jsonl \
        --output results/setting_A/api_results.jsonl \
        --qps 2 \
        --max-concurrent 32
"""

import asyncio
import json
import time
import os
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm


# ==================== Prompt 模板 ====================

# System Prompt - 极简化，只定义角色和格式
SYSTEM_PROMPT_ORACLE = """你是一个专业的QA agent，能够根据提供的信息回答问题。

你必须严格按照以下格式回答，不能跳过任何部分：

第一步：先输出 <think> 标签，在其中写出你的推理过程（例如：分析哪些事实相关、如何推理得出答案等）

第二步：再输出 <answer> 标签，在其中写出最终答案

格式示例：
<think>
[必须填写推理过程，不能为空]
</think>

<answer>
[最终答案]
</answer>

重要：必须同时包含 <think> 和 <answer> 两个部分，<think> 中必须有实际的推理内容。"""

# User Prompt 模板 - facts在前，问题在后
USER_PROMPT_ORACLE_TEMPLATE = """<facts>
{formatted_atomic_facts}
</facts>

根据以上事实信息，回答下列问题：

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

    Example:
        Input: {"萨内在2027/28赛季效力于拜仁": "成立", "奥蓬达进球数": "10"}
        Output: "1. 萨内在2027/28赛季效力于拜仁：成立\n2. 奥蓬达进球数：10"
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
    """
    从模型回复中提取答案

    Args:
        response_text: 模型的完整回复

    Returns:
        提取的答案文本

    Example:
        Input: "<think>思考...</think>\n<answer>皇家马德里</answer>"
        Output: "皇家马德里"
    """
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


# ==================== 核心推理函数 ====================

async def process_single_sample_oracle(
    sample: Dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    semaphore: asyncio.Semaphore,
    qps_limiter: TokenBucket,
    max_tokens: int = 2048,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Oracle模式处理单个样本（API版本）

    Args:
        sample: 输入样本
        client: AsyncOpenAI客户端
        model_name: 模型名称
        semaphore: 并发控制信号量
        qps_limiter: QPS速率限制器
        max_tokens: 最大生成token数
        temperature: 采样温度

    Returns:
        处理结果
    """
    async with semaphore:
        try:
            # 1. 提取数据
            index = sample.get('index', -1)

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

            # 4. 速率限制
            await qps_limiter.acquire()

            # 5. 调用LLM（单次）
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # 6. 提取答案
            assistant_message = response.choices[0].message.content
            prediction = extract_answer(assistant_message)

            # 7. 添加assistant消息到历史
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

            # 8. 构建结果
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
                    "format_errors": 0
                }
            }

            return result

        except Exception as e:
            # 错误处理
            print(f"[{sample.get('index', -1)}] Error: {e}")
            return {
                "index": sample.get('index', -1),
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
                    "format_errors": 0
                }
            }


async def run_inference_oracle_api(
    input_file: str,
    output_file: str,
    client: AsyncOpenAI,
    model_name: str,
    max_concurrent: int = 32,
    qps: float = 2.0,
    max_tokens: int = 2048,
    temperature: float = 0.0
):
    """
    运行Oracle推理（API版本）

    Args:
        input_file: 输入JSONL文件
        output_file: 输出JSONL文件
        client: AsyncOpenAI客户端
        model_name: 模型名称
        max_concurrent: 最大并发数
        qps: QPS限制
        max_tokens: 最大生成token数
        temperature: 采样温度
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

    # 4. 创建并发控制
    semaphore = asyncio.Semaphore(max_concurrent)
    qps_limiter = TokenBucket(rate=qps, capacity=int(qps * 2), name="QPS")

    # 5. 创建任务
    tasks = [
        process_single_sample_oracle(sample, client, model_name, semaphore, qps_limiter, max_tokens, temperature)
        for sample in samples_to_process
    ]

    # 6. 执行并发任务（带进度条）
    print(f"\n🚀 开始推理...")
    print(f"   模型: {model_name}")
    print(f"   最大并发: {max_concurrent}")
    print(f"   QPS限制: {qps}")
    print(f"   Max Tokens: {max_tokens}")
    print(f"   Temperature: {temperature}")
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

    # 7. 统计结果
    finished = sum(1 for r in results if r['status'] == 'finished')
    errors = sum(1 for r in results if r['status'] == 'error')

    print(f"\n✅ 推理完成！")
    print(f"   总样本数: {len(results)}")
    print(f"   成功: {finished}")
    print(f"   错误: {errors}")
    print(f"   输出文件: {output_file}")


# ==================== 主函数 ====================

async def main():
    parser = argparse.ArgumentParser(description="Setting A: Oracle-Facts QA (API版本)")
    parser.add_argument('--provider', required=True, help='API提供商 (openai/azure/custom)')
    parser.add_argument('--model', required=True, help='模型名称')
    parser.add_argument('--input', required=True, help='输入JSONL文件')
    parser.add_argument('--output', required=True, help='输出JSONL文件')
    parser.add_argument('--qps', type=float, default=2.0, help='每秒请求数（默认2）')
    parser.add_argument('--max-concurrent', type=int, default=32, help='最大并发数（默认32）')
    parser.add_argument('--max-tokens', type=int, default=2048, help='最大生成token数（默认2048）')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度（默认0.0）')
    parser.add_argument('--api-key', default=None, help='API密钥（可选，默认从环境变量读取）')
    parser.add_argument('--base-url', default=None, help='API基础URL（仅custom provider需要）')

    args = parser.parse_args()

    print("=" * 80)
    print("  Setting A: Oracle-Facts QA - API推理")
    print("=" * 80)
    print()

    # 创建客户端
    print(f"📡 初始化API客户端...")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model}")
    client = APIClientFactory.create_client(args.provider, args.api_key, args.base_url)
    print(f"✅ 客户端初始化成功")
    print()

    # 运行推理
    await run_inference_oracle_api(
        args.input,
        args.output,
        client,
        args.model,
        args.max_concurrent,
        args.qps,
        args.max_tokens,
        args.temperature
    )


if __name__ == '__main__':
    asyncio.run(main())
