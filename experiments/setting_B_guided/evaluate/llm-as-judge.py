#!/usr/bin/env python3
"""
llm-as-judge.py - Setting B: Guided Search (Few-shot) 评估（LLM-as-Judge）

使用LLM判断模型预测是否与ground truth一致。
只有两种结果：Correct 或 Incorrect

注意：Setting B的数据格式中，question需要从messages中提取
"""

import asyncio
import json
import os
import argparse
import re
from typing import Dict, Any, List
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import httpx


# ==================== Judge Prompt ====================

JUDGE_SYSTEM_PROMPT = """你是一个专业的答案评估专家。你的任务是判断模型的预测答案是否与标准答案一致。

评估标准：
1. 如果预测答案与标准答案在语义上一致，判定为"Correct"
2. 如果预测答案与标准答案不一致、缺少关键信息、或完全错误，判定为"Incorrect"

注意事项：
- 关注语义一致性，不要求字面完全相同
- 数字、日期、名称等关键信息必须准确
- 如果标准答案包含多个部分，预测答案必须包含所有部分才算正确
- 预测答案可以包含额外的解释，只要核心信息正确

输出格式：
<think>
你的判断理由
</think>
<answer>
Correct 或 Incorrect（只能二选一）
</answer>"""

JUDGE_USER_PROMPT_TEMPLATE = """问题：
{question}

标准答案：
{ground_truth}

模型预测：
{prediction}

请判断模型预测是否正确。"""


# ==================== 工具函数 ====================

def extract_answer(response_text: str) -> str:
    """从Judge响应中提取判决结果"""
    if not response_text:
        return "Incorrect"

    # 尝试匹配<answer>标签
    match = re.search(r'<answer>\s*(Correct|Incorrect)\s*</answer>', response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # 如果没有标签，尝试直接匹配
    if "Correct" in response_text and "Incorrect" not in response_text:
        return "Correct"

    # 默认返回Incorrect
    return "Incorrect"


def extract_think(response_text: str) -> str:
    """从Judge响应中提取推理过程"""
    if not response_text:
        return ""

    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response_text.strip()


# ==================== Judge 引擎 ====================

class SettingBJudgeEngine:
    """Setting B 的 LLM-as-Judge 评估引擎"""

    def __init__(
        self,
        judge_model: str,
        api_base: str,
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
        max_tokens: int = 8192,
        temperature: float = 0.8,
        max_retries: int = 3,
        enable_thinking: bool = True
    ):
        """
        初始化Judge引擎

        Args:
            judge_model: Judge模型名称
            api_base: API地址
            api_key: API密钥
            max_concurrent: 最大并发数
            max_tokens: 最大生成token数
            temperature: 采样温度
            max_retries: 失败重试次数
            enable_thinking: 是否启用thinking模式
        """
        timeout_config = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout_config)
        self.judge_model = judge_model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking

        print(f"✅ [Judge Engine] 初始化完成")
        print(f"   Judge Model: {judge_model}")
        print(f"   API Base: {api_base}")
        print(f"   Max Concurrent: {max_concurrent}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Temperature: {temperature}")
        print(f"   Max Retries: {max_retries}")
        print(f"   Enable Thinking: {enable_thinking}")

    def _extract_question_from_messages(self, messages: list) -> str:
        """
        从messages中提取用户问题

        Args:
            messages: 对话消息列表

        Returns:
            提取的问题文本
        """
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # 多模态内容，提取text部分
                    for item in content:
                        if item.get('type') == 'text':
                            return item.get('text', '')
        return ""

    async def judge_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本（带重试机制）

        Args:
            sample: 包含messages, ground_truth, prediction的字典

        Returns:
            评估结果
        """
        async with self.semaphore:
            index = sample.get('index', -1)

            # Setting B特有：从messages中提取question
            messages = sample.get('messages', [])
            question = self._extract_question_from_messages(messages)

            ground_truth = sample.get('ground_truth', '')
            prediction = sample.get('prediction', '')

            # 如果prediction为空，直接判定为Incorrect
            if not prediction or prediction.strip() == "":
                return {
                    "index": index,
                    "answer": "Incorrect",
                    "think": "预测答案为空",
                    "judge_status": "skipped"
                }

            # 如果ground_truth为空，无法评估
            if not ground_truth or ground_truth.strip() == "":
                return {
                    "index": index,
                    "answer": "Incorrect",
                    "think": "标准答案缺失，无法评估",
                    "judge_status": "skipped"
                }

            # 重试循环
            for attempt in range(self.max_retries):
                try:
                    # 构建Judge prompt
                    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction
                    )

                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]

                    # 准备请求参数
                    request_params = {
                        "model": self.judge_model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }

                    # 如果启用thinking模式
                    if self.enable_thinking:
                        request_params["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": True}
                        }

                    # 调用Judge模型
                    response = await self.client.chat.completions.create(**request_params)

                    judge_response = response.choices[0].message.content

                    # 提取判决和推理
                    answer = extract_answer(judge_response)
                    think = extract_think(judge_response)

                    return {
                        "index": index,
                        "answer": answer,
                        "think": think,
                        "judge_response": judge_response,
                        "judge_status": "success"
                    }

                except Exception as e:
                    # 如果还有重试机会，继续重试
                    if attempt < self.max_retries - 1:
                        print(f"[{index}] Judge attempt {attempt + 1}/{self.max_retries} failed: {e}, retrying...")
                        await asyncio.sleep(2)  # 等待2秒后重试
                        continue
                    else:
                        # 最后一次重试也失败，记录错误
                        print(f"[{index}] All {self.max_retries} judge attempts failed: {e}")
                        return {
                            "index": index,
                            "answer": "Incorrect",
                            "think": f"评估失败（重试{self.max_retries}次）: {str(e)}",
                            "judge_status": "error"
                        }

    async def evaluate_all(
        self,
        input_file: str,
        output_file: str
    ):
        """
        评估所有样本

        Args:
            input_file: 输入的inference_results.jsonl文件
            output_file: 输出的evaluated_results.jsonl文件
        """
        # 1. 加载推理结果
        print(f"📂 加载推理结果: {input_file}")
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        print(f"✅ 加载完成，共 {len(samples)} 条样本")

        # 2. 检查断点续传
        completed_indices = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    completed_indices.add(result['index'])
            print(f"🔄 检测到已评估 {len(completed_indices)} 条样本，继续处理剩余样本")

        # 3. 过滤未评估的样本
        samples_to_judge = [s for s in samples if s['index'] not in completed_indices]
        print(f"📊 待评估样本数: {len(samples_to_judge)}")

        if not samples_to_judge:
            print("✅ 所有样本已评估完成！")
            return

        # 4. 创建任务
        tasks = [
            self.judge_single_sample(sample)
            for sample in samples_to_judge
        ]

        # 5. 执行并发评估
        print(f"\n🚀 开始评估...\n")

        results = []
        pbar = async_tqdm(total=len(tasks), desc="LLM-as-Judge")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for coro in asyncio.as_completed(tasks):
            judge_result = await coro

            # 合并原样本和Judge结果
            original_sample = next(s for s in samples_to_judge if s['index'] == judge_result['index'])
            combined_result = {**original_sample, **judge_result}

            results.append(combined_result)

            # 实时保存
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(combined_result, ensure_ascii=False) + '\n')

            pbar.update(1)

        pbar.close()

        # 6. 统计结果
        correct = sum(1 for r in results if r.get('answer') == 'Correct')
        incorrect = sum(1 for r in results if r.get('answer') == 'Incorrect')
        accuracy = correct / len(results) * 100 if results else 0

        print(f"\n✅ 评估完成！")
        print(f"   总样本数: {len(results)}")
        print(f"   Correct: {correct}")
        print(f"   Incorrect: {incorrect}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   输出文件: {output_file}")


# ==================== 主函数 ====================

async def main():
    parser = argparse.ArgumentParser(description="Setting B: Guided Search - LLM-as-Judge 评估")
    parser.add_argument('--input', required=True, help='输入的inference_results.jsonl文件')
    parser.add_argument('--output', required=True, help='输出的evaluated_results.jsonl文件')
    parser.add_argument('--judge-model', default='models', help='Judge模型名称（默认models）')
    parser.add_argument('--judge-api-base', default='http://localhost:8000/v1', help='Judge API地址')
    parser.add_argument('--judge-api-key', default='EMPTY', help='Judge API密钥（默认EMPTY）')
    parser.add_argument('--max-concurrent', type=int, default=32, help='最大并发数（默认32）')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Judge最大token数（默认1024）')
    parser.add_argument('--temperature', type=float, default=0.0, help='Judge采样温度（默认0.0）')
    parser.add_argument('--max-retries', type=int, default=3, help='失败重试次数（默认3）')
    parser.add_argument('--enable-thinking', action='store_true', default=True, help='启用thinking模式（默认True）')
    parser.add_argument('--disable-thinking', dest='enable_thinking', action='store_false', help='禁用thinking模式')

    args = parser.parse_args()

    print("=" * 80)
    print("  Setting B: Guided Search (Few-shot) - LLM-as-Judge 评估")
    print("=" * 80)
    print()

    # 创建Judge引擎
    engine = SettingBJudgeEngine(
        args.judge_model,
        args.judge_api_base,
        args.judge_api_key,
        args.max_concurrent,
        args.max_tokens,
        args.temperature,
        args.max_retries,
        args.enable_thinking
    )

    # 运行评估
    await engine.evaluate_all(args.input, args.output)


if __name__ == '__main__':
    asyncio.run(main())
