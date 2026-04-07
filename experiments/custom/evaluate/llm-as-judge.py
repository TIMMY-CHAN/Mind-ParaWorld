#!/usr/bin/env python3
"""
llm-as-judge.py - LLM-as-Judge 评估脚本（Custom Prompt 通用版）

使用 LLM 作为评判者，对推理结果进行自动评估。

使用方法：
    python llm-as-judge.py \
        --input results.jsonl \
        --output evaluated_results.jsonl \
        --judge-model gpt-4 \
        --judge-api-base http://localhost:8000/v1
"""

import argparse
import asyncio
import json
import os
import re
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import httpx


# ================= 评估 Prompt =================

JUDGE_SYSTEM_PROMPT = """你是一个专业的答案评估专家。你的任务是判断模型的预测答案是否与标准答案一致。

评估标准：
1. 如果预测答案与标准答案在语义上一致，判定为"Correct"
2. 如果预测答案与标准答案不一致、缺少关键信息、或完全错误，判定为"Incorrect"

注意：
- 对于数值类答案，允许合理的误差范围（如百分比误差在5%以内）
- 对于时间类答案，格式可以不同但时间点应一致
- 对于实体名称，别名或同义词可视为正确

请严格按照以下格式输出：
<judgment>Correct</judgment> 或 <judgment>Incorrect</judgment>
<reason>判断理由</reason>
"""

JUDGE_USER_PROMPT_TEMPLATE = """请评估以下问答结果：

问题：
{question}

标准答案：
{ground_truth}

模型预测答案：
{prediction}

请判断模型预测是否正确，并给出理由。
"""


# ================= 评估引擎 =================

class LLMAsJudge:
    """LLM-as-Judge 评估引擎"""

    def __init__(
        self,
        judge_model: str,
        judge_api_base: str,
        judge_api_key: str = "EMPTY",
        max_concurrent: int = 32,
        max_retries: int = 3,
    ):
        """
        初始化评估引擎

        Args:
            judge_model: Judge 模型名称
            judge_api_base: Judge API 地址
            judge_api_key: API 密钥
            max_concurrent: 最大并发数
            max_retries: 最大重试次数
        """
        timeout_config = httpx.Timeout(connect=100.0, read=300.0, write=100.0, pool=100.0)
        self.client = AsyncOpenAI(base_url=judge_api_base, api_key=judge_api_key, timeout=timeout_config)
        self.judge_model = judge_model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries

        print(f"[Judge] Initialized:")
        print(f"   - Judge Model: {judge_model}")
        print(f"   - API Base: {judge_api_base}")
        print(f"   - Max Concurrent: {max_concurrent}")
        print(f"   - Max Retries: {max_retries}")

    def _extract_judgment(self, response: str) -> Dict[str, str]:
        """从响应中提取判断结果"""
        judgment = "Unknown"
        reason = ""

        # 提取判断
        judgment_match = re.search(r'<judgment>(.*?)</judgment>', response, re.DOTALL | re.IGNORECASE)
        if judgment_match:
            judgment = judgment_match.group(1).strip()

        # 提取理由
        reason_match = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip()

        return {"judgment": judgment, "reason": reason}

    async def judge_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个样本"""
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    # 提取字段
                    question = ""
                    ground_truth = sample.get("ground_truth", "")
                    prediction = sample.get("final_answer", "")

                    # 从 messages 中提取问题
                    messages = sample.get("messages", []) or sample.get("full_messages", [])
                    for msg in messages:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                question = content
                                break

                    # 构建评估 Prompt
                    user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction
                    )

                    # 调用 Judge API
                    response = await self.client.chat.completions.create(
                        model=self.judge_model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=1024
                    )

                    judge_response = response.choices[0].message.content
                    result = self._extract_judgment(judge_response)

                    # 更新样本
                    sample["judge_judgment"] = result["judgment"]
                    sample["judge_reason"] = result["reason"]
                    sample["judge_response"] = judge_response
                    sample["judge_status"] = "success"

                    return sample

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        sample["judge_status"] = "error"
                        sample["judge_error"] = str(e)
                        return sample

            return sample

    async def judge_all(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """评估所有样本"""
        tasks = [self.judge_single(sample) for sample in samples]
        results = await tqdm.gather(*tasks, desc="Judging")

        # 统计结果
        correct = sum(1 for r in results if r.get("judge_judgment", "").lower() == "correct")
        incorrect = sum(1 for r in results if r.get("judge_judgment", "").lower() == "incorrect")
        error = sum(1 for r in results if r.get("judge_status") == "error")

        print(f"\n[Judge] Results:")
        print(f"   - Correct: {correct}")
        print(f"   - Incorrect: {incorrect}")
        print(f"   - Error: {error}")
        if correct + incorrect > 0:
            accuracy = correct / (correct + incorrect) * 100
            print(f"   - Accuracy: {accuracy:.2f}%")

        return results


# ================= 主函数 =================

async def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge Evaluation')

    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--judge-model', type=str, default="gpt-4", help='Judge model name')
    parser.add_argument('--judge-api-base', type=str, required=True, help='Judge API base URL')
    parser.add_argument('--judge-api-key', type=str, default="EMPTY", help='Judge API key')
    parser.add_argument('--max-concurrent', type=int, default=32, help='Max concurrent requests')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries per sample')

    args = parser.parse_args()

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

    # 读取输入数据
    samples = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in processed_indices:
                continue
            try:
                data = json.loads(line)
                if 'index' not in data:
                    data['index'] = idx
                samples.append(data)
            except:
                continue

    print(f"Samples to evaluate: {len(samples)}")

    if not samples:
        print("No samples to evaluate!")
        return

    # 初始化 Judge
    judge = LLMAsJudge(
        judge_model=args.judge_model,
        judge_api_base=args.judge_api_base,
        judge_api_key=args.judge_api_key,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries
    )

    # 运行评估
    results = await judge.judge_all(samples)

    # 保存结果
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
