#!/usr/bin/env python3
"""
async_inference_api_fewshot.py - Few-shot Decomposition 消融实验版本

核心改动：
1. 在 system prompt 中添加 Few-shot Query Decomposition 示例
2. 明确展示"什么样的查询是好的"vs"什么样的查询是差的"
3. 其他逻辑完全保持不变

使用方法：
    python async_inference_api_fewshot.py \
        --provider openai \
        --model Qwen/Qwen2-VL-72B-Instruct-AWQ \
        --input data/test.jsonl \
        --output results/fewshot_test/inference_results.jsonl \
        --qps 10 \
        --max-concurrent-turns 100

对比实验：
    # 基线（原始 prompt）
    ./run_api_inference_standard.sh openai model_name

    # Few-shot（本脚本）
    python async_inference_api_fewshot.py --provider openai --model model_name ...
"""

# ==================== 导入完全相同的代码 ====================
import sys
import os

# 将原始脚本的路径添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始脚本的所有内容
from async_inference_api import *

# ==================== 唯一的改动：Few-shot System Prompt ====================

INSTRUCTION_PROMPT_SYSTEM_FEWSHOT = """
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

---

# 查询分解指南（Query Decomposition Guide）

**重要**：复杂问题需要分解为多个简单查询。以下是好的查询和差的查询的对比示例：

## 示例 1：比较类问题

❌ **差的查询**（过于复杂，搜索引擎难以理解）：
```
比较 2022-23赛季 尤文图斯和那不勒斯的客场进球数哪个更多
```
**问题**：一次查询包含多个实体和多个维度，搜索引擎无法准确匹配。

✅ **好的查询**（分解为原子查询）：
```
步骤1: 搜索 "2022-23赛季 尤文图斯 客场进球数"
步骤2: 搜索 "2022-23赛季 那不勒斯 客场进球数"
步骤3: 比较两个数值
```
**原因**：每个查询只关注一个实体的一个属性，更容易命中准确信息。

---

## 示例 2：时间差计算问题

❌ **差的查询**：
```
郑伊健和郁可唯在深圳的演唱会开唱时间相差多少分钟
```
**问题**：搜索引擎很难直接返回"时间差"，需要先获取各自的时间。

✅ **好的查询**：
```
步骤1: 搜索 "郑伊健 深圳演唱会 开唱时间"
步骤2: 搜索 "郁可唯 深圳演唱会 开唱时间"
步骤3: 计算时间差（自己完成计算）
```
**原因**：先收集事实，再自行计算，而非期待搜索引擎直接给出答案。

---

## 示例 3：条件筛选问题

❌ **差的查询**：
```
2026年9月之后在深圳举办演唱会的歌手中哪位粉丝最多
```
**问题**：包含时间筛选、地点筛选、粉丝比较，搜索引擎难以一次性处理。

✅ **好的查询**：
```
步骤1: 搜索 "2026年9月后 深圳演唱会 歌手列表"
步骤2: 对每位歌手搜索 "歌手名 粉丝数量"
步骤3: 比较并找出粉丝最多的歌手
```
**原因**：逐步筛选和收集信息，而非一次性查询复杂条件。

---

## 示例 4：多维度比较问题

❌ **差的查询**：
```
比较巴塞罗那和皇家马德里在2023-24赛季的主场胜率和平均进球数
```
**问题**：一次查询包含2个队伍 × 2个指标 = 4个维度。

✅ **好的查询**：
```
步骤1: 搜索 "巴塞罗那 2023-24赛季 主场胜率"
步骤2: 搜索 "巴塞罗那 2023-24赛季 主场平均进球数"
步骤3: 搜索 "皇家马德里 2023-24赛季 主场胜率"
步骤4: 搜索 "皇家马德里 2023-24赛季 主场平均进球数"
步骤5: 整理和比较数据
```
**原因**：每个查询只关注一个实体的一个属性，确保信息准确。

---

## 核心原则

1. **原子化查询**：一个查询只关注一个实体、一个属性、一个时间点
2. **先收集后计算**：不要期待搜索引擎帮你做计算或比较，先获取原始数据
3. **明确实体**：使用具体名称，避免"他们"、"哪个"等模糊指代
4. **分步推理**：复杂问题 = 多个简单查询 + 自己的推理

---

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


# ==================== 重写关键函数，使用新 Prompt ====================

class AsyncInferenceEngine_FewShot(AsyncInferenceEngine):
    """Few-shot Decomposition 版本的推理引擎"""

    def initialize_sample(self, line_idx: int, data: Dict) -> Optional[SampleState]:
        """
        初始化单个样本（重写以使用 Few-shot prompt）
        """
        try:
            # 提取字段
            raw_messages = data.get('prompt', []) or data.get('messages', [])
            ground_truth = data.get('answer', '')
            extra_info = data.get('extra_info', {})

            # 选择世界模型 Agent
            world_truth_info = extra_info.get('world_truth_info', {})
            category = extra_info.get('category', 'default')
            agent = self.choose_world_model_agent(world_truth_info, category)

            # 准备消息（使用 Few-shot prompt）
            chat_messages = raw_messages.copy()
            system_prompt = data.get('system', INSTRUCTION_PROMPT_SYSTEM_FEWSHOT)  # ← 唯一改动
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


# ==================== 主程序入口 ====================

async def main():
    """
    主程序：使用 Few-shot Decomposition Prompt
    """
    parser = argparse.ArgumentParser(description="Few-shot Decomposition Ablation Study")

    # 基础参数（与原始脚本相同）
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "azure", "custom"],
                       help="API 提供商")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")

    # API 配置
    parser.add_argument("--api-key", type=str, default=None, help="API Key（可选，优先从环境变量读取）")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL (custom provider)")

    # 性能参数
    parser.add_argument("--qps", type=float, default=10, help="QPS 限制（默认 10）")
    parser.add_argument("--max-concurrent-turns", type=int, default=100,
                       help="最大并发 Turn 数（默认 100）")
    parser.add_argument("--max-turns-per-sample", type=int, default=32,
                       help="每个样本最大轮次（默认 32）")

    # 世界模型配置
    parser.add_argument("--world-model-provider", type=str, default="vllm",
                       choices=["openai", "azure", "custom", "vllm"],
                       help="世界模型 API 提供商")
    parser.add_argument("--world-model-name", type=str, default="qwen-plus",
                       help="世界模型名称")
    parser.add_argument("--world-model-vllm-url", type=str, default=None,
                       help="vLLM URL（多个节点用逗号分隔）")

    # 其他
    parser.add_argument("--resume", action="store_true", help="断点续传")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 打印配置
    print("=" * 80)
    print("FEW-SHOT DECOMPOSITION ABLATION STUDY")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"QPS: {args.qps}")
    print(f"Max Concurrent Turns: {args.max_concurrent_turns}")
    print(f"Max Turns per Sample: {args.max_turns_per_sample}")
    print(f"World Model Provider: {args.world_model_provider}")
    print(f"World Model: {args.world_model_name}")
    if args.world_model_vllm_url:
        print(f"World Model vLLM URL: {args.world_model_vllm_url}")
    print("=" * 80)
    print("\n🎯 使用 Few-shot Decomposition Prompt")
    print("   添加了 4 个查询分解示例\n")

    # 创建 Few-shot 推理引擎
    engine = AsyncInferenceEngine_FewShot(
        api_provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        qps=args.qps,
        max_concurrent_turns=args.max_concurrent_turns,
        max_turns_per_sample=args.max_turns_per_sample,
        world_model_provider=args.world_model_provider,
        world_model_name=args.world_model_name,
        world_model_vllm_url=args.world_model_vllm_url,
    )

    # 运行推理
    await engine.run_inference(args.input, args.output, resume=args.resume)


if __name__ == "__main__":
    asyncio.run(main())
