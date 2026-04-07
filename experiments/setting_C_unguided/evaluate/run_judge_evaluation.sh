#!/bin/bash
# run_judge_evaluation.sh - Setting C Unguided Search (Zero-shot) LLM-as-Judge 重新评估

set -e

echo "========================================================================"
echo "  Setting C: Unguided Search (Zero-shot) - LLM-as-Judge 重新评估"
echo "========================================================================"
echo ""

# ================= 配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# Judge模型配置
JUDGE_MODEL="models"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:8000/v1}"

# 数据配置（修改这里以评估不同模型的结果）
# 可以设置为模型名称，也可以直接指定完整路径
MODEL="Qwen3-32B_round1"

# 自动检测输入文件路径
if [ -f "$MODEL" ]; then
    # 如果MODEL是完整路径且文件存在
    INPUT_JSONL="$MODEL"
    # 从路径中提取模型名称
    MODEL_NAME=$(basename $(dirname "$MODEL"))
    OUTPUT_DIR=$(dirname "$MODEL")
else
    # 如果MODEL是模型名称，构建路径
    INPUT_JSONL="setting_c_results/$MODEL/evaluated_results.jsonl"
    MODEL_NAME="$MODEL"
    OUTPUT_DIR="setting_c_results/$MODEL"
fi

OUTPUT_JSONL="$OUTPUT_DIR/re_evaluated_results.jsonl"

# 并发配置
MAX_CONCURRENT=32              # Judge并发数
MAX_RETRIES=3                  # 失败重试次数

# Judge生成参数
MAX_TOKENS=8192                # Judge最大token数
TEMPERATURE=0.8                # Judge温度

# 其他选项
FORCE_REEVALUATE=false         # 是否强制重新评估所有样本

# ================= 显示配置 =================

echo "📋 配置信息:"
echo "  模型名称: $MODEL_NAME"
echo "  Judge模型: $JUDGE_MODEL"
echo "  Judge API: $JUDGE_API_BASE"
echo "  输入文件: $INPUT_JSONL"
echo "  输出文件: $OUTPUT_JSONL"
echo "  最大并发: $MAX_CONCURRENT"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Max Retries: $MAX_RETRIES"
echo "  强制重新评估: $FORCE_REEVALUATE"
echo ""

# ================= 检查输入文件 =================

if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ 错误: 输入文件不存在: $INPUT_JSONL"
    echo ""
    echo "提示: 请检查以下内容："
    echo "  1. 模型名称是否正确（例如：Qwen3-32B_round1）"
    echo "  2. 文件是否在 setting_c_results/$MODEL/ 目录下"
    echo "  3. 文件名是否为 evaluated_results.jsonl"
    exit 1
fi

# ================= 确认开始 =================

echo "📊 输入文件样本数: $(wc -l < "$INPUT_JSONL") 条"
echo ""

read -p "确认开始评估？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ================= 运行评估 =================

echo ""
echo "🚀 开始评估..."
echo "----------------------------------------"

# 构建命令参数
CMD_ARGS=(
    --input "$INPUT_JSONL"
    --output "$OUTPUT_JSONL"
    --judge-model "$JUDGE_MODEL"
    --judge-api-base "$JUDGE_API_BASE"
    --max-concurrent "$MAX_CONCURRENT"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --max-retries "$MAX_RETRIES"
)

# 如果启用强制重新评估
if [ "$FORCE_REEVALUATE" = true ]; then
    CMD_ARGS+=(--force-reevaluate)
fi

python experiments/setting_C_unguided/evaluate/llm-as-judge.py "${CMD_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 评估完成！"
    echo "📁 结果文件: $OUTPUT_JSONL"
    echo ""

    # 统计结果
    echo "📊 评估统计:"
    python -c "
import json

results = []
with open('$OUTPUT_JSONL', 'r') as f:
    for line in f:
        results.append(json.loads(line.strip()))

correct = sum(1 for r in results if r.get('answer') == 'Correct')
incorrect = sum(1 for r in results if r.get('answer') == 'Incorrect')
total = len(results)
accuracy = correct / total * 100 if total > 0 else 0

print(f'  总样本数: {total}')
print(f'  Correct: {correct} ({correct/total*100:.2f}%)')
print(f'  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)')
print(f'  Accuracy: {accuracy:.2f}%')
"
else
    echo ""
    echo "❌ 评估失败"
    exit 1
fi
