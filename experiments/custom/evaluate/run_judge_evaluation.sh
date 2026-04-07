#!/bin/bash
# run_judge_evaluation.sh - LLM-as-Judge 评估脚本

set -e

echo "========================================================================"
echo "  Custom Prompt Evaluation - LLM-as-Judge"
echo "========================================================================"
echo ""

# ================= 环境配置 =================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# ================= 可配置参数 =================

# 输入/输出文件
INPUT_DIR="${PROJECT_ROOT}/custom_results"
OUTPUT_DIR="${PROJECT_ROOT}/custom_results"

# Judge 配置
JUDGE_MODEL="models"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:8000/v1}"
MAX_CONCURRENT=32
MAX_RETRIES=3

# ================= 选择要评估的模型 =================

echo "📁 可用的结果目录:"
echo ""

# 列出所有 prompt 目录
prompt_dirs=()
if [ -d "$INPUT_DIR" ]; then
    for prompt_dir in "$INPUT_DIR"/*; do
        if [ -d "$prompt_dir" ]; then
            prompt_name=$(basename "$prompt_dir")
            echo "  Prompt: $prompt_name"
            # 列出该 prompt 下的模型
            for model_dir in "$prompt_dir"/*; do
                if [ -d "$model_dir" ]; then
                    model_name=$(basename "$model_dir")
                    result_file="$model_dir/inference_results.jsonl"
                    if [ -f "$result_file" ]; then
                        result_count=$(wc -l < "$result_file")
                        echo "    - $model_name ($result_count samples)"
                    fi
                fi
            done
            echo ""
            prompt_dirs+=("$prompt_name")
        fi
    done
fi

if [ ${#prompt_dirs[@]} -eq 0 ]; then
    echo "❌ 没有找到推理结果"
    echo "请先运行推理脚本"
    exit 1
fi

# 选择 prompt
read -p "请输入 Prompt 名称: " SELECTED_PROMPT

# 选择模型
read -p "请输入模型名称: " SELECTED_MODEL

# 设置文件路径
INPUT_FILE="${INPUT_DIR}/${SELECTED_PROMPT}/${SELECTED_MODEL}/inference_results.jsonl"
OUTPUT_FILE="${OUTPUT_DIR}/${SELECTED_PROMPT}/${SELECTED_MODEL}/evaluated_results.jsonl"

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 输入文件不存在: $INPUT_FILE"
    exit 1
fi

echo ""
echo "📋 配置信息:"
echo "  输入文件: $INPUT_FILE"
echo "  输出文件: $OUTPUT_FILE"
echo "  Judge Model: $JUDGE_MODEL"
echo "  Judge API: $JUDGE_API_BASE"
echo ""

# ================= 确认开始 =================

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

python "${SCRIPT_DIR}/llm-as-judge.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-base "$JUDGE_API_BASE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --max-retries "$MAX_RETRIES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 评估完成！"
    echo ""
    echo "📁 结果文件: $OUTPUT_FILE"
    echo ""
else
    echo ""
    echo "❌ 评估失败"
    exit 1
fi
