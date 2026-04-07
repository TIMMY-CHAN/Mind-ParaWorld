#!/bin/bash
# run_oracle_inference.sh - Setting A Oracle-Facts QA (API版本)

set -e

echo "========================================================================"
echo "  Setting A: Oracle-Facts QA - API推理"
echo "========================================================================"
echo ""

# ================= 配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# API配置
API_PROVIDER=${1:-openai}    # API提供商（默认openai）
MODEL_NAME=${2:-gpt-4}       # 模型名称（默认gpt-4）

# 数据配置
INPUT_JSONL="data/mpw_bench_full.jsonl"
OUTPUT_DIR="results/setting_A_${API_PROVIDER}_${MODEL_NAME}"
OUTPUT_JSONL="$OUTPUT_DIR/inference_results.jsonl"

# 并发配置
QPS=2                          # API速率限制
MAX_CONCURRENT=32              # 最大并发样本数

# 生成参数
MAX_TOKENS=2048                # 最大生成token数
TEMPERATURE=0.0                # 采样温度（0.0=贪婪解码）

# ================= 显示配置 =================

echo "📋 配置信息:"
echo "  API提供商: $API_PROVIDER"
echo "  模型: $MODEL_NAME"
echo "  输入文件: $INPUT_JSONL"
echo "  输出文件: $OUTPUT_JSONL"
echo "  QPS限制: $QPS"
echo "  最大并发: $MAX_CONCURRENT"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo ""

# ================= 创建目录 =================

mkdir -p "$OUTPUT_DIR"

# ================= 运行推理 =================

echo "🚀 开始推理..."
echo "----------------------------------------"

python experiments/setting_A_oracle/api/async_inference_oracle_api.py \
    --provider "$API_PROVIDER" \
    --model "$MODEL_NAME" \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL" \
    --qps "$QPS" \
    --max-concurrent "$MAX_CONCURRENT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推理完成！"
    echo "📁 结果文件: $OUTPUT_JSONL"
    echo "📊 处理数量: $(wc -l < "$OUTPUT_JSONL") 条"
else
    echo ""
    echo "❌ 推理失败"
    exit 1
fi

