#!/bin/bash
# run_vllm_oracle.sh - Setting A Oracle-Facts QA (vLLM版本)

set -e

echo "========================================================================"
echo "  Setting A: Oracle-Facts QA - vLLM推理"
echo "========================================================================"
echo ""

# ================= 配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# vLLM配置
MODEL_NAME="models"
# API_BASE="http://10.72.8.1:8000/v1"   # MindWatcher
# API_BASE="http://10.72.8.56:8000/v1"    # m2_1
API_BASE="${API_BASE:-http://localhost:8000/v1}"
# API_BASE="http://10.72.0.129:8000/v1" # qwen3 32B
# 数据配置
INPUT_JSONL="data/mpw_bench_full.jsonl"
OUTPUT_DIR="setting_A_results/youtu-agent"
OUTPUT_JSONL="$OUTPUT_DIR/inference_results.jsonl"

# 并发配置
MAX_CONCURRENT=64              # 最大并发样本数（因为每个样本只有1轮）

# 生成参数
MAX_TOKENS=8192                # 最大生成token数
TEMPERATURE=0.8                # 采样温度（0.0=贪婪解码）
ENABLE_THINKING=false           # 启用thinking模式（Qwen等模型需要，设置为false关闭）
MAX_RETRIES=3                  # 失败重试次数

# ================= 显示配置 =================

echo "📋 配置信息:"
echo "  vLLM地址: $API_BASE"
echo "  模型: $MODEL_NAME"
echo "  输入文件: $INPUT_JSONL"
echo "  输出文件: $OUTPUT_JSONL"
echo "  最大并发: $MAX_CONCURRENT"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo "  Enable Thinking: $ENABLE_THINKING"
echo "  Max Retries: $MAX_RETRIES"
echo ""

# ================= 创建目录 =================

mkdir -p "$OUTPUT_DIR"

# ================= 确认开始 =================

read -p "确认开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ================= 运行推理 =================

echo ""
echo "🚀 开始推理..."
echo "----------------------------------------"

# 构建命令参数
PYTHON_CMD="python experiments/setting_A_oracle/vllm/async_inference.py \
    --input \"$INPUT_JSONL\" \
    --output \"$OUTPUT_JSONL\" \
    --model \"$MODEL_NAME\" \
    --api-base \"$API_BASE\" \
    --max-concurrent \"$MAX_CONCURRENT\" \
    --max-tokens \"$MAX_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --max-retries \"$MAX_RETRIES\""

# 如果启用thinking，添加参数
if [ "$ENABLE_THINKING" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --enable-thinking"
fi

# 执行命令
eval $PYTHON_CMD

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

