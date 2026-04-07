#!/bin/bash
# run_custom_inference.sh - Custom Prompt Evaluation - vLLM版本

set -e

echo "========================================================================"
echo "  Custom Prompt Evaluation - vLLM推理"
echo "========================================================================"
echo ""

# ================= 环境配置 =================

# 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# ================= 可配置参数 =================

# 数据路径
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/custom_results"

# Search Agent 配置（vLLM 部署的推理模型）
MODEL_NAME="models"
API_BASE="${API_BASE:-http://localhost:8000/v1}"

# 世界模型配置（支持多节点，逗号分隔）
WORLD_MODEL_ENDPOINTS="${WORLD_MODEL_ENDPOINTS:-http://localhost:8000/v1}"

# 推理配置
MAX_CONCURRENT_TURNS=64
MAX_TURNS_PER_SAMPLE=32
MAX_CONTEXT_CHARS=128000
ENABLE_THINKING=false

# ================= Prompt 配置 =================

echo "📝 请提供自定义 Prompt 文件:"
echo ""
echo "  方式1: 输入 prompt 文件路径（相对于 prompts/ 目录）"
echo "         例如: my_prompt.py (将使用 prompts/my_prompt.py)"
echo ""
echo "  方式2: 输入完整路径"
echo "         例如: /path/to/my_prompt.py"
echo ""
read -p "请输入 Prompt 文件路径: " PROMPT_FILE_INPUT

# 判断是否为绝对路径
if [[ "$PROMPT_FILE_INPUT" = /* ]]; then
    PROMPT_FILE="$PROMPT_FILE_INPUT"
else
    PROMPT_FILE="${SCRIPT_DIR}/../prompts/$PROMPT_FILE_INPUT"
fi

# 检查文件是否存在
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Prompt 文件不存在: $PROMPT_FILE"
    echo ""
    echo "提示: 请确保文件存在，且定义了 system_prompt 变量"
    exit 1
fi

echo "✅ Prompt 文件: $PROMPT_FILE"

# 提取 prompt 名称（用于结果目录）
PROMPT_NAME=$(basename "$PROMPT_FILE" .py)
echo ""

# ================= 模型配置 =================

echo ""
echo "🔧 请输入模型名称（用于结果目录命名）:"
read -p "模型名称 [默认: default]: " MODEL_ID
MODEL_ID=${MODEL_ID:-default}

# 文件命名
INPUT_JSONL="${DATA_DIR}/mpw_bench_full.jsonl"
OUTPUT_JSONL="${RESULTS_DIR}/${PROMPT_NAME}/${MODEL_ID}/inference_results.jsonl"

# ================= 显示配置 =================

echo ""
echo "📋 配置信息:"
echo "  输入文件: $INPUT_JSONL"
echo "  输出文件: $OUTPUT_JSONL"
echo ""
echo "  推理配置:"
echo "    - 模型: $MODEL_NAME"
echo "    - Search Agent API: $API_BASE"
echo "    - 世界模型节点: $WORLD_MODEL_ENDPOINTS"
echo "    - Turn 并发数: $MAX_CONCURRENT_TURNS"
echo "    - 最大轮数/样本: $MAX_TURNS_PER_SAMPLE"
echo "    - 上下文窗口: $MAX_CONTEXT_CHARS 字符"
echo "    - Enable Thinking: $ENABLE_THINKING"
echo ""
echo "  Prompt 配置:"
echo "    - Prompt 文件: $PROMPT_FILE"
echo "    - Prompt 名称: $PROMPT_NAME"
echo ""

# ================= 确认开始 =================

read -p "确认开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ================= 步骤 1: 数据准备 =================

echo ""
echo "========================================================================"
echo "  步骤 1/2: 数据准备"
echo "========================================================================"
echo ""

# 创建目录
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$OUTPUT_JSONL")"

# 检查数据文件
if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ 数据文件不存在: $INPUT_JSONL"
    echo "请先准备数据文件"
    exit 1
else
    echo "✅ 数据文件已存在: $INPUT_JSONL"

    # 显示统计
    echo ""
    echo "📊 数据统计:"
    total_count=$(wc -l < "$INPUT_JSONL")
    echo "   总数据量: $total_count 条"
fi

echo ""

# ================= 步骤 2: 运行推理 =================

echo "========================================================================"
echo "  步骤 2/2: 运行推理"
echo "========================================================================"
echo ""

# 检查已处理数量
processed_count=0
if [ -f "$OUTPUT_JSONL" ]; then
    processed_count=$(wc -l < "$OUTPUT_JSONL")
    echo "✅ 已处理 $processed_count 条数据"
    echo ""
fi

# 计算剩余数量
total_count=$(wc -l < "$INPUT_JSONL")
remaining_count=$((total_count - processed_count))

if [ $remaining_count -eq 0 ]; then
    echo "✅ 所有数据已处理完成！"
    echo ""
    echo "📁 结果文件: $OUTPUT_JSONL"
    exit 0
fi

echo "📊 推理任务:"
echo "   - 总数据量: $total_count 条"
echo "   - 已处理: $processed_count 条"
echo "   - 待处理: $remaining_count 条"
echo ""

read -p "开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始推理..."
echo "----------------------------------------"

# 导出世界模型节点配置
export WORLD_MODEL_ENDPOINTS

# 构建命令参数
PYTHON_CMD="python ${SCRIPT_DIR}/async_inference.py \
    --input \"$INPUT_JSONL\" \
    --output \"$OUTPUT_JSONL\" \
    --prompt-file \"$PROMPT_FILE\" \
    --model \"$MODEL_NAME\" \
    --api-base \"$API_BASE\" \
    --max-concurrent-turns \"$MAX_CONCURRENT_TURNS\" \
    --max-turns-per-sample \"$MAX_TURNS_PER_SAMPLE\" \
    --max-context-chars \"$MAX_CONTEXT_CHARS\""

# 如果启用thinking，添加参数
if [ "$ENABLE_THINKING" = "true" ]; then
    PYTHON_CMD="$PYTHON_CMD --enable-thinking"
fi

# 执行命令
eval $PYTHON_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推理完成！"
    echo ""
    echo "📁 结果文件: $OUTPUT_JSONL"
    echo "📊 处理数量: $(wc -l < "$OUTPUT_JSONL") 条"
    echo ""
    echo "========================================================================"
    echo "  推理完成！"
    echo "========================================================================"
    echo ""
    echo "📝 后续步骤:"
    echo "   1. 使用 evaluate/llm-as-judge.py 进行评估"
    echo "   2. 对比不同 Prompt 的效果"
    echo ""
else
    echo ""
    echo "❌ 推理失败"
    exit 1
fi
