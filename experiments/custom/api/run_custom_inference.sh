#!/bin/bash
# run_custom_inference.sh - Custom Prompt Evaluation (API Version)

set -e

echo "========================================================================"
echo "  Custom Prompt Evaluation - API推理"
echo "========================================================================"
echo ""

# ================= 环境配置 =================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# ================= 可配置参数 =================

# 数据路径
DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/custom_results"

# API 配置默认值
PROVIDER="custom"
MODEL_NAME="gpt-4"
BASE_URL="https://api.openai.com/v1"
QPS=10
MAX_CONCURRENT=32
MAX_TOKENS=4096
TEMPERATURE=0.7

# ================= Prompt 配置 =================

echo "请提供自定义 Prompt 文件:"
echo ""
read -p "请输入 Prompt 文件路径: " PROMPT_FILE_INPUT

# 判断是否为绝对路径
if [[ "$PROMPT_FILE_INPUT" = /* ]]; then
    PROMPT_FILE="$PROMPT_FILE_INPUT"
else
    PROMPT_FILE="${SCRIPT_DIR}/../prompts/$PROMPT_FILE_INPUT"
fi

# 检查文件
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt 文件不存在: $PROMPT_FILE"
    exit 1
fi

echo "Prompt 文件: $PROMPT_FILE"
PROMPT_NAME=$(basename "$PROMPT_FILE" .py)

# ================= 模型配置 =================

echo ""
echo "请输入模型配置:"
read -p "API Provider [openai/azure/custom, 默认: custom]: " INPUT_PROVIDER
PROVIDER=${INPUT_PROVIDER:-custom}

read -p "模型名称 [默认: gpt-4]: " INPUT_MODEL
MODEL_NAME=${INPUT_MODEL:-gpt-4}

read -p "API Base URL [默认: https://api.openai.com/v1]: " INPUT_URL
BASE_URL=${INPUT_URL:-https://api.openai.com/v1}

read -p "模型标识符（用于结果目录）[默认: default]: " MODEL_ID
MODEL_ID=${MODEL_ID:-default}

# ================= 工具调用模式 =================

echo ""
echo "请选择工具调用模式:"
echo "  text   - 文本解析模式（与 vLLM 对齐，用于公平对比，默认）"
echo "  native - 原生 function calling（商业 API 推荐，模型能力上限评估）"
read -p "工具调用模式 [text/native, 默认: text]: " INPUT_TOOL_MODE
TOOL_MODE=${INPUT_TOOL_MODE:-text}

PARSER_ARG=""
if [ "$TOOL_MODE" = "text" ]; then
    echo ""
    echo "请选择 tool call 文本解析器（仅 text 模式需要）:"
    echo "  default  - Hermes / NousResearch / Qwen3（默认）"
    echo "  deepseek - DeepSeek-V2 / V3 / R1"
    echo "  glm4     - GLM-4 / GLM-Z1"
    echo "  minimax  - MiniMax-M2.5"
    echo "  qwen35   - Qwen3.5"
    echo "  kimi_k2  - Kimi-K2（单工具兜底模式）"
    read -p "Parser [默认: default]: " INPUT_PARSER
    PARSER=${INPUT_PARSER:-default}
    PARSER_ARG="--parser $PARSER"
fi

# 文件路径
INPUT_JSONL="${DATA_DIR}/mpw_bench_full.jsonl"
OUTPUT_JSONL="${RESULTS_DIR}/${PROMPT_NAME}/${MODEL_ID}/api_results.jsonl"

# ================= 显示配置 =================

echo ""
echo "配置信息:"
echo "  输入文件: $INPUT_JSONL"
echo "  输出文件: $OUTPUT_JSONL"
echo ""
echo "  API 配置:"
echo "    - Provider: $PROVIDER"
echo "    - Model: $MODEL_NAME"
echo "    - Base URL: $BASE_URL"
echo "    - QPS: $QPS"
echo "    - Max Concurrent: $MAX_CONCURRENT"
echo ""
echo "  Prompt 配置:"
echo "    - Prompt 文件: $PROMPT_FILE"
echo "    - Prompt 名称: $PROMPT_NAME"
echo ""
echo "  工具调用:"
echo "    - Tool Mode: $TOOL_MODE"
if [ "$TOOL_MODE" = "text" ]; then
    echo "    - Parser: $PARSER"
fi
echo ""

# ================= 确认开始 =================

read -p "确认开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ================= 运行推理 =================

echo ""
echo "开始推理..."
echo "----------------------------------------"

python "${SCRIPT_DIR}/async_inference.py" \
    --provider "$PROVIDER" \
    --model "$MODEL_NAME" \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL" \
    --prompt-file "$PROMPT_FILE" \
    --base-url "$BASE_URL" \
    --tool-mode "$TOOL_MODE" \
    $PARSER_ARG \
    --qps "$QPS" \
    --max-concurrent-turns "$MAX_CONCURRENT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --resume

if [ $? -eq 0 ]; then
    echo ""
    echo "推理完成！"
    echo ""
    echo "结果文件: $OUTPUT_JSONL"
    echo ""
else
    echo ""
    echo "推理失败"
    exit 1
fi
