#!/bin/bash
# run_api_inference_standard.sh
# 标准 API 推理配置（统一实验条件）

set -e

echo "=========================================================================="
echo "  MPW-bench API 推理 - 统一实验配置"
echo "=========================================================================="
echo ""
# ==================== 统一实验配置 ====================

# 数据路径
DATA_DIR="data"
INPUT_JSONL="$DATA_DIR/mpw_bench_full.jsonl"

# Agent 配置（统一标准）
MAX_TURNS=32              # 所有闭源模型统一：最大 32 轮

# ==================== API 配置（根据提供商选择）====================

# 从命令行参数读取或使用默认值
API_PROVIDER=${1:-"openai"}
MODEL_NAME=${2:-"gemini-3-pro-preview"}
OUTPUT_DIR=${3:-"results/${API_PROVIDER}_${MODEL_NAME}"}

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

OUTPUT_JSONL="$OUTPUT_DIR/inference_results.jsonl"

# ==================== 显示配置 ====================

echo "📋 统一实验配置:"
echo "  Agent 配置:"
echo "    - 最大轮数: $MAX_TURNS 轮（统一标准）"
echo "    - 上下文管理: 由 API 自动处理（无手动截断）"
echo ""
echo "  API 配置:"
echo "    - 提供商: $API_PROVIDER"
echo "    - 模型: $MODEL_NAME"
echo ""
echo "  数据配置:"
echo "    - 输入文件: $INPUT_JSONL"
echo "    - 输出目录: $OUTPUT_DIR"
echo ""

# ==================== 根据不同 API 设置参数 ====================

case $API_PROVIDER in
    "openai")
        echo "🔧 配置 OpenAI API"
        QPS=5
        QPM=300
        MAX_CONCURRENT=50
        API_KEY_ENV="OPENAI_API_KEY"
        ;;

    "azure")
        echo "🔧 配置 Azure OpenAI API"
        QPS=5
        QPM=300
        MAX_CONCURRENT=50
        API_KEY_ENV="AZURE_OPENAI_KEY"
        BASE_URL=${AZURE_OPENAI_ENDPOINT:-""}
        ;;

    *)
        echo "❌ 不支持的 API 提供商: $API_PROVIDER"
        echo "支持的提供商: openai, azure"
        exit 1
        ;;
esac

echo "  并发配置:"
echo "    - QPS: $QPS"
echo "    - QPM: $QPM"
echo "    - 最大并发 Turn 数: $MAX_CONCURRENT"
echo ""

# ==================== 检查 API Key ====================

if [ -z "${!API_KEY_ENV}" ]; then
    echo "❌ 错误: 未设置环境变量 $API_KEY_ENV"
    echo ""
    echo "请设置 API Key:"
    echo "  export $API_KEY_ENV='your-api-key'"
    echo ""
    exit 1
fi

echo "✅ API Key 已设置: $API_KEY_ENV"
echo ""

# ==================== 确认开始 ====================

read -p "确认使用统一配置开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ==================== 运行推理 ====================

echo ""
echo "=========================================================================="
echo "  开始推理"
echo "=========================================================================="
echo ""

# 构建命令
CMD="python async_inference_api.py \
    --input \"$INPUT_JSONL\" \
    --output \"$OUTPUT_JSONL\" \
    --api-provider $API_PROVIDER \
    --model \"$MODEL_NAME\" \
    --qps $QPS \
    --max-concurrent $MAX_CONCURRENT \
    --max-turns $MAX_TURNS"

# 添加 QPM 参数（如果设置）
if [ ! -z "$QPM" ]; then
    CMD="$CMD --qpm $QPM"
fi

# 添加 base-url（如果是 Azure）
if [ "$API_PROVIDER" = "azure" ] && [ ! -z "$BASE_URL" ]; then
    CMD="$CMD --base-url \"$BASE_URL\""
fi

echo "🚀 执行命令:"
echo "$CMD"
echo ""

# 执行
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================================="
    echo "  推理完成！"
    echo "=========================================================================="
    echo ""
    echo "📁 结果文件: $OUTPUT_JSONL"
    echo "📊 处理数量: $(wc -l < \"$OUTPUT_JSONL\") 条"
    echo ""
    echo "📝 后续步骤:"
    echo "   1. 计算指标:"
    echo "      python calculate_metrics.py $OUTPUT_JSONL"
    echo ""
    echo "   2. 可视化分析:"
    echo "      python visualize_results.py --input ${OUTPUT_JSONL%.jsonl}_stats.json"
    echo ""
else
    echo ""
    echo "❌ 推理失败"
    exit 1
fi
