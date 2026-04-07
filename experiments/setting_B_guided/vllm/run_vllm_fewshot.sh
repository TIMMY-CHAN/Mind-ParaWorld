#!/bin/bash
# run_vllm_fewshot.sh - Setting B: Guided Search (Few-shot) - vLLM版本

set -e

echo "========================================================================"
echo "  Setting B: Guided Search (Few-shot) - vLLM推理"
echo "========================================================================"
echo ""

# ================= Prompt 选择 =================

echo "📝 请选择要使用的 System Prompt:"
echo "  1) fewshot_prompt   - 包含4个查询分解示例（详细）"
echo "  2) guidance_prompt   - 原子化查询指南（简洁）"
echo ""
read -p "请输入选项 (1 或 2): " prompt_choice

case $prompt_choice in
    1)
        PROMPT_TYPE="fewshot_prompt"
        PROMPT_DESC="Few-shot Query Decomposition (4个示例)"
        ;;
    2)
        PROMPT_TYPE="guidance_prompt"
        PROMPT_DESC="Atomic Query Guidance (简洁指南)"
        ;;
    *)
        echo "❌ 无效选项，请输入 1 或 2"
        exit 1
        ;;
esac

echo "✅ 已选择: $PROMPT_TYPE - $PROMPT_DESC"
echo ""

# ================= 可配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# 数据路径
DATA_DIR="data"
RESULTS_DIR="setting_b_results/$PROMPT_TYPE"

# Search Agent 配置（vLLM 部署的推理模型）
MODEL_NAME="models"
# API_BASE="http://10.72.8.45:8000/v1"
API_BASE="${API_BASE:-http://localhost:8000/v1}"

# 世界模型配置（支持多节点，逗号分隔）
# WORLD_MODEL_ENDPOINTS="http://10.72.8.36:8000/v1"
WORLD_MODEL_ENDPOINTS="${WORLD_MODEL_ENDPOINTS:-http://localhost:8000/v1}"
# WORLD_MODEL_ENDPOINTS="http://node1:8000/v1,http://node2:8000/v1,http://node3:8000/v1"

# 推理配置
# MAX_CONCURRENT_TURNS=64        # Turn 级别最大并发数（建议 32-128）
# MAX_TURNS_PER_SAMPLE=32         # 每个样本的最大轮数
# MAX_CONTEXT_CHARS=196608        # 上下文最大字符数（128K模型建议128000）
MAX_CONCURRENT_TURNS=64        # Turn 级别最大并发数（建议 32-128）
MAX_TURNS_PER_SAMPLE=32         # 每个样本的最大轮数
MAX_CONTEXT_CHARS=128000        # 上下文最大字符数（128K模型建议128000）
ENABLE_THINKING=false           # 启用thinking模式（Qwen等模型需要，设置为false关闭）

MODEL="tongyi_deepresearch"
# MODEL="Qwen3-30B-A3B"
# 文件命名
INPUT_JSONL="$DATA_DIR/mpw_bench_full.jsonl"
OUTPUT_JSONL="$RESULTS_DIR/$MODEL/inference_results.jsonl"

# ================= 显示配置 =================

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
echo "  实验设定: Setting B (Guided Search)"
echo "    - System Prompt: $PROMPT_TYPE"
echo "    - 描述: $PROMPT_DESC"
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
mkdir -p "$RESULTS_DIR"

# 检查数据文件
if [ ! -f "$INPUT_JSONL" ]; then
    echo "❌ 数据文件不存在: $INPUT_JSONL"
    echo "请先运行: python merge_mpw_bench.py"
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
echo "  步骤 2/2: 运行推理（Few-shot版本）"
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

# 估算时间
estimated_minutes=$((remaining_count * 90 / 60))
estimated_hours=$((estimated_minutes / 60))
estimated_remaining_minutes=$((estimated_minutes % 60))

echo "⏱️  预计耗时: ~${estimated_hours}小时${estimated_remaining_minutes}分钟"
echo "   (按每条 0.9-1.0 分钟估算)"
echo ""

read -p "开始推理？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始推理（$PROMPT_DESC）..."
echo "----------------------------------------"

# 导出世界模型节点配置（由 world_model_web_search_tool.py 读取）
export WORLD_MODEL_ENDPOINTS

# 构建命令参数
PYTHON_CMD="python experiments/setting_B_guided/vllm/async_inference.py \
    --input \"$INPUT_JSONL\" \
    --output \"$OUTPUT_JSONL\" \
    --model \"$MODEL_NAME\" \
    --api-base \"$API_BASE\" \
    --max-concurrent-turns \"$MAX_CONCURRENT_TURNS\" \
    --max-turns-per-sample \"$MAX_TURNS_PER_SAMPLE\" \
    --max-context-chars \"$MAX_CONTEXT_CHARS\" \
    --prompt-type \"$PROMPT_TYPE\""

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
    echo "   1. 使用 evaluation_pipeline 进行评估"
    echo "   2. 与其他 Prompt 版本对比分析"
    echo "   3. 分析不同Prompt对查询质量的影响"
    echo ""
else
    echo ""
    echo "❌ 推理失败"
    exit 1
fi
