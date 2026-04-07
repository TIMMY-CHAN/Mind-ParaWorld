#!/bin/bash
# run_fewshot_ablation.sh - Few-shot Decomposition 消融实验
#
# 使用方法：
#   ./run_fewshot_ablation.sh
#
# 说明：
#   - 使用 Few-shot Decomposition Prompt
#   - 仅在 vLLM 部署的本地模型上运行（节省成本）
#   - 与基线版本（无 Few-shot）对比，评估 prompt 工程的效果

set -e

echo "========================================================================"
echo "  MPW-bench Few-shot Decomposition 消融实验"
echo "========================================================================"
echo ""

# ================= 可配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

# 数据路径（可通过环境变量覆盖）
MPW_BENCH_DIR="${MPW_BENCH_DIR:-${PROJECT_ROOT}/data/MPW-bench}"
DATA_DIR="data"
RESULTS_DIR="abslation_results/Qwen3-32B"  # 专门的输出目录

# API 配置（vLLM 本地模型）
MODEL_NAME="models"
# API_BASE="http://10.72.8.53:8000/v1"
API_BASE="${API_BASE:-http://localhost:8000/v1}"

# 推理配置
MAX_CONCURRENT_TURNS=128        # Turn 级别最大并发数
MAX_TURNS_PER_SAMPLE=32         # 每个样本的最大轮数
MAX_CONTEXT_CHARS=40960        # 上下文最大字符数

# 文件命名
INPUT_JSONL="$DATA_DIR/mpw_bench_full.jsonl"
OUTPUT_JSONL="$RESULTS_DIR/inference_results_fewshot.jsonl"

# ================= 显示配置 =================

echo "🎯 消融实验设计:"
echo "  - H1: 模型已固化搜索习惯 → Few-shot 提示无效"
echo "  - H2: 模型具备分解能力但缺指导 → Few-shot 提示有效"
echo ""
echo "📋 配置信息:"
echo "  数据集配置:"
echo "    - MPW-bench 目录: $MPW_BENCH_DIR"
echo "    - 输入文件: $INPUT_JSONL"
echo "    - 输出文件: $OUTPUT_JSONL"
echo ""
echo "  推理配置:"
echo "    - 模型: $MODEL_NAME (vLLM 本地)"
echo "    - API 地址: $API_BASE"
echo "    - Turn 并发数: $MAX_CONCURRENT_TURNS"
echo "    - 最大轮数/样本: $MAX_TURNS_PER_SAMPLE"
echo "    - 上下文窗口: $MAX_CONTEXT_CHARS 字符"
echo ""
echo "  Prompt 版本: Few-shot Decomposition (4 个示例)"
echo ""

# ================= 确认开始 =================

read -p "确认开始 Few-shot 消融实验？(y/N) " -n 1 -r
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

# 检查是否已合并
if [ ! -f "$INPUT_JSONL" ]; then
    echo "📋 合并 MPW-bench 数据..."
    python merge_mpw_bench.py

    if [ $? -ne 0 ]; then
        echo "❌ 数据合并失败"
        exit 1
    fi
else
    echo "✅ 数据文件已存在: $INPUT_JSONL"

    # 显示统计
    echo ""
    echo "📊 数据统计:"
    total_count=$(wc -l < "$INPUT_JSONL")
    echo "   总数据量: $total_count 条"
fi

# ================= 步骤 2: 推理（使用 Few-shot Prompt）=================

echo ""
echo "========================================================================"
echo "  步骤 2/2: 推理（Few-shot Decomposition）"
echo "========================================================================"
echo ""

echo "🚀 开始推理..."
echo "   Few-shot 示例："
echo "   1. 比较类问题分解"
echo "   2. 时间差计算分解"
echo "   3. 条件筛选分解"
echo "   4. 多维度比较分解"
echo ""

# 运行推理（使用 Few-shot 版本）
python experiments/setting_B_guided/vllm/async_inference.py \
    --model_name "$MODEL_NAME" \
    --api_base "$API_BASE" \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL" \
    --max_concurrent_turns "$MAX_CONCURRENT_TURNS" \
    --max_turns_per_sample "$MAX_TURNS_PER_SAMPLE" \
    --max_context_chars "$MAX_CONTEXT_CHARS"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 推理失败"
    exit 1
fi

# ================= 完成 =================

echo ""
echo "========================================================================"
echo "  Few-shot 消融实验完成！"
echo "========================================================================"
echo ""
echo "📁 结果文件:"
echo "   推理结果: $OUTPUT_JSONL"
echo ""
echo "📊 下一步："
echo "   1. 评估 Few-shot 版本:"
echo "      python evaluate_with_judge.py \\"
echo "          --input $OUTPUT_JSONL \\"
echo "          --output ${RESULTS_DIR}/evaluated_results.jsonl \\"
echo "          --judge-provider vllm \\"
echo "          --judge-vllm-url http://localhost:8000 \\"
echo "          --judge-max-concurrent 100"
echo ""
echo "   2. 对比基线（无 Few-shot）版本:"
echo "      python compare_models.py \\"
echo "          --baseline results/baseline/evaluated_results.jsonl \\"
echo "          --fewshot ${RESULTS_DIR}/evaluated_results.jsonl \\"
echo "          --output few shot_comparison.md"
echo ""
echo "✅ 完成！"
