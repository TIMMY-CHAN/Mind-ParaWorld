#!/bin/bash
# run_full_inference_v2.sh
# 完整推理脚本 - 只推理，不评测

set -e

echo "========================================================================"
echo "  MPW-bench 完整推理 - V2 异步引擎"
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
RESULTS_DIR="setting_c_results/Qwen3-30b-a3b"

# Search Agent 配置（vLLM 部署的推理模型）
MODEL_NAME="models"
# API_BASE="http://10.72.8.56:8000/v1"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
# 世界模型配置（支持多节点，逗号分隔）
# WORLD_MODEL_ENDPOINTS="http://10.72.0.89:8000/v1"
WORLD_MODEL_ENDPOINTS="${WORLD_MODEL_ENDPOINTS:-http://localhost:8000/v1}"
# WORLD_MODEL_ENDPOINTS="http://node1:8000/v1,http://node2:8000/v1,http://node3:8000/v1"

# 推理配置
MAX_CONCURRENT_TURNS=64        # Turn 级别最大并发数（建议 32-64）
MAX_TURNS_PER_SAMPLE=32         # 每个样本的最大轮数
MAX_CONTEXT_CHARS=128000        # 上下文最大字符数（32K模型建议60000，128K模型建议200000）

# 文件命名
INPUT_JSONL="$DATA_DIR/mpw_bench_full.jsonl"
OUTPUT_JSONL="$RESULTS_DIR/inference_results_full.jsonl"

# ================= 显示配置 =================

echo "📋 配置信息:"
echo "  数据集配置:"
echo "    - MPW-bench 目录: $MPW_BENCH_DIR"
echo "    - 输入文件: $INPUT_JSONL"
echo "    - 输出文件: $OUTPUT_JSONL"
echo ""
echo "  推理配置:"
echo "    - 模型: $MODEL_NAME"
echo "    - Search Agent API: $API_BASE"
echo "    - 世界模型节点: $WORLD_MODEL_ENDPOINTS"
echo "    - Turn 并发数: $MAX_CONCURRENT_TURNS"
echo "    - 最大轮数/样本: $MAX_TURNS_PER_SAMPLE"
echo "    - 上下文窗口: $MAX_CONTEXT_CHARS 字符"
echo ""
echo "  引擎版本: V2 (真正异步)"
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

    # 统计各类别
    echo ""
    echo "   各类别分布:"
    python3 << EOF
import json
from collections import Counter

categories = []
with open('$INPUT_JSONL', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        categories.append(item.get('category', 'unknown'))

counter = Counter(categories)
for category, count in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"     {category:50s}: {count:4d} 条")
EOF
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
echo "🚀 开始推理..."
echo "----------------------------------------"

# 导出世界模型节点配置（由 world_model_web_search_tool.py 读取）
export WORLD_MODEL_ENDPOINTS

# 运行 V2 异步推理引擎
python experiments/setting_C_unguided/vllm/async_inference.py \
    --input "$INPUT_JSONL" \
    --output "$OUTPUT_JSONL" \
    --model "$MODEL_NAME" \
    --api-base "$API_BASE" \
    --max-concurrent-turns "$MAX_CONCURRENT_TURNS" \
    --max-turns-per-sample "$MAX_TURNS_PER_SAMPLE" \
    --max-context-chars "$MAX_CONTEXT_CHARS"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推理完成！"
    echo ""
    echo "📁 结果文件: $OUTPUT_JSONL"
    echo "📊 处理数量: $(wc -l < "$OUTPUT_JSONL") 条"
    echo ""

    # 显示完成统计
    echo "📈 各类别完成情况:"
    python3 << EOF
import json
from collections import Counter

categories = []
with open('$OUTPUT_JSONL', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line)
            # 从 messages 或其他地方提取 category（如果保存了）
            # 这里简化处理，只统计总数
        except:
            pass

total = sum(1 for _ in open('$OUTPUT_JSONL'))
print(f"   总计: {total} 条")
EOF

    echo ""
    echo "========================================================================"
    echo "  推理完成！"
    echo "========================================================================"
    echo ""
    echo "📝 后续步骤:"
    echo "   1. 使用 analyze_results.py 进行统计分析"
    echo "   2. 使用 visualize_results.py 生成可视化图表"
    echo "   3. 可选：使用 LLM as Judge 进行精确评测"
    echo ""
    echo "   示例命令:"
    echo "   python verl/analyze_results.py --input $OUTPUT_JSONL --output results_full/stats.json"
    echo ""
else
    echo ""
    echo "❌ 推理失败"
    exit 1
fi
