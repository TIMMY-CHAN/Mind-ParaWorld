#!/bin/bash
# batch_calculate_difficulty_pass1.sh - 批量计算 Setting A 所有模型的按难度分级 Pass@1

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="$PROJECT_ROOT"

SETTING_A_DIR="$PROJECT_ROOT/setting_A_results"
SCRIPT_PATH="$PROJECT_ROOT/analysis/scripts/calculate_difficulty_pass1.py"

# 查找所有 evaluated_results.jsonl 文件
echo "🔍 扫描 Setting A 模型..."
echo ""

MODEL_COUNT=0
MODELS=()

for model_dir in "$SETTING_A_DIR"/*/; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi

    model_name=$(basename "$model_dir")
    eval_file="$model_dir/evaluated_results.jsonl"

    if [ -f "$eval_file" ]; then
        MODELS+=("$model_name")
        MODEL_COUNT=$((MODEL_COUNT + 1))
        echo "  ✅ $model_name"
    else
        echo "  ⚠️  $model_name (无 evaluated_results.jsonl)"
    fi
done

echo ""
echo "找到 $MODEL_COUNT 个模型"
echo ""

if [ $MODEL_COUNT -eq 0 ]; then
    echo "❌ 未找到任何模型"
    exit 1
fi

read -p "确认开始计算？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "========================================================================"
echo "  开始计算..."
echo "========================================================================"
echo ""

for model_name in "${MODELS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "模型: $model_name"
    echo "----------------------------------------"

    eval_file="$SETTING_A_DIR/$model_name/evaluated_results.jsonl"
    output_json="$SETTING_A_DIR/$model_name/difficulty_pass1_stats.json"

    python "$SCRIPT_PATH" "$eval_file" --output "$output_json"

    echo ""
done

echo ""
echo "========================================================================"
echo "✅ 所有模型计算完成！"
echo "========================================================================"
echo ""
echo "统计文件保存在各模型目录下的 difficulty_pass1_stats.json"
echo ""
