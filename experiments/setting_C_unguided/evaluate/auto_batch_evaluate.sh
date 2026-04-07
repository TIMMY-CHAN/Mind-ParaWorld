#!/bin/bash
# auto_batch_evaluate.sh - 自动扫描并批量评估 Setting C 结果

set -e

echo "========================================================================"
echo "  Setting C: 自动批量评估（扫描 + 循环评测）"
echo "========================================================================"
echo ""

# ================= 配置参数 =================

# 环境配置 - 自动推导项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}"

SETTING_C_DIR="${PROJECT_ROOT}/setting_c_results"
JUDGE_MODEL="models"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://localhost:8000/v1}"
MAX_CONCURRENT=32
MAX_RETRIES=3
MAX_TOKENS=8192
TEMPERATURE=0.8

# 输入和输出文件名
INPUT_FILENAME="inference_results_full.jsonl"
OUTPUT_FILENAME="re_evaluated_results.jsonl"

echo "📋 配置信息:"
echo "  扫描目录: $SETTING_C_DIR"
echo "  Judge模型: $JUDGE_MODEL"
echo "  Judge API: $JUDGE_API_BASE"
echo "  输入文件名: $INPUT_FILENAME"
echo "  输出文件名: $OUTPUT_FILENAME"
echo "  最大并发: $MAX_CONCURRENT"
echo ""

# ================= 扫描未评测模型 =================

echo "🔍 扫描未评测的模型..."
echo ""

UNEVALUATED_MODELS=()

for model_dir in "$SETTING_C_DIR"/*/; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi

    model_name=$(basename "$model_dir")
    input_file="$model_dir$INPUT_FILENAME"
    output_file="$model_dir$OUTPUT_FILENAME"

    # 检查输入文件是否存在
    if [ ! -f "$input_file" ]; then
        echo "  ⚠️  跳过 $model_name: 输入文件不存在"
        continue
    fi

    # 检查输出文件是否存在
    if [ -f "$output_file" ]; then
        # 检查输出是否完整（行数是否一致）
        input_lines=$(wc -l < "$input_file")
        output_lines=$(wc -l < "$output_file")

        if [ "$input_lines" -eq "$output_lines" ]; then
            echo "  ✅ 跳过 $model_name: 已完成评估 ($output_lines/$input_lines)"
        else
            echo "  🔄 待评估 $model_name: 未完成 ($output_lines/$input_lines)"
            UNEVALUATED_MODELS+=("$model_name")
        fi
    else
        echo "  📝 待评估 $model_name: 未开始 (0/$(wc -l < "$input_file"))"
        UNEVALUATED_MODELS+=("$model_name")
    fi
done

echo ""
echo "========================================"
echo "📊 扫描结果:"
echo "  总模型数: $(ls -d "$SETTING_C_DIR"/*/ 2>/dev/null | wc -l)"
echo "  待评估数: ${#UNEVALUATED_MODELS[@]}"
echo "========================================"
echo ""

if [ ${#UNEVALUATED_MODELS[@]} -eq 0 ]; then
    echo "✅ 所有模型已完成评估！"
    exit 0
fi

echo "待评估模型列表:"
for model in "${UNEVALUATED_MODELS[@]}"; do
    echo "  - $model"
done
echo ""

read -p "确认开始批量评估？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# ================= 循环评估 =================

echo ""
echo "========================================================================"
echo "🚀 开始批量评估..."
echo "========================================================================"
echo ""

TOTAL_COUNT=${#UNEVALUATED_MODELS[@]}
CURRENT_INDEX=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for model in "${UNEVALUATED_MODELS[@]}"; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))

    echo ""
    echo "========================================================================"
    echo "  [$CURRENT_INDEX/$TOTAL_COUNT] 评估模型: $model"
    echo "========================================================================"
    echo ""

    input_file="$SETTING_C_DIR/$model/$INPUT_FILENAME"
    output_file="$SETTING_C_DIR/$model/$OUTPUT_FILENAME"

    # 显示任务信息
    echo "📂 输入文件: $input_file"
    echo "📂 输出文件: $output_file"
    echo "📊 样本数量: $(wc -l < "$input_file")"
    echo ""

    # 运行评估
    python experiments/setting_C_unguided/evaluate/llm-as-judge.py \
        --input "$input_file" \
        --output "$output_file" \
        --judge-model "$JUDGE_MODEL" \
        --judge-api-base "$JUDGE_API_BASE" \
        --max-concurrent "$MAX_CONCURRENT" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --max-retries "$MAX_RETRIES"

    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # 统计结果
        echo ""
        echo "✅ [$CURRENT_INDEX/$TOTAL_COUNT] $model 评估完成！"
        echo ""
        echo "📊 评估统计:"
        python3 << EOF
import json

results = []
with open('$output_file', 'r') as f:
    for line in f:
        results.append(json.loads(line.strip()))

correct = sum(1 for r in results if r.get('answer') == 'Correct')
incorrect = sum(1 for r in results if r.get('answer') == 'Incorrect')
total = len(results)
accuracy = correct / total * 100 if total > 0 else 0

print(f"  总样本数: {total}")
print(f"  Correct: {correct} ({correct/total*100:.2f}%)")
print(f"  Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
print(f"  Accuracy: {accuracy:.2f}%")
EOF

    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        echo "❌ [$CURRENT_INDEX/$TOTAL_COUNT] $model 评估失败"
    fi

    echo ""
    echo "----------------------------------------"
    echo "进度: $CURRENT_INDEX/$TOTAL_COUNT 完成 | 成功: $SUCCESS_COUNT | 失败: $FAIL_COUNT"
    echo "----------------------------------------"

    # 重新扫描，看是否还有新的未评测模型（动态添加）
    # 这里可以选择性启用，目前注释掉
    # NEW_UNEVALUATED=$(find "$SETTING_C_DIR" -maxdepth 2 -name "$INPUT_FILENAME" -type f | ...)
done

# ================= 最终汇总 =================

echo ""
echo "========================================================================"
echo "✅ 批量评估完成！"
echo "========================================================================"
echo ""
echo "📊 最终统计:"
echo "  总评估数: $TOTAL_COUNT"
echo "  成功数: $SUCCESS_COUNT"
echo "  失败数: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 所有模型评估成功！"
else
    echo "⚠️  部分模型评估失败，请检查日志"
fi

echo ""
echo "📁 结果文件位置: $SETTING_C_DIR/{model_name}/$OUTPUT_FILENAME"
echo ""
