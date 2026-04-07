#!/bin/bash
# batch_quantitative_analysis_settingB.sh - 批量计算 Setting B 所有模型的定量分析指标

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="$PROJECT_ROOT"

SETTING_B_BASE="$PROJECT_ROOT/setting_b_results"
SCRIPT_PATH="$PROJECT_ROOT/analysis/tools/quantitative_analysis_settingB_C.py"
OUTPUT_DIR="$PROJECT_ROOT/analysis_results/quantitative_settingB"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找所有 evaluated_results.jsonl 文件
echo "🔍 扫描 Setting B 模型..."
echo ""

EVAL_FILES=()
MODEL_NAMES=()

for eval_file in $(find "$SETTING_B_BASE" -name "evaluated_results.jsonl"); do
    model_dir=$(dirname "$eval_file")
    model_name=$(basename "$model_dir")

    EVAL_FILES+=("$eval_file")
    MODEL_NAMES+=("$model_name")

    echo "  ✅ $model_name"
done

echo ""
echo "找到 ${#EVAL_FILES[@]} 个模型"
echo ""

if [ ${#EVAL_FILES[@]} -eq 0 ]; then
    echo "❌ 未找到任何模型"
    exit 1
fi

# 显示将要分析的模型
echo "📋 将分析以下模型:"
for model_name in "${MODEL_NAMES[@]}"; do
    echo "   - $model_name"
done
echo ""

read -p "确认开始分析？(y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "========================================================================"
echo "  开始分析..."
echo "========================================================================"
echo ""

# 运行定量分析 (一次性分析所有模型)
python "$SCRIPT_PATH" --inputs "${EVAL_FILES[@]}" --output-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ 所有模型分析完成！"
    echo "========================================================================"
    echo ""
    echo "📂 结果保存在: $OUTPUT_DIR"
    echo ""
    echo "📊 生成的文件:"
    ls -lh "$OUTPUT_DIR"/*.json | awk '{print "   " $9 " (" $5 ")"}'
    echo ""
else
    echo ""
    echo "❌ 分析失败"
    exit 1
fi
