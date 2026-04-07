# Setting C 评估工具 - 快速开始

## ✅ 已完成部署

Setting C 的 LLM-as-Judge 重新评估工具已成功部署在：

```
experiments/setting_C_unguided/evaluate/
```

## 📁 文件清单

- ✅ `llm-as-judge.py` - 评估引擎（14KB，支持重新评估和断点续传）
- ✅ `run_judge_evaluation.sh` - 一键运行脚本（4.1KB，可执行）
- ✅ `README.md` - 完整文档（6.7KB）

## 🚀 使用方法

### 单个模型评估

```bash
cd /path/to/Mind-ParaWorld

# 1. 编辑配置
vim experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
# 修改第 14 行：MODEL="Qwen3-32B_round1"

# 2. 运行评估
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh

# 3. 查看结果
# 输出文件：setting_c_results/Qwen3-32B_round1/re_evaluated_results.jsonl
```

### 批量评估所有模型

创建批量脚本 `batch_evaluate_setting_c.sh`：

```bash
#!/bin/bash

MODELS=(
    "Qwen3-32B_round1"
    "Qwen3-32B_round2"
    "Qwen3-32B_round3"
    "Qwen3-30B-A3B_round1"
    "Qwen3-30B-A3B_round2"
    "Qwen3-30B-A3B_round3"
    "MindWatcher_round1"
    "MindWatcher_round2"
    "licloud_doubao-seed-1-6"
    "licloud_gemini-3-pro-preview"
    "licloud_gemini-2_5-flash"
    "licloud_gemini-2_5-pro"
    "licloud_kivy-glm-4_6"
    "licloud_kivy-minimax-m2"
    "licloud_volcengine-kimi-k2-250711"
    "licloud_azure-gpt-5"
    "tongyi_deepresearch_round1"
    "tongyi_deepresearch_round2"
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "评估模型: $MODEL"
    echo "========================================"

    sed -i "s/^MODEL=.*/MODEL=\"$MODEL\"/" experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
    yes | bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh

    echo ""
done

echo "✅ 所有模型评估完成！"
```

## 🔧 配置说明

在 `run_judge_evaluation.sh` 中可配置：

```bash
JUDGE_MODEL="models"                      # Judge 模型
JUDGE_API_BASE="http://10.72.8.35:8000/v1"  # Judge API 地址
MODEL="Qwen3-32B_round1"                  # 待评估模型
MAX_CONCURRENT=32                         # 并发数
MAX_TOKENS=8192                           # 最大 tokens
TEMPERATURE=0.8                           # 采样温度
```

## 📊 输出说明

### 输出文件位置

```
setting_c_results/{MODEL}/re_evaluated_results.jsonl
```

### 字段变化

| 字段 | 变化 |
|------|------|
| `evaluation` (旧) | ❌ 移除 |
| `answer` | ✅ 新增（Correct/Incorrect）|
| `think` | ✅ 新增（判断理由）|
| `judge_response` | ✅ 新增（完整响应）|
| `judge_status` | ✅ 新增（success/error/skipped）|

其他字段（messages, ground_truth, prediction 等）保持不变。

## ✅ 测试结果

已在 5 个样本上成功测试：
- ✅ 评估引擎正常工作
- ✅ 输出格式正确
- ✅ 旧 evaluation 字段成功移除
- ✅ 新 Judge 结果正确保存
- ✅ 断点续传功能正常

## 🎯 与 Setting B 的一致性

| 特性 | Setting B | Setting C |
|------|-----------|-----------|
| Judge Prompt | ✅ 相同 | ✅ 相同 |
| 评估标准 | ✅ Correct/Incorrect | ✅ Correct/Incorrect |
| 并发机制 | ✅ 异步 | ✅ 异步 |
| 重试机制 | ✅ 3次 | ✅ 3次 |
| 断点续传 | ✅ 支持 | ✅ 支持 |

确保两个 Setting 使用完全相同的评估逻辑，结果可直接对比。

## 📝 下一步

1. **配置 Judge 模型**：确认 Judge API 可用
2. **选择评估模型**：修改 `run_judge_evaluation.sh` 中的 `MODEL` 变量
3. **开始评估**：运行脚本，等待完成
4. **对比结果**：比较新旧评估结果的差异

---

**部署完成时间**: 2026-02-18
**测试状态**: ✅ 通过（5/5 样本）
