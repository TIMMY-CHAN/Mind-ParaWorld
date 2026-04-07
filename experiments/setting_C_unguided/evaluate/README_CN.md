# Setting C: Unguided Search — LLM-as-Judge 评估

本目录包含用于评估 Setting C 推理结果的 LLM-as-Judge 脚本。

## 目录结构

```
evaluate/
├── llm-as-judge.py              # LLM-as-Judge 评估引擎
├── run_judge_evaluation.sh      # 一键评估脚本
└── README_CN.md
```

## 使用场景

1. **首次评估**：对推理结果文件进行 Correct / Incorrect 判定
2. **更换 Judge**：使用不同 Judge 模型重新评估，对比评分一致性
3. **验证结果**：抽检已有评估结果的准确性

## 快速开始

### 方式 1：一键脚本（推荐）

```bash
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

脚本以交互方式询问 Judge 模型地址、待评估结果文件等配置。

### 方式 2：直接调用 Python 脚本

```bash
python experiments/setting_C_unguided/evaluate/llm-as-judge.py \
    --input results/setting_C/<model_name>/inference_results.jsonl \
    --output results/setting_C/<model_name>/evaluated_results.jsonl \
    --judge-model <judge_model_name> \
    --judge-api-base http://<judge_host>:8000/v1 \
    --max-concurrent 32 \
    --max-tokens 8192 \
    --temperature 0.8
```

## 配置参数

### Judge 模型

```bash
JUDGE_MODEL="<judge_model_name>"
JUDGE_API_BASE="http://<judge_host>:8000/v1"
```

### 并发与重试

```bash
MAX_CONCURRENT=32      # Judge 并发数
MAX_RETRIES=3          # 失败重试次数
MAX_TOKENS=8192        # Judge 最大 token 数
TEMPERATURE=0.8        # Judge 采样温度
```

### 强制重新评估

```bash
FORCE_REEVALUATE=false  # 设为 true 时忽略已有结果，全量重新评估
```

## 输出格式

每条样本在原有字段基础上新增：

```json
{
  "answer": "Correct",
  "think": "判断理由",
  "judge_response": "Judge 完整响应",
  "judge_status": "success"
}
```

旧的 `evaluation` 字段会被移除。

## 断点续传

评估中断后重新运行脚本，会自动跳过已完成的样本。如需全量重新评估，设置 `FORCE_REEVALUATE=true`。

## Judge Prompt

与 Setting B 使用相同的 Judge Prompt，确保两个 Setting 的评分标准一致可比：

```
你是一个专业的答案评估专家。你的任务是判断模型的预测答案是否与标准答案一致。

评估标准：
1. 如果预测答案与标准答案在语义上一致，判定为"Correct"
2. 如果预测答案与标准答案不一致、缺少关键信息、或完全错误，判定为"Incorrect"
```

## 与 Setting B 的差异

| 特性 | Setting B | Setting C |
|------|-----------|-----------|
| 输入文件 | `inference_results.jsonl` | `evaluated_results.jsonl` |
| 是否有旧 evaluation 字段 | 否 | 是（会被移除） |
| 输出文件名 | `evaluated_results.jsonl` | `re_evaluated_results.jsonl` |
| 强制重新评估选项 | 不支持 | 支持 |

## 常见问题

**输入文件不存在**：确认推理已完成，文件路径与命名正确。

**Judge API 连接失败**：确认 Judge 服务已启动，地址与端口配置正确。

**内存不足**：降低 `MAX_CONCURRENT`（如 32 → 16）或 `MAX_TOKENS`（如 8192 → 4096）。
