# Setting B: Guided Search (Few-shot Query Decomposition)

**实验类型**: 有查询指导的搜索 - Few-shot Query Decomposition

## 实验设定

在这个实验设定中，模型需要通过web_search工具检索信息，但**System Prompt中包含了Few-shot查询分解示例**，明确展示：
- 什么样的查询是好的 vs 什么样的查询是差的
- 如何将复杂问题分解为原子化查询
- 查询分解的核心原则

### 核心特征

1. **Few-shot Query Decomposition Guide**
   - 4个详细示例（比较类、时间差计算、条件筛选、多维度比较）
   - 每个示例都展示"差的查询"和"好的查询"对比
   - 明确的分解步骤说明

2. **核心原则**
   - 原子化查询：一个查询只关注一个实体、一个属性
   - 先收集后计算：不要期待搜索引擎帮你做计算
   - 明确实体：使用具体名称，避免模糊指代
   - 分步推理：复杂问题 = 多个简单查询 + 自己的推理

## 目录结构

```
setting_B_guided/
├── api/                                      # API版本（已实现）
│   ├── async_inference_api_fewshot.py       # Few-shot API推理引擎
│   └── run_fewshot_inference.sh             # 执行脚本
└── vllm/                                     # vLLM版本（已实现）
    ├── async_inference_v2_fewshot.py        # Few-shot vLLM推理引擎
    └── run_vllm_fewshot.sh                  # 执行脚本
```

## 使用方法

### API版本（已实现）

```bash
# 方式1: 使用执行脚本（推荐）
cd experiments/setting_B_guided/api
./run_fewshot_inference.sh

# 方式2: 直接调用Python脚本
python experiments/setting_B_guided/api/async_inference_api_fewshot.py \
    --provider licloud \
    --model Qwen/Qwen2-VL-72B-Instruct-AWQ \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/inference_results.jsonl \
    --qps 2 \
    --max-concurrent-turns 32
```

### vLLM版本（已实现）

```bash
# 方式1: 使用执行脚本（推荐）
cd experiments/setting_B_guided/vllm
./run_vllm_fewshot.sh

# 方式2: 直接调用Python脚本
python experiments/setting_B_guided/vllm/async_inference_v2_fewshot.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B_vllm/inference_results.jsonl \
    --model models \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16 \
    --max-context-chars 60000 \
    --max-retries 5
```

**配置说明**:
- `--max-concurrent-turns`: Turn级别最大并发数（建议32-128）
- `--max-turns-per-sample`: 每个样本的最大轮数（建议16-32）
- `--max-context-chars`: 上下文窗口字符数（128K模型建议128000）
- `--max-retries`: API调用遇到网络错误时的最大重试次数（指数退避，默认5）
- `--enable-thinking`: 强制启用模型内置 thinking 模式
- `--disable-thinking`: 强制禁用模型内置 thinking 模式（不传则使用模型默认行为）

## Prompt对比

### Few-shot System Prompt (Setting B)

```python
INSTRUCTION_PROMPT_SYSTEM_FEWSHOT = """
你是一个ReAct范式的多模态agent...

---

# 查询分解指南（Query Decomposition Guide）

## 示例 1：比较类问题
❌ 差的查询：比较 2022-23赛季 尤文图斯和那不勒斯的客场进球数哪个更多
✅ 好的查询：
  步骤1: 搜索 "2022-23赛季 尤文图斯 客场进球数"
  步骤2: 搜索 "2022-23赛季 那不勒斯 客场进球数"
  步骤3: 比较两个数值

## 示例 2-4: (更多分解示例)
...

## 核心原则
1. 原子化查询
2. 先收集后计算
3. 明确实体
4. 分步推理
"""
```

### Zero-shot Baseline (Setting C)

```python
INSTRUCTION_PROMPT_SYSTEM = """
你是一个ReAct范式的多模态agent...
你可以调用的工具包括以下两种：
web_search:
...
"""
```

**关键差异**: Setting B增加了147行查询分解指南和示例

## 实验目的

通过对比Setting B (有查询指导) 和Setting C (无指导)，评估：
1. Few-shot Query Decomposition的有效性
2. 查询质量对检索效果的影响
3. 模型是否能学会分解复杂查询

## 评估指标

重点关注：
- **FCR (Fact Coverage Rate)**: 查询质量直接影响事实覆盖率
- **Hit Precision**: 原子化查询应提高命中精度
- **Avg Tool Calls**: 可能增加（因为分解为更多查询）
- **Pass@1**: 最终准确率是否提升

## 与其他Setting的对比

| Setting | Prompt类型 | 查询指导 | 代码行数差异 |
|---------|-----------|----------|-------------|
| **Setting B (Guided)** | Few-shot | ✅ 4个示例 + 核心原则 | +147行 |
| **Setting C (Unguided)** | Zero-shot | ❌ 无 | 基线 |

---

**最后更新**: 2026-02-10
**状态**: API版本和vLLM版本均已实现
