# Experiments

本目录包含 MPW benchmark 的三种官方评测设置（Setting A / B / C）以及自定义 Prompt 评测入口（Custom）。

```
experiments/
├── setting_A_oracle/       # Oracle：直接提供原子事实，单轮回答
├── setting_B_guided/       # Guided：多轮 ReAct + 查询分解引导
├── setting_C_unguided/     # Unguided：多轮 ReAct + 标准 Prompt
└── custom/                 # 自定义 Prompt 评测（详见 custom/README.md）
```

---

## 三种设置对比

| | **Setting A — Oracle** | **Setting B — Guided** | **Setting C — Unguided** |
|---|---|---|---|
| **核心用途** | 估计信息充分时的推理上界 | 评估查询分解引导对 agent 的增益 | 评估端到端 agent 真实能力 |
| **工具调用** | ❌ 无 | ✅ 多轮 `web_search` | ✅ 多轮 `web_search` |
| **System Prompt** | 给定原子事实 + 单轮 QA | 含查询分解示例的 guided prompt | 标准 ReAct prompt |
| **世界模型** | 不需要 | 需要 | 需要 |
| **典型对比意义** | 上界参考 | 与 C 对比，量化引导的作用 | 基准线 |

---

## Setting A — Oracle

把 `atomic_facts` 直接嵌入 prompt 交给模型，单轮生成答案。不涉及工具调用，用于衡量"如果信息已完全提供，模型能否正确整合并回答"。

### vLLM 版本

```bash
python experiments/setting_A_oracle/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/vllm_results.jsonl \
    --model <model_name> \
    --api-base http://<vllm_host>:8000/v1 \
    --max-concurrent 64
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入 JSONL 文件 |
| `--output` | 必填 | 输出 JSONL 文件 |
| `--model` | `models` | 模型名称 |
| `--api-base` | `http://10.72.8.1:8000/v1` | vLLM 服务地址 |
| `--max-concurrent` | `64` | 最大并发样本数 |
| `--max-tokens` | `2048` | 最大生成 token 数 |
| `--temperature` | `0.0` | 采样温度 |
| `--enable-thinking` | - | 启用 thinking 模式 |
| `--max-retries` | `3` | 失败重试次数 |

### API 版本

```bash
python experiments/setting_A_oracle/api/async_inference_oracle_api.py \
    --provider openai \
    --model gpt-4o \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/api_results.jsonl \
    --qps 5 \
    --max-concurrent 32
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--provider` | 必填 | `openai` / `azure` / `custom` |
| `--model` | 必填 | 模型名称 |
| `--qps` | `2.0` | 每秒请求数限制 |
| `--max-concurrent` | `32` | 最大并发数 |
| `--max-tokens` | `2048` | 最大生成 token 数 |
| `--temperature` | `0.0` | 采样温度 |
| `--base-url` | - | Custom provider 的 base URL |
| `--resume` | - | 断点续传 |

### 评测

```bash
bash experiments/setting_A_oracle/evaluate/run_judge_evaluation.sh
```

---

## Setting B — Guided

在 Setting C 基础上，给模型提供查询分解引导（`guidance_prompt` 或 `fewshot_prompt`），观察引导是否能改善检索质量与最终正确率。通过与 Setting C 的结果对比，可以量化 guided prompt 的作用。

两种 prompt 变体：
- **`guidance_prompt`**：原子化查询原则说明
- **`fewshot_prompt`**：原则 + 4 个带正反例的查询分解示例

> 世界模型需要独立部署，见下方[世界模型配置](#世界模型配置)。

### vLLM 版本

```bash
python experiments/setting_B_guided/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/vllm_results.jsonl \
    --model <model_name> \
    --api-base http://<vllm_host>:8000/v1 \
    --prompt-type fewshot_prompt \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入 JSONL 文件 |
| `--output` | 必填 | 输出 JSONL 文件 |
| `--model` | `models` | 模型名称 |
| `--api-base` | - | vLLM 服务地址 |
| `--prompt-type` | `fewshot_prompt` | `fewshot_prompt` 或 `guidance_prompt` |
| `--max-concurrent-turns` | `32` | 最大并发 turn 数 |
| `--max-turns-per-sample` | `8` | 每样本最大轮次 |
| `--max-context-chars` | `60000` | 上下文最大字符数 |
| `--enable-thinking` | - | 启用 thinking 模式 |
| `--disable-thinking` | - | 显式关闭 thinking 模式 |
| `--max-retries` | `5` | 失败重试次数 |

世界模型节点通过环境变量配置：

```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
```

### API 版本

**guidance_prompt（默认）：**

```bash
python experiments/setting_B_guided/api/async_inference_api.py \
    --provider openai \
    --model gpt-4o \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/api_guidance_results.jsonl \
    --qps 10 \
    --max-concurrent-turns 100 \
    --world-model-vllm-url http://<world_model_host>:8000/v1
```

**fewshot_prompt（消融实验）：**

```bash
python experiments/setting_B_guided/api/async_inference_api_fewshot.py \
    --provider openai \
    --model gpt-4o \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/api_fewshot_results.jsonl \
    --qps 10 \
    --max-concurrent-turns 100 \
    --world-model-vllm-url http://<world_model_host>:8000/v1
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--provider` | 必填 | `openai` / `azure` / `custom` |
| `--model` | 必填 | 模型名称 |
| `--qps` | `10.0` | 每秒请求数限制 |
| `--qpm` | - | 每分钟请求数限制（可选） |
| `--max-concurrent-turns` | `100` | 最大并发 turn 数 |
| `--max-turns-per-sample` | `32` | 每样本最大轮次 |
| `--world-model-vllm-url` | - | 世界模型地址，多节点用逗号分隔 |
| `--world-model-provider` | `vllm` | 世界模型 provider |
| `--resume` | - | 断点续传 |

### 评测

```bash
bash experiments/setting_B_guided/evaluate/run_judge_evaluation.sh
```

---

## Setting C — Unguided

标准多轮 ReAct agent 评测，不提供任何查询分解引导，模型自主规划工具调用策略。这是最能体现端到端 agent 能力的设置，也是论文中的主要基准线。

> 世界模型需要独立部署，见下方[世界模型配置](#世界模型配置)。

### vLLM 版本

```bash
python experiments/setting_C_unguided/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/vllm_results.jsonl \
    --model <model_name> \
    --api-base http://<vllm_host>:8000/v1 \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入 JSONL 文件 |
| `--output` | 必填 | 输出 JSONL 文件 |
| `--model` | `models` | 模型名称 |
| `--api-base` | - | vLLM 服务地址 |
| `--max-concurrent-turns` | `32` | 最大并发 turn 数 |
| `--max-turns-per-sample` | `8` | 每样本最大轮次 |
| `--max-context-chars` | `60000` | 上下文最大字符数 |
| `--read-timeout` | `600.0` | API 读取超时（秒） |
| `--enable-thinking` | - | 启用 thinking 模式 |
| `--disable-thinking` | - | 显式关闭 thinking 模式 |
| `--max-retries` | `5` | 失败重试次数 |

世界模型节点通过环境变量配置：

```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
```

### API 版本

```bash
python experiments/setting_C_unguided/api/async_inference_api.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/api_results.jsonl \
    --provider openai \
    --model gpt-4o
```

### 评测

```bash
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

---

## 世界模型配置

Setting B 和 Setting C 需要额外部署一个世界模型节点，用于模拟 Parallel World 搜索引擎。

### 配置方式

**vLLM 版本**（环境变量）：

```bash
export WORLD_MODEL_ENDPOINTS=http://<host>:8000/v1
# 多节点负载均衡：
export WORLD_MODEL_ENDPOINTS=http://host1:8000/v1,http://host2:8000/v1
```

**API 版本**（命令行参数）：

```bash
--world-model-vllm-url http://<host>:8000/v1
# 多节点：
--world-model-vllm-url http://host1:8000/v1,http://host2:8000/v1
```

未配置时默认回退到 `http://localhost:8000/v1`。

### 世界模型的作用

世界模型接收 agent 的搜索查询，结合样本的 `atomic_facts` 判断查询命中情况，并返回模拟的搜索结果。每次工具调用的命中情况会记录在 `trajectory_log.hit_logs` 中，用于后续 FCR 计算。

---

## 完整评测流程

```
推理（vllm/ 或 api/）
    ↓
轨迹修复（evaluation/pipeline/step1_fix_trajectories.py）
    ↓
LLM-as-Judge 评测（evaluation/pipeline/step2_evaluate.py）
    ↓
生成主表（evaluation/pipeline/step3_generate_tables.py）
    ↓
分析（analysis/）
```

详细用法见根目录 [README.md](../README.md) 中的"常见工作流"章节。

---

## 自定义评测

如需使用自定义 System Prompt 或自定义工具，请参考 [custom/README.md](custom/README.md)。
