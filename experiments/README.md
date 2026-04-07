# Experiments

This directory contains three official evaluation settings (Setting A / B / C) for the MPW benchmark, as well as the custom prompt evaluation entry point.

```
experiments/
├── setting_A_oracle/       # Oracle: directly provide atomic facts, single-turn response
├── setting_B_guided/       # Guided: multi-turn ReAct + query decomposition guidance
├── setting_C_unguided/     # Unguided: multi-turn ReAct + standard prompt
└── custom/                 # Custom prompt evaluation (see custom/README.md)
```

---

## Three Settings Comparison

| | **Setting A — Oracle** | **Setting B — Guided** | **Setting C — Unguided** |
|---|---|---|---|
| **Core Purpose** | Estimate reasoning upper bound when information is sufficient | Evaluate query decomposition guidance benefit for agents | Evaluate end-to-end agent real capabilities |
| **Tool Calling** | ❌ None | ✅ Multi-turn `web_search` | ✅ Multi-turn `web_search` |
| **System Prompt** | Given atomic facts + single-turn QA | Guided prompt with query decomposition examples | Standard ReAct prompt |
| **World Model** | Not required | Required | Required |
| **Comparison Significance** | Upper bound reference | Compare with C to quantify guidance effect | Baseline |

---

## Setting A — Oracle

Embed `atomic_facts` directly into the prompt for single-turn answer generation. No tool calling involved, used to measure "if information is fully provided, can the model correctly integrate and answer?"

### vLLM Version

```bash
python experiments/setting_A_oracle/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/vllm_results.jsonl \
    --model <model_name> \
    --api-base http://<vllm_host>:8000/v1 \
    --max-concurrent 64
```

| Parameter | Default | Description |
|------|--------|------|
| `--input` | Required | Input JSONL file |
| `--output` | Required | Output JSONL file |
| `--model` | `models` | Model name |
| `--api-base` | `http://10.72.8.1:8000/v1` | vLLM service address |
| `--max-concurrent` | `64` | Maximum concurrent samples |
| `--max-tokens` | `2048` | Maximum generation tokens |
| `--temperature` | `0.0` | Sampling temperature |
| `--enable-thinking` | - | Enable thinking mode |
| `--max-retries` | `3` | Maximum retry attempts |

### API Version

```bash
python experiments/setting_A_oracle/api/async_inference_oracle_api.py \
    --provider openai \
    --model gpt-4o \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/api_results.jsonl \
    --qps 5 \
    --max-concurrent 32
```

| Parameter | Default | Description |
|------|--------|------|
| `--provider` | Required | `openai` / `azure` / `custom` |
| `--model` | Required | Model name |
| `--qps` | `2.0` | Requests per second limit |
| `--max-concurrent` | `32` | Maximum concurrent requests |
| `--max-tokens` | `2048` | Maximum generation tokens |
| `--temperature` | `0.0` | Sampling temperature |
| `--base-url` | - | Base URL for custom provider |
| `--resume` | - | Resume from interruption |

### Evaluation

```bash
bash experiments/setting_A_oracle/evaluate/run_judge_evaluation.sh
```

---

## Setting B — Guided

On top of Setting C, provide query decomposition guidance (`guidance_prompt` or `fewshot_prompt`) in the system prompt to observe whether guidance improves retrieval quality and final accuracy. Comparison with Setting C quantifies the effect of guided prompts.

Two prompt variants:
- **`guidance_prompt`**: Atomic query principle explanation
- **`fewshot_prompt`**: Principles + 4 query decomposition examples with positive/negative cases

> World model needs independent deployment, see [World Model Configuration](#world-model-configuration) below.

### vLLM Version

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

| Parameter | Default | Description |
|------|--------|------|
| `--input` | Required | Input JSONL file |
| `--output` | Required | Output JSONL file |
| `--model` | `models` | Model name |
| `--api-base` | - | vLLM service address |
| `--prompt-type` | `fewshot_prompt` | `fewshot_prompt` or `guidance_prompt` |
| `--max-concurrent-turns` | `32` | Maximum concurrent turns |
| `--max-turns-per-sample` | `8` | Maximum turns per sample |
| `--max-context-chars` | `60000` | Maximum context characters |
| `--enable-thinking` | - | Enable thinking mode |
| `--disable-thinking` | - | Explicitly disable thinking mode |
| `--max-retries` | `5` | Maximum retry attempts |

World model endpoints configured via environment variable:

```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
```

### API Version

**guidance_prompt (default):**

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

**fewshot_prompt (ablation study):**

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

| Parameter | Default | Description |
|------|--------|------|
| `--provider` | Required | `openai` / `azure` / `custom` |
| `--model` | Required | Model name |
| `--qps` | `10.0` | Requests per second limit |
| `--qpm` | - | Requests per minute limit (optional) |
| `--max-concurrent-turns` | `100` | Maximum concurrent turns |
| `--max-turns-per-sample` | `32` | Maximum turns per sample |
| `--world-model-vllm-url` | - | World model address, comma-separated for multiple nodes |
| `--world-model-provider` | `vllm` | World model provider |
| `--resume` | - | Resume from interruption |

### Evaluation

```bash
bash experiments/setting_B_guided/evaluate/run_judge_evaluation.sh
```

---

## Setting C — Unguided

Standard multi-turn ReAct agent evaluation without any query decomposition guidance. The model autonomously plans tool call strategies. This is the setting that best reflects end-to-end agent capabilities and serves as the main baseline in the paper.

> World model needs independent deployment, see [World Model Configuration](#world-model-configuration) below.

### vLLM Version

```bash
python experiments/setting_C_unguided/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/vllm_results.jsonl \
    --model <model_name> \
    --api-base http://<vllm_host>:8000/v1 \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16
```

| Parameter | Default | Description |
|------|--------|------|
| `--input` | Required | Input JSONL file |
| `--output` | Required | Output JSONL file |
| `--model` | `models` | Model name |
| `--api-base` | - | vLLM service address |
| `--max-concurrent-turns` | `32` | Maximum concurrent turns |
| `--max-turns-per-sample` | `8` | Maximum turns per sample |
| `--max-context-chars` | `60000` | Maximum context characters |
| `--read-timeout` | `600.0` | API read timeout (seconds) |
| `--enable-thinking` | - | Enable thinking mode |
| `--disable-thinking` | - | Explicitly disable thinking mode |
| `--max-retries` | `5` | Maximum retry attempts |

World model endpoints configured via environment variable:

```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
```

### API Version

```bash
python experiments/setting_C_unguided/api/async_inference_api.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/api_results.jsonl \
    --provider openai \
    --model gpt-4o
```

### Evaluation

```bash
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

---

## World Model Configuration

Setting B and Setting C require additional world model node deployment to simulate the Parallel World search engine.

### Configuration

**vLLM Version** (environment variable):

```bash
export WORLD_MODEL_ENDPOINTS=http://<host>:8000/v1
# Multi-node load balancing:
export WORLD_MODEL_ENDPOINTS=http://host1:8000/v1,http://host2:8000/v1
```

**API Version** (command-line argument):

```bash
--world-model-vllm-url http://<host>:8000/v1
# Multi-node:
--world-model-vllm-url http://host1:8000/v1,http://host2:8000/v1
```

Defaults to `http://localhost:8000/v1` if not configured.

### World Model Role

The world model receives agent search queries, determines query hit status based on sample's `atomic_facts`, and returns simulated search results. Each tool call's hit status is recorded in `trajectory_log.hit_logs` for subsequent FCR calculation.

---

## Complete Evaluation Pipeline

```
Inference (vllm/ or api/)
    ↓
Trajectory Repair (evaluation/pipeline/step1_fix_trajectories.py)
    ↓
LLM-as-Judge Evaluation (evaluation/pipeline/step2_evaluate.py)
    ↓
Generate Main Table (evaluation/pipeline/step3_generate_tables.py)
    ↓
Analysis (analysis/)
```

See the "Common Workflows" section in the root [README.md](../README.md) for detailed usage.

---

## Custom Evaluation

To use custom System Prompt or custom tools, see [custom/README.md](custom/README.md).
