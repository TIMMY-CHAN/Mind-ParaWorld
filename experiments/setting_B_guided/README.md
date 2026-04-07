# Setting B: Guided Search (Few-shot Query Decomposition)

**Experiment Type**: Guided Search - Few-shot Query Decomposition

## Experiment Setting

In this experiment setting, the model retrieves information through the web_search tool, but the **System Prompt contains Few-shot query decomposition examples** that clearly demonstrate:
- What makes a good query vs. a bad query
- How to decompose complex questions into atomic queries
- Core principles of query decomposition

### Core Features

1. **Few-shot Query Decomposition Guide**
   - 4 detailed examples (comparison, time difference calculation, conditional filtering, multi-dimensional comparison)
   - Each example shows "bad query" vs. "good query" comparison
   - Clear step-by-step decomposition instructions

2. **Core Principles**
   - Atomic queries: One query focuses on one entity, one attribute
   - Collect then calculate: Don't expect search engines to do calculations
   - Explicit entities: Use specific names, avoid vague references
   - Step-by-step reasoning: Complex question = multiple simple queries + your own reasoning

## Directory Structure

```
setting_B_guided/
├── api/                                      # API version (implemented)
│   ├── async_inference_api_fewshot.py       # Few-shot API inference engine
│   └── run_fewshot_inference.sh             # Execution script
└── vllm/                                     # vLLM version (implemented)
    ├── async_inference_v2_fewshot.py        # Few-shot vLLM inference engine
    └── run_vllm_fewshot.sh                  # Execution script
```

## Usage

### API Version (Implemented)

```bash
# Method 1: Use execution script (recommended)
cd experiments/setting_B_guided/api
./run_fewshot_inference.sh

# Method 2: Direct Python script
python experiments/setting_B_guided/api/async_inference_api_fewshot.py \
    --provider licloud \
    --model Qwen/Qwen2-VL-72B-Instruct-AWQ \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/inference_results.jsonl \
    --qps 2 \
    --max-concurrent-turns 32
```

### vLLM Version (Implemented)

```bash
# Method 1: Use execution script (recommended)
cd experiments/setting_B_guided/vllm
./run_vllm_fewshot.sh

# Method 2: Direct Python script
python experiments/setting_B_guided/vllm/async_inference_v2_fewshot.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B_vllm/inference_results.jsonl \
    --model models \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16 \
    --max-context-chars 60000 \
    --max-retries 5
```

**Configuration Notes**:
- `--max-concurrent-turns`: Turn-level maximum concurrency (recommended 32-128)
- `--max-turns-per-sample`: Maximum turns per sample (recommended 16-32)
- `--max-context-chars`: Context window character count (recommended 128000 for 128K models)
- `--max-retries`: Maximum retry attempts for API network errors (exponential backoff, default 5)
- `--enable-thinking`: Force enable model's built-in thinking mode
- `--disable-thinking`: Force disable model's built-in thinking mode (omit to use model default behavior)

## Prompt Comparison

### Few-shot System Prompt (Setting B)

```python
INSTRUCTION_PROMPT_SYSTEM_FEWSHOT = """
You are a ReAct paradigm multimodal agent...

---

# Query Decomposition Guide

## Example 1: Comparison Questions
❌ Bad query: Compare Juventus and Napoli's away goals in the 2022-23 season
✅ Good query:
  Step 1: Search "2022-23 season Juventus away goals"
  Step 2: Search "2022-23 season Napoli away goals"
  Step 3: Compare the two values

## Examples 2-4: (More decomposition examples)
...

## Core Principles
1. Atomic queries
2. Collect then calculate
3. Explicit entities
4. Step-by-step reasoning
"""
```

### Zero-shot Baseline (Setting C)

```python
INSTRUCTION_PROMPT_SYSTEM = """
You are a ReAct paradigm multimodal agent...
Available tools:
web_search:
...
"""
```

**Key Difference**: Setting B adds 147 lines of query decomposition guide and examples

## Experiment Purpose

By comparing Setting B (with query guidance) and Setting C (without guidance), evaluate:
1. Effectiveness of Few-shot Query Decomposition
2. Impact of query quality on retrieval effectiveness
3. Whether models can learn to decompose complex queries

## Evaluation Metrics

Focus on:
- **FCR (Fact Coverage Rate)**: Query quality directly affects fact coverage rate
- **Hit Precision**: Atomic queries should improve hit precision
- **Avg Tool Calls**: May increase (due to decomposition into more queries)
- **Pass@1**: Whether final accuracy improves

## Comparison with Other Settings

| Setting | Prompt Type | Query Guidance | Code Line Difference |
|---------|-----------|----------|-------------|
| **Setting B (Guided)** | Few-shot | ✅ 4 examples + core principles | +147 lines |
| **Setting C (Unguided)** | Zero-shot | ❌ None | Baseline |

---

**Last Updated**: 2026-02-10
**Status**: Both API and vLLM versions implemented
