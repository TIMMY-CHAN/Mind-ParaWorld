---
name: mpw-bench
description: Run the official Mind-ParaWorld benchmark pipeline. Use when reproducing paper results or evaluating a new model on MPW Setting A (Oracle), B (Guided Search), or C (Unguided Search).
disable-model-invocation: true
---

# Mind-ParaWorld Benchmark Evaluation

Interactively guide the user through running the official MPW benchmark evaluation. Work through the steps below in order, collecting the necessary information before generating ready-to-run commands.

## Settings overview

| Setting | Description | World Model needed |
|---------|-------------|-------------------|
| **A — Oracle** | Atomic facts injected into prompt; single-turn QA. Upper-bound on reasoning given perfect information. | No |
| **B — Guided** | Multi-turn ReAct + query-decomposition guidance in the prompt. Quantifies how much guidance helps vs. Setting C. | Yes |
| **C — Unguided** | Standard multi-turn ReAct; no guidance. End-to-end agent capability baseline. | Yes |

## Key paths

```
experiments/
  setting_A_oracle/
    vllm/async_inference.py
    api/async_inference_oracle_api.py
    evaluate/run_judge_evaluation.sh
  setting_B_guided/
    vllm/async_inference.py
    api/async_inference_api.py           # guidance_prompt
    api/async_inference_api_fewshot.py   # fewshot_prompt (ablation)
    evaluate/run_judge_evaluation.sh
  setting_C_unguided/
    vllm/async_inference.py
    api/async_inference_api.py
    evaluate/run_judge_evaluation.sh
data/mpw_bench_full.jsonl               # official benchmark data
```

## Data availability

The MPW benchmark data (`data/mpw_bench_full.jsonl`) is **not bundled in this repository**. It will be released as a Gated Dataset on HuggingFace. If the user does not yet have the data file, direct them to `data/README.md` for instructions on how to request access.

All commands below assume the data has been placed at `data/mpw_bench_full.jsonl`.

## Step 1 — Clarify the goal

Ask:
> "Which setting(s) do you want to run? (A / B / C / all three for full paper reproduction)"

Common use cases:
- **Reproduce paper results**: run all three settings for the same model
- **Quick upper-bound check**: Setting A only (no world model required)
- **Measure guidance effect**: Settings B and C, then compare
- **Full end-to-end agent eval**: Setting C only

## Step 2 — Subject model deployment

Ask:
> "How is the subject model (the model being evaluated) deployed? (a) Local vLLM server  (b) Remote/commercial API"

- **(a) vLLM**: collect `--api-base` (e.g. `http://localhost:8000/v1`) and `--model`
- **(b) API**: collect `--provider` (`openai` / `azure` / `custom`), `--model`, and `--base-url` if self-hosted

## Step 3 — World model configuration (Settings B and C only)

Skip this step if only running Setting A.

Ask:
> "Where is the world model deployed? Provide the vLLM endpoint URL(s)."

The world model simulates the Parallel World search engine. It receives the agent's queries, matches them against `atomic_facts`, and returns synthetic search results.

**vLLM inference (environment variable):**
```bash
export WORLD_MODEL_ENDPOINTS=http://<host>:8000/v1
# Multiple nodes for load balancing:
export WORLD_MODEL_ENDPOINTS=http://host1:8000/v1,http://host2:8000/v1
```

**API inference (command-line argument):**
```bash
--world-model-vllm-url http://<host>:8000/v1
# Multiple nodes:
--world-model-vllm-url http://host1:8000/v1,http://host2:8000/v1
```

Falls back to `http://localhost:8000/v1` if not configured.

## Step 4 — Generate commands

Suggest output paths: `results/setting_<X>/<model_name>/inference_results.jsonl`

---

### Setting A — Oracle

**vLLM:**
```bash
python experiments/setting_A_oracle/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/<model_name>/inference_results.jsonl \
    --model <model_name> \
    --api-base <api_base> \
    --max-concurrent 64
```

**API:**
```bash
python experiments/setting_A_oracle/api/async_inference_oracle_api.py \
    --provider <provider> --model <model_name> [--base-url <url>] \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_A/<model_name>/inference_results.jsonl \
    --qps 5 --max-concurrent 32 \
    --resume
```

Key options: `--enable-thinking` (for reasoning models), `--max-tokens 2048`, `--temperature 0.0`

**Evaluate:**
```bash
bash experiments/setting_A_oracle/evaluate/run_judge_evaluation.sh
```

---

### Setting B — Guided

Two prompt variants:
- `fewshot_prompt` (default, recommended): guidance principles + 4 positive/negative query decomposition examples
- `guidance_prompt`: principles only (lighter, for ablation)

**vLLM:**
```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
python experiments/setting_B_guided/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/<model_name>/inference_results.jsonl \
    --model <model_name> \
    --api-base <api_base> \
    --prompt-type fewshot_prompt \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16
```

**API (guidance_prompt):**
```bash
python experiments/setting_B_guided/api/async_inference_api.py \
    --provider <provider> --model <model_name> [--base-url <url>] \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/<model_name>/inference_results.jsonl \
    --world-model-vllm-url http://<world_model_host>:8000/v1 \
    --qps 10 --max-concurrent-turns 100 --max-turns-per-sample 32 \
    --resume
```

**API (fewshot_prompt ablation):**
```bash
python experiments/setting_B_guided/api/async_inference_api_fewshot.py \
    --provider <provider> --model <model_name> \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_B/<model_name>/fewshot_results.jsonl \
    --world-model-vllm-url http://<world_model_host>:8000/v1 \
    --qps 10 --max-concurrent-turns 100 \
    --resume
```

**Evaluate:**
```bash
bash experiments/setting_B_guided/evaluate/run_judge_evaluation.sh
```

---

### Setting C — Unguided

**vLLM:**
```bash
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1
python experiments/setting_C_unguided/vllm/async_inference.py \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/<model_name>/inference_results.jsonl \
    --model <model_name> \
    --api-base <api_base> \
    --max-concurrent-turns 64 \
    --max-turns-per-sample 16
```

**API:**
```bash
python experiments/setting_C_unguided/api/async_inference_api.py \
    --provider <provider> --model <model_name> [--base-url <url>] \
    --input data/mpw_bench_full.jsonl \
    --output results/setting_C/<model_name>/inference_results.jsonl \
    --qps 10 --max-concurrent-turns 100 \
    --resume
```

**Evaluate:**
```bash
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

---

## Common questions

**Q: Which world model should I use?**
Any open-source model deployable via vLLM works. The paper uses models fine-tuned on MPW data. For a fair comparison with paper results, use the same model checkpoint.

**Q: What does Setting A tell me?**
Setting A is the reasoning upper bound — it answers "if the model had perfect information, could it answer correctly?" A large A−C gap means the bottleneck is retrieval/search quality, not reasoning.

**Q: How large is the full benchmark?**
`data/mpw_bench_full.jsonl` contains the complete MPW benchmark set. Use `head -n 50 data/mpw_bench_full.jsonl > data/smoke_test.jsonl` to create a quick smoke-test subset.

**Q: Run interrupted — how do I resume?**
Add `--resume` to the API inference commands to skip already-completed samples. vLLM scripts detect existing output files automatically.

**Q: How to read the evaluation results?**
Each `run_judge_evaluation.sh` is interactive: it prompts you to select the inference results file, then runs LLM-as-Judge scoring and prints a summary table.
