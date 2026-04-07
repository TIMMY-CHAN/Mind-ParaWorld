# Setting C: Unguided Search — LLM-as-Judge Evaluation

This directory contains the LLM-as-Judge scripts for evaluating Setting C inference results.

## Directory Structure

```
evaluate/
├── llm-as-judge.py              # LLM-as-Judge evaluation engine
├── run_judge_evaluation.sh      # One-click evaluation script
└── README.md
```

## Use Cases

1. **First Evaluation**: Perform Correct / Incorrect judgment on inference result files
2. **Change Judge**: Re-evaluate with different Judge models to compare scoring consistency
3. **Verify Results**: Spot-check accuracy of existing evaluation results

## Quick Start

### Method 1: One-click Script (Recommended)

```bash
bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

The script interactively prompts for Judge model address, result file to evaluate, and other configurations.

### Method 2: Direct Python Script

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

## Configuration Parameters

### Judge Model

```bash
JUDGE_MODEL="<judge_model_name>"
JUDGE_API_BASE="http://<judge_host>:8000/v1"
```

### Concurrency and Retry

```bash
MAX_CONCURRENT=32      # Judge concurrency
MAX_RETRIES=3          # Maximum retry attempts
MAX_TOKENS=8192        # Judge maximum tokens
TEMPERATURE=0.8        # Judge sampling temperature
```

### Force Re-evaluation

```bash
FORCE_REEVALUATE=false  # Set to true to ignore existing results and re-evaluate all
```

## Output Format

Each sample gets new fields added:

```json
{
  "answer": "Correct",
  "think": "Judgment reasoning",
  "judge_response": "Judge full response",
  "judge_status": "success"
}
```

The old `evaluation` field will be removed.

## Resume from Interruption

If evaluation is interrupted, re-running the script will automatically skip completed samples. To re-evaluate all samples, set `FORCE_REEVALUATE=true`.

## Judge Prompt

Uses the same Judge Prompt as Setting B to ensure consistent scoring standards across settings:

```
You are a professional answer evaluation expert. Your task is to judge whether the model's predicted answer is consistent with the reference answer.

Evaluation criteria:
1. If the predicted answer is semantically consistent with the reference answer, judge as "Correct"
2. If the predicted answer is inconsistent with the reference answer, missing key information, or completely wrong, judge as "Incorrect"
```

## Differences from Setting B

| Feature | Setting B | Setting C |
|------|-----------|-----------|
| Input file | `inference_results.jsonl` | `evaluated_results.jsonl` |
| Has old evaluation field | No | Yes (will be removed) |
| Output filename | `evaluated_results.jsonl` | `re_evaluated_results.jsonl` |
| Force re-evaluation option | Not supported | Supported |

## Common Issues

**Input file not found**: Confirm inference is complete, file path and naming are correct.

**Judge API connection failed**: Confirm Judge service is started, address and port are configured correctly.

**Out of memory**: Lower `MAX_CONCURRENT` (e.g., 32 → 16) or `MAX_TOKENS` (e.g., 8192 → 4096).
