---
name: mpw-custom
description: Set up and run custom agent evaluation on any dataset. Use when evaluating a model with your own benchmark data, testing a custom system prompt or tool strategy, or comparing prompting approaches.
disable-model-invocation: true
---

# Custom Agent Evaluation Setup

Interactively guide the user through setting up and running a custom evaluation using the Mind-ParaWorld evaluation framework. This framework is **general-purpose** — it supports any dataset, any system prompt, and any model family. Work through the steps below in order, asking one question at a time, and generate ready-to-run commands only after all required information is collected.

## What the framework provides

- Multi-turn ReAct agent loop with pluggable tool calling
- Six built-in parsers: `default` (Qwen3/Hermes), `deepseek`, `glm4`, `minimax`, `qwen35`, `kimi_k2`
- Optional parallel tool calls (multiple searches per turn)
- Built-in tools: `web_search`, `FlashRAGSearchTool`, `PythonCodeInterpreterTool`
- LLM-as-Judge evaluation

## Key paths

```
experiments/custom/
  vllm/async_inference.py             # inference (local vLLM)
  api/async_inference.py              # inference (OpenAI-compat API)
  evaluate/run_judge_evaluation.sh    # LLM-as-Judge scoring
  prompts/
    example_prompt.py                 # starter prompt (single tool call)
    parallel_search_prompt.py         # parallel tool calls example
verl/workers/agent/envs/tools/        # built-in tool implementations
verl/workers/agent/parsers.py         # tool call parsers
```

## Step 1 — Understand the task

Ask:
> "What are you trying to evaluate? (e.g. a QA benchmark, reasoning dataset, custom task)"

Use the answer to tailor advice in later steps.

## Step 2 — Data preparation

The framework accepts JSONL. Each line:
```json
{
  "index": 0,
  "prompt": [{"role": "user", "content": "Your question or task here"}],
  "answer": "ground truth (used by LLM-as-Judge)",
  "extra_info": {}
}
```

- `answer`: reference for the judge; any string format works
- `extra_info`: optional metadata, use `{}` if unused

If the user has data in another format (CSV, HuggingFace, etc.), write a conversion script on the spot. Example for CSV:
```python
import json, csv
with open("my_data.csv") as f, open("my_data.jsonl", "w") as out:
    for i, row in enumerate(csv.DictReader(f)):
        out.write(json.dumps({
            "index": i,
            "prompt": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"],
            "extra_info": {}
        }) + "\n")
```

Tip: `head -n 10 my_data.jsonl > test.jsonl` to build a quick smoke-test set.

If they have no data yet, note that the MPW benchmark data is not bundled in the repo (it will be available as a Gated Dataset on HuggingFace). Suggest they use any other JSONL dataset they have, or create a small synthetic one to try the framework.

## Step 3 — System prompt

Ask:
> "Do you have a system prompt file ready, or would you like to start from the built-in example?"

The prompt file is a Python file defining `system_prompt`. It must include:
1. Tool definitions and their call format
2. Final answer format: `<answer>...</answer>`
3. Task-specific behavior rules

Minimal template (`default` parser):
```python
# my_prompt.py
system_prompt = """You are a helpful agent that answers questions by searching for information.

## Available Tools

web_search: Search for relevant information.
<tool_call>{"name": "web_search", "arguments": {"query": "your query"}}</tool_call>

## Rules
- Use tools when you need external information
- End with: <answer>your answer</answer>
"""
```

Point to `experiments/custom/prompts/example_prompt.py` as a complete working reference.

If the task needs no tools (pure reasoning), omit the tool section — the framework completes in a single turn.

## Step 4 — Model deployment

Ask:
> "How is your model deployed? (a) Local vLLM server  (b) Remote/commercial API"

- **(a) vLLM**: collect `--api-base` (e.g. `http://localhost:8000/v1`) and `--model`
- **(b) API**: collect `--provider` (`openai` / `azure` / `custom`), `--model`, and `--base-url` if self-hosted

## Step 5 — Parser

Ask:
> "Which model family? This determines how the framework parses tool calls from the model's text output."

| `--parser` | Model family |
|------------|-------------|
| `default`  | Qwen3, Hermes, NousResearch — `<tool_call>{...}</tool_call>` |
| `deepseek` | DeepSeek-V2 / V3 / R1 |
| `glm4`     | GLM-4 / GLM-Z1 |
| `minimax`  | MiniMax-M2.5 |
| `qwen35`   | Qwen3.5 |
| `kimi_k2`  | Kimi-K2 |

Default to `default` if unsure. Remind: **the tool call format in the prompt must match the parser**.

## Step 6 — Parallel tool calls (optional)

Ask:
> "Does your task involve independent sub-questions that could be searched simultaneously?"

- Yes → add `--parallel-tool-calls`; recommend `experiments/custom/prompts/parallel_search_prompt.py` as reference
- No → skip

Requires `--parser default`.

## Step 7 — Custom tools (optional)

Ask:
> "Do you need tools beyond `web_search`? Built-ins: `FlashRAGSearchTool` (local RAG), `PythonCodeInterpreterTool` (sandboxed Python)."

If yes, show how to register:
```python
from verl.workers.agent.envs.tools.python_code_interpreter import PythonCodeInterpreterTool
from verl.workers.agent.envs.agent_eval import AgentEval

agent = AgentEval(tools=[PythonCodeInterpreterTool()])
```

Custom tools: subclass `ToolBase` in `verl/workers/agent/tool_envs.py`.

## Step 8 — Generate commands

Suggest output path: `results/custom/<task_name>/<model_name>/inference_results.jsonl`

**vLLM:**
```bash
python experiments/custom/vllm/async_inference.py \
    --input <data.jsonl> \
    --output <output.jsonl> \
    --prompt-file <prompt.py> \
    --model <model_name> \
    --api-base <api_base> \
    --parser <parser> \
    --max-concurrent-turns 32 \
    --max-turns-per-sample 8 \
    [--parallel-tool-calls]
```

**API — text mode** (mirrors vLLM behavior, for cross-deployment comparison):
```bash
python experiments/custom/api/async_inference.py \
    --provider <provider> --model <model_name> [--base-url <url>] \
    --input <data.jsonl> --output <output.jsonl> \
    --prompt-file <prompt.py> \
    --tool-mode text --parser <parser> \
    --qps 10 --max-concurrent-turns 100 --max-turns-per-sample 8 \
    --resume [--parallel-tool-calls]
```

**API — native mode** (commercial APIs, uses built-in function calling):
```bash
python experiments/custom/api/async_inference.py \
    --provider <provider> --model <model_name> \
    --input <data.jsonl> --output <output.jsonl> \
    --prompt-file <prompt.py> \
    --tool-mode native \
    --qps 10 --max-concurrent-turns 100 --max-turns-per-sample 8 \
    --resume
```

**Evaluation:**
```bash
bash experiments/custom/evaluate/run_judge_evaluation.sh
# Interactive: select the inference results file to score
```

## Debugging tips

- **Model never calls tools** → prompt format doesn't match parser; verify the tool call format example
- **All answers wrong** → check `answer` field format in JSONL; adjust the judge prompt if needed
- **Run interrupted** → rerun with `--resume` to skip completed samples
