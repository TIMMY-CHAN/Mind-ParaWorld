# Mind-ParaWorld (MPW)

[中文文档](README_CN.md)

**A lightweight modular async inference framework for Tool-Integrated-Reasoning Agents, supporting both local vLLM and commercial API backends, with the built-in Mind-ParaWorld (MPW) Benchmark providing a complete agentic search evaluation environment.**

The core problem this repository addresses is: **How to run an agent's inference pipeline lightly and efficiently at scale in multi-turn, tool-calling scenarios?** The framework provides turn-level async scheduling, pluggable tools and parsers, unified trajectory logging, and inference results saved as trajectory + predicted answer format, making it easy to connect to any downstream evaluation pipeline. The MPW Benchmark provides a complete agentic search evaluation environment on top of this, constructing a Parallel World search scenario isolated from training data to provide an end-to-end capability baseline for search agents.

> Paper: [Evaluating the Search Agent in a Parallel World](https://arxiv.org/abs/2603.04751)

---

## Project Overview / Why this repo

**First and foremost, a lightweight, modular agent inference framework.** You can bring your own dataset, system prompt, and tool strategy to complete large-scale multi-turn tool-use inference, with results saved as trajectory + predicted answer format, and build your own evaluation pipeline under `evaluation/`. The framework supports:

- Arbitrary JSONL format datasets, one sample per line
- Local vLLM service or any OpenAI-compatible API
- 6 built-in tool-call parsers (Qwen series, DeepSeek series, GLM-4 series, MiniMax series, Kimi-K2 series)
- Pluggable tools: built-in `web_search`, `FlashRAGSearchTool`, `PythonCodeInterpreterTool`, or custom tools with one-click registration
- Parallel tool calls (multiple tool_calls in a single turn executed simultaneously)
- Resume from interruption (`--resume`)
- Built-in Claude Code Skills for interactive evaluation configuration with AI coding assistants

**Also includes the MPW Benchmark evaluation environment.** If you need to evaluate search agents in a controlled Parallel World scenario, the framework provides complete inference and evaluation scripts for three standard settings (Oracle / Guided / Unguided), along with FCR, Hit Rate, and other metric calculation tools.

If you care about any of the following scenarios, this repository can serve as your starting point:

- Complete large-scale async inference for any multi-turn tool-use agent on your own dataset
- Compare agent capabilities across different system prompt strategies or model families
- Compare guided / unguided / oracle settings in the MPW Parallel World environment
- Perform offline evaluation, main table generation, and error analysis based on existing result files

---

## Inference/Evaluation Configuration with AI Assistants

The framework includes two [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills-tutorial) that can interactively guide you through evaluation configuration with Claude Code or compatible AI assistants (like OpenClaw), **without requiring you to read the documentation**. After cloning this repository, simply run in Claude Code:

| Skill | Use Case |
|---|---|
| `/mpw-custom` | Configure and run custom agent inference on any dataset |
| `/mpw-bench` | Reproduce paper results, or run MPW Setting A / B / C evaluations |

The assistant will step through data format, model deployment, parser selection, and ultimately generate a ready-to-run inference command. The entire process doesn't require prior knowledge of the framework structure—**let the agent assistant help you complete agent evaluation.**

---

## Core Capabilities

### 1) Turn-level Async Scheduling

The inference entry point adopts **turn-level async scheduling** instead of sample-level serial rollout. The global task queue dynamically allocates concurrent turns; when one sample is waiting for tool or model response, subsequent turns of other samples can proceed immediately, significantly improving throughput compared to serial approaches. This mechanism is fully implemented in `experiments/custom/vllm/async_inference.py`.

### 2) Agent / Tool / Environment Decoupling

Core components under `verl/`:

- **`verl/sample_state.py`**: `SampleState` manages single-sample state; `StateManager` handles global progress tracking and result export.
- **`verl/workers/agent/tool_envs.py`**: `ToolBase` defines tool abstraction, `ToolMeta` auto-registers subclasses.
- **`verl/workers/agent/envs/agent_eval.py`**: `AgentEval` parses tool calls, executes tools, collects trajectory logs, supports pluggable tool lists and multiple parsers.
- **`verl/workers/agent/envs/tools/world_model_web_search_tool.py`**: `WorldModelWebSearchTool` simulates search engine via world model, supports endpoint pools, caching, and async retries.

### 3) Complete Experiment Loop

Inference → LLM-as-Judge evaluation → main table generation → quantitative analysis. Each stage has independent scripts with unified result file formats, allowing you to plug into any stage as needed.

---

## System Components / Repository Structure

```text
Mind-ParaWorld/
├── data/                         # Data examples and format description
├── experiments/                  # Inference entry points
│   ├── setting_A_oracle/         # MPW Oracle: directly provide facts, single-turn
│   ├── setting_B_guided/         # MPW Guided: multi-turn + query decomposition guidance
│   ├── setting_C_unguided/       # MPW Unguided: multi-turn standard ReAct
│   └── custom/                   # Custom dataset inference entry
├── evaluation/                   # Evaluation pipeline (user-defined or connect to MPW judge)
├── verl/                         # agent-loop core components
│   ├── sample_state.py
│   └── workers/agent/
│       ├── tool_envs.py
│       └── envs/
│           ├── agent_eval.py
│           └── tools/world_model_web_search_tool.py
└── analysis/                     # Quantitative analysis, error attribution, main tables, and visualization
```

To quickly understand this repository from an engineering perspective, we recommend prioritizing:

- `experiments/custom/vllm/async_inference.py`
- `verl/sample_state.py`
- `verl/workers/agent/envs/agent_eval.py`
- `verl/workers/agent/envs/tools/world_model_web_search_tool.py`

---

## Quick Start

### 1) Basic Environment

```bash
pip install openai httpx pillow tqdm
export PYTHONPATH=$(pwd)
```

### 2) Data

The framework accepts JSONL format, one sample per line:

```json
{"index": 0, "prompt": [{"role": "user", "content": "Your question"}], "answer": "Reference answer", "extra_info": {}}
```

`data/mpw_bench.jsonl` contains three examples for format reference or smoke testing. See [`data/README.md`](data/README.md) for details.

---

## Common Workflows

For complete parameter descriptions, see [`experiments/README.md`](experiments/README.md).

### Custom Agent Evaluation (General Entry)

For open-source community users, the most direct starting point is to bring your own data and use the inference scripts under `custom/` without any MPW dependencies:

```bash
# vLLM backend
python experiments/custom/vllm/async_inference.py \
  --input data/your_data.jsonl \
  --output results/my_model/inference_results.jsonl \
  --model <model_name> \
  --api-base http://<vllm_host>:8000/v1 \
  --parser default \
  --max-concurrent-turns 64 \
  --max-turns-per-sample 16
```

For commercial APIs, switch to `experiments/custom/api/async_inference.py` with the same parameters. Evaluation results are processed by `experiments/custom/evaluate/llm-as-judge.py`.

### Running MPW Setting C (Unguided, MPW-bench specific)

The following workflow **only applies to MPW Benchmark evaluation** and requires additional World Model deployment to provide the simulated search environment:

```bash
# Deploy world model then set endpoint
export WORLD_MODEL_ENDPOINTS=http://<world_model_host>:8000/v1

python experiments/setting_C_unguided/vllm/async_inference.py \
  --input data/mpw_bench_full.jsonl \
  --output results/my_model/inference_results.jsonl \
  --model <model_name> \
  --api-base http://<vllm_host>:8000/v1 \
  --max-concurrent-turns 64 \
  --max-turns-per-sample 16

bash experiments/setting_C_unguided/evaluate/run_judge_evaluation.sh
```

### Generating Main Results Table

```bash
python analysis/utils/generate_main_table.py \
  --inputs results/model1/evaluated_results.jsonl results/model2/evaluated_results.jsonl \
  --labels model1 model2 \
  --output main_table.md
```

---

## MPW Benchmark Introduction

MPW's core concept: evaluate agents in a **Parallel World** search environment occurring in the future, constructing a virtual search environment isolated from model knowledge, rather than relying on the open real Web.

The basic approach is:

1. Maintain a set of agent-unaware atomic facts for each question
2. Agent progressively retrieves these facts through `web_search` queries
3. Judge response quality based on fact coverage and evaluation results

The benchmark contains **19 categories, 1608 questions**. The dataset is planned for release via HuggingFace Gated Dataset (see License section). Each sample's `extra_info.world_truth_info.atomic_facts` provides the Parallel World's atomic fact set.

### Three Evaluation Settings

| Setting | Purpose | Characteristics |
|---|---|---|
| **A — Oracle** | Evaluate integration and reasoning capabilities under sufficient information | Directly provide atomic facts, single-turn response |
| **B — Guided** | Evaluate whether "prompt-guided query decomposition" helps agents | Multi-turn ReAct + guided prompt |
| **C — Unguided** | Evaluate end-to-end agent capabilities | Multi-turn ReAct + standard prompt |

---

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| **Pass@1** | Proportion of final answers judged correct by the Judge |
| **FCR** | Fact Coverage Rate, proportion of atomic facts hit |
| **Hit Rate / Hit Precision** | Tool call hit efficiency |
| **Avg Turns** | Average dialogue turns per sample |

The main table reports `Pass@1 | FCR | Hit Rate | Avg Turns`. Aggregation logic in `analysis/utils/generate_main_table.py`.

---

## Result Analysis

`analysis/` covers complete post-analysis from result tables to error attribution, quantitative statistics, and visualization. See [`analysis/README.md`](analysis/README.md).

```
analysis/
├── core/      # Comprehensive analysis, error analysis, multi-model comparison
├── tools/     # Error attribution, quantitative analysis, comprehensive reports, visualization
├── utils/     # Metric calculation, difficulty stratification, main table generation, Excel conversion
└── scripts/   # Batch processing scripts and specialized plotting scripts
```

---

## Citation

```bibtex
@misc{chen2026evaluatingsearchagentparallel,
      title={Evaluating the Search Agent in a Parallel World},
      author={Jiawei Chen and Xintian Shen and Lihao Zheng and Lifu Mu and Haoyi Sun and Ning Mao and Hao Ma and Tao Wei and Pan Zhou and Kun Zhan},
      year={2026},
      eprint={2603.04751},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.04751},
}
```

## License

**Code framework** is released under the [Apache License 2.0](LICENSE).

**MPW Benchmark dataset** is not publicly released with the code repository for the following reasons:

**Content Review**: The dataset is constructed based on real-world entity names in fictional parallel world events. We are systematically reviewing content involving real individuals and organizations to ensure no statements that may cause ambiguity or controversy. It will be released through controlled channels after review completion.

The dataset is planned for release as a **Gated Dataset** on HuggingFace.

---

**Academic Research Statement**

This project and the MPW Benchmark dataset are for academic research use only. All events in the dataset are fictional, set in a parallel world isolated from reality, and do not represent facts that actually occurred. They do not constitute statements, evaluations, or predictions about any real individuals, organizations, or institutions. The authors are not responsible for the consequences of use outside academic research purposes.
