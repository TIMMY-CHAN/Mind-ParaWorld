# Custom Prompt Evaluation

Custom prompt evaluation scenario that allows users to use their own System Prompt for agent evaluation.

## Directory Structure

```
custom/
├── vllm/                           # vLLM version inference
│   ├── async_inference.py          # Async inference script
│   └── run_custom_inference.sh     # One-click run script
├── api/                            # API version inference
│   ├── async_inference.py          # Async inference script
│   └── run_custom_inference.sh     # One-click run script
├── evaluate/                       # Evaluation module
│   ├── llm-as-judge.py             # LLM-as-Judge evaluation
│   └── run_judge_evaluation.sh     # One-click evaluation script
├── prompts/                        # Prompt file directory
│   ├── example_prompt.py           # Single tool call example prompt
│   └── parallel_search_prompt.py   # Parallel tool call example prompt (use with --parallel-tool-calls)
└── README.md                       # This document
```

## Quick Start

### 1. Prepare Custom Prompt

Create a Python file defining the `system_prompt` variable:

```python
# my_prompt.py

system_prompt = """You are a ReAct paradigm agent that can accept text input and answer user questions.
For complex questions, you can choose to call tools to help solve them.

## Available Tools

web_search: Search external information
<|action_start|>
{"name": "web_search", "arguments": {"query": "your query"}}
<|action_end|>

... (Add other rules and format requirements as needed)
"""
```

Place the file in the `prompts/` directory or use an absolute path.

### 2. Run Inference

#### vLLM Version (Recommended)

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/vllm/run_custom_inference.sh
```

Follow prompts to enter:
- Prompt file path
- Model name

#### API Version

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/api/run_custom_inference.sh
```

Follow prompts to enter:
- Prompt file path
- API Provider (openai/azure/custom)
- Model name
- API Base URL

### 3. Run Evaluation

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/evaluate/run_judge_evaluation.sh
```

Follow prompts to select the Prompt and model to evaluate.

## Command Line Arguments

### vLLM Version

```bash
python experiments/custom/vllm/async_inference.py \
    --input data.jsonl \              # Input file
    --output results.jsonl \          # Output file
    --prompt-file my_prompt.py \      # Prompt file
    --model models \                  # Model name
    --api-base http://localhost:8000/v1 \  # API address
    --max-concurrent-turns 32 \       # Max concurrent turns
    --max-turns-per-sample 8 \        # Max turns per sample
    --enable-thinking \               # Enable thinking mode (optional)
    --parser default \                # Tool call parser (optional, see below)
    --parallel-tool-calls             # Enable parallel tool calls (optional, see below)
```

### API Version

API version supports two tool call modes, controlled by `--tool-mode` parameter:

**Text Mode (default, aligned with vLLM)**

```bash
python experiments/custom/api/async_inference.py \
    --provider openai \               # API provider (openai/azure/custom)
    --model gpt-4o \                  # Model name
    --input data.jsonl \              # Input file
    --output results.jsonl \          # Output file
    --prompt-file my_prompt.py \      # Prompt file
    --tool-mode text \                # Text parsing mode (default)
    --parser default \                # Tool call parser (same as vllm version)
    --parallel-tool-calls \           # Enable parallel tool calls (optional, see below)
    --qps 10 \                        # QPS limit
    --qpm 600 \                       # QPM limit (optional)
    --max-concurrent-turns 100 \      # Max concurrent turns
    --max-turns-per-sample 32 \       # Max turns per sample
    --resume                          # Resume from interruption
```

**Native Mode (recommended for commercial APIs)**

```bash
python experiments/custom/api/async_inference.py \
    --provider openai \
    --model gpt-4o \
    --input data.jsonl \
    --output results.jsonl \
    --prompt-file my_prompt.py \
    --tool-mode native \              # Native API function calling
    --qps 10 \
    --max-concurrent-turns 100 \
    --max-turns-per-sample 32 \
    --resume
```

| Mode | `--tool-mode` | Description |
|------|--------------|------|
| Text mode | `text` (default) | System prompt defines format, parser extracts tool calls from text; aligned with vLLM version, suitable for cross-deployment comparison evaluation |
| Native mode | `native` | Pass `tools` parameter, read results from `tool_calls` field; parsed by API provider, no `<answer>` tag needed, suitable for evaluating commercial model true capability |

> **Note**: `--parser` parameter only works in `--tool-mode text`, ignored in native mode.

## Input Data Format

Input JSONL file format:

```json
{
  "index": 0,
  "prompt": [
    {"role": "user", "content": "Your question"}
  ],
  "answer": "Reference answer",
  "extra_info": {}  // Optional metadata, pass empty object if not needed
}
```

## Output Data Format

Output JSONL file format:

```json
{
  "index": 0,
  "messages": [...],
  "full_messages": [...],
  "final_answer": "Model predicted answer",
  "ground_truth": "Reference answer",
  "status": "finished",
  "trajectory_log": {...},
  "judge_judgment": "Correct",
  "judge_reason": "Judgment reasoning"
}
```

## Differences from Other Settings

| Feature | Setting A | Setting B | Setting C | Custom |
|------|-----------|-----------|-----------|--------|
| Tool calling | ❌ | ✅ | ✅ | ✅ |
| System Prompt | Fixed | Predefined options | None | Fully customizable |
| Use case | Oracle test | Guided Search | Unguided | Custom experiments |

## Prompt Writing Guide

### Required Elements

1. **Tool Definition**: Describe available tools and their usage
2. **Output Format**: Define thinking and answer output format
3. **Behavior Guidance**: Describe when to use tools, how to answer questions

### Example Template

```python
system_prompt = """You are a ReAct paradigm agent...

## Available Tools

web_search: ...

## Output Format

<|thought_start|>
Your thinking process
<|thought_end|>

<|action_start|>
{"name": "web_search", "arguments": {"query": "..."}}
<|action_end|>

<answer>
Your final answer
</answer>

## Notes

1. Wait for results after each tool call
2. End conversation after providing answer
3. Do not refuse to answer questions
"""
```

### Parallel Tool Calls

With the `--parallel-tool-calls` flag, the framework can identify multiple tool call blocks in a single turn from model output, execute them in parallel, and merge them into a single user message, reducing dialogue turns. All built-in parsers support this feature.

**Prerequisites**

| Condition | Requirement |
|---|---|
| Parser | All built-in parsers support multiple tool_call extraction (see parser table below); `kimi_k2` falls back to `web_search` for all parallel calls since function name is not in text |
| System Prompt | Must explicitly tell the model it can output multiple tool calls in the same turn, format examples must match selected parser |

**Message Format**

When enabled, N parallel calls in a single turn produce the following messages structure:

```
[assistant]   <\|action_start|>{"query": "A"}<\|action_end|>
              <\|action_start|>{"query": "B"}<\|action_end|>

[user]       Search result: A → ...

             Search result: B → ...
```

N results are merged into a single user message separated by `\n\n`; if only 1 `<\|action_start|>` is detected, it automatically falls back to normal call behavior, identical to without the flag.

**Quick Start**

The repository provides accompanying Prompt examples:

```bash
# vLLM
python experiments/custom/vllm/async_inference.py \
    --prompt-file experiments/custom/prompts/parallel_search_prompt.py \
    --parser default \
    --parallel-tool-calls \
    --model <model> \
    --api-base http://localhost:8000/v1 \
    --input data/your_data.jsonl \
    --output results/parallel/inference_results.jsonl

# API (text mode)
python experiments/custom/api/async_inference.py \
    --prompt-file experiments/custom/prompts/parallel_search_prompt.py \
    --provider custom --base-url http://localhost:8000/v1 \
    --model <model> \
    --tool-mode text --parser default \
    --parallel-tool-calls \
    --input data/your_data.jsonl \
    --output results/parallel/inference_results.jsonl
```

**Prompt Writing Points**

1. **Format Declaration**: Clearly declare the tool call format corresponding to the selected parser (e.g., `default` parser uses `<\|action_start|>{"name": "...", "arguments": {...}}<\|action_end|>`)
2. **Parallel Description**: Tell the model it can output multiple tool call blocks in the same turn, indicate when to parallelize (independent queries)
3. **Serial Description**: Tell the model to issue in separate turns when subsequent queries depend on previous results
4. **Format Example**: Provide a complete output example containing multiple tool call blocks

See `prompts/parallel_search_prompt.py` (`default` parser example).

## Custom Tool Registration

The `custom` scenario supports registering any tool beyond `web_search`. The framework by default only loads `web_search` (`WorldModelWebSearchTool`), which can be replaced or extended in two ways.

### Implement a Custom Tool

Inherit from `ToolBase`, define the `name` class attribute, and implement `execute_async()` (recommended) or `execute()`:

```python
from verl.workers.agent.tool_envs import ToolBase

class MyCustomTool(ToolBase):
    name = "my_tool"

    def __init__(self, *args, **kwargs):
        super().__init__(
            name=self.name,
            description="Describe your tool functionality",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Input parameter"}
                },
                "required": ["query"]
            }
        )

    async def execute_async(self, action_string: str, **kwargs) -> tuple:
        # kwargs will automatically include:
        #   agent_trajectory: str  Current complete trajectory
        #   world_truth: dict      Current sample's world truth
        # Tool uses as needed, ignore unneeded parameters

        args = ...  # Parse parameters from action_string
        result = ...  # Tool execution logic
        obs = f"\n<|im_start|>user\n{result}\n<|im_end|>\n<|im_start|>assistant\n"
        return obs, 0.0, False, {}

    def reset(self, **kwargs):
        pass  # Reset tool internal state before each sample

    def execute(self, *args, **kwargs):
        pass  # Only for synchronous scenarios, implement execute_async for async scenarios
```

### Parameter Validation (Optional)

`ToolBase` provides a `validate_args(args: dict)` helper method that validates parameters against the `parameters` JSON Schema defined in `__init__`. **This method is called by the tool author within their own `execute_async` / `execute`, the framework does not trigger it automatically.**

```python
async def execute_async(self, action_string: str, **kwargs) -> tuple:
    args = ...  # Parse parameters
    valid, msg = self.validate_args(args)
    if not valid:
        obs = f"\n<|im_start|>user\nError: {msg}\n<|im_end|>\n<|im_start|>assistant\n"
        return obs, 0.0, False, {"error": msg}
    # Parameters valid, continue execution
    ...
```

### Register Tool

**Method 1: Pass at construction** (recommended, one-time replacement of entire tool set)

```python
from verl.workers.agent.envs.agent_eval import AgentEval
from verl.workers.agent.envs.tools.world_model_web_search_tool import WorldModelWebSearchTool

agent = AgentEval(tools=[
    WorldModelWebSearchTool(),      # Keep default web_search
    MyCustomTool(),       # Add custom tool
])
```

**Method 2: Register dynamically after construction** (suitable for conditionally adding tools)

```python
agent = AgentEval()
agent.register_tool(MyCustomTool())
```

After registration, the tool name automatically appears in the dispatch routing, and the agent can trigger calls using the corresponding `name` in `<\|action_start|>`.

### Switch Tool Call Parser

The framework has six built-in parsers corresponding to different model text output formats:

| Name | Applicable Models | Tool Call Format | Parallel Calls |
|---|---|---|---|
| `default` | Hermes / NousResearch / Qwen3 / This framework default | `<\|action_start|>{"name":..., "arguments":{...}}<\|action_end|>` | ✅ |
| `deepseek` | DeepSeek-V2 / V3 / R1 | `<\|tool▁calls▁begin\|>...<\|tool▁sep\|>name\n```json\n{...}\n```...` | ✅ |
| `glm4` | GLM-4 / GLM-Z1 | `<\|action_start|>name<\|action_sep|>k=v<\|action_sep|>...<\|action_end|>` | ✅ |
| `minimax` | MiniMax-M2.5 | `<minimax:tool_call><invoke name="..."><parameter name="k">v</parameter></invoke></minimax:tool_call>` | ✅ |
| `qwen35` | Qwen3.5 | `<\|action_start|>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n<\|action_end|>` | ✅ |
| `kimi_k2` | Kimi-K2 | `<\|tool_call_begin\|>call_id<\|tool_call_argument_begin\|>{...}<\|tool_call_end|>` (⚠️ function name not in text, defaults to `web_search`, see `parsers.py` doc for multi-tool scenarios)| ✅ ⚠️ |

**Important**: Parser and System Prompt are bound. When switching parsers, the tool call format examples in System Prompt must match, otherwise the model will output in the wrong format.

Command line usage:

```bash
python experiments/custom/vllm/async_inference.py \
    --prompt-file prompts/my_glm_prompt.py \
    --parser glm4 \
    ...
```

Code usage:

```python
agent = AgentEval(parser="glm4")
# Or pass instance directly
from verl.workers.agent.parsers import GLM4Parser
agent = AgentEval(parser=GLM4Parser())
```

To support other model formats, inherit from `ToolCallParser` and register to `PARSER_REGISTRY`:

```python
from verl.workers.agent.parsers import ToolCallParser, PARSER_REGISTRY

class MyModelParser(ToolCallParser):
    name = "my_model"

    def extract_action(self, text):
        # Parse and normalize to {"name": "...", "arguments": {...}}
        ...

    def extract_answer(self, text):
        ...

PARSER_REGISTRY["my_model"] = MyModelParser
```

Then use via `--parser my_model` or `AgentEval(parser="my_model")`.

---

## Common Questions

### Q: How to debug Prompt?

A: Test with a small dataset first, observe model behavior:

```bash
# Extract first 10 samples
head -n 10 data/your_data.jsonl > data/test.jsonl

# Run inference
python experiments/custom/vllm/async_inference.py \
    --input data/test.jsonl \
    --output results/test.jsonl \
    --prompt-file prompts/my_prompt.py \
    --model models \
    --api-base http://localhost:8000/v1
```

### Q: How to compare different Prompt effects?

A: Run inference with different Prompt files, results will be saved in different directories:

```
custom_results/
├── prompt_v1/
│   └── model_name/
│       └── inference_results.jsonl
├── prompt_v2/
│   └── model_name/
│       └── inference_results.jsonl
```

### Q: How to use existing Prompts?

A: You can directly use Prompts from `setting_B_guided`:

```python
# prompts/fewshot_prompt.py
from experiments.setting_B_guided.prompt import fewshot_prompt as system_prompt
```

Or copy the content to a new file.

## Best Practices

1. **Naming Convention**: Name Prompt files clearly, e.g., `web_search_optimized.py`
2. **Version Control**: Record the reason and effect of each Prompt modification
3. **Comparison Experiments**: Keep other variables constant, only change Prompt
4. **Resume from Interruption**: If inference is interrupted, re-run will automatically skip processed samples
