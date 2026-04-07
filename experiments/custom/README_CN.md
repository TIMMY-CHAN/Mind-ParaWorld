# Custom Prompt Evaluation

自定义 Prompt 评估场景，允许用户使用自己的 System Prompt 进行 Agent 评估。

## 目录结构

```
custom/
├── vllm/                           # vLLM 版本推理
│   ├── async_inference.py          # 异步推理脚本
│   └── run_custom_inference.sh     # 一键运行脚本
├── api/                            # API 版本推理
│   ├── async_inference.py          # 异步推理脚本
│   └── run_custom_inference.sh     # 一键运行脚本
├── evaluate/                       # 评估模块
│   ├── llm-as-judge.py             # LLM-as-Judge 评估
│   └── run_judge_evaluation.sh     # 一键评估脚本
├── prompts/                        # Prompt 文件目录
│   ├── example_prompt.py           # 单工具调用示例 Prompt
│   └── parallel_search_prompt.py   # 并行工具调用示例 Prompt（配合 --parallel-tool-calls）
└── README.md                       # 本文档
```

## 快速开始

### 1. 准备自定义 Prompt

创建一个 Python 文件，定义 `system_prompt` 变量：

```python
# my_prompt.py

system_prompt = """你是一个 ReAct 范式的 agent，能够接受文本输入，回答用户问题。
对于复杂问题，你可以选择调用工具辅助解决。

## 可用工具

web_search: 搜索外部信息
<tool_call>{"name": "web_search", "arguments": {"query": "your query"}}</tool_call>

...（根据需要补充其他规则与格式要求）
"""
```

将文件放入 `prompts/` 目录或使用绝对路径。

### 2. 运行推理

#### vLLM 版本（推荐）

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/vllm/run_custom_inference.sh
```

按提示输入：
- Prompt 文件路径
- 模型名称

#### API 版本

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/api/run_custom_inference.sh
```

按提示输入：
- Prompt 文件路径
- API Provider（openai/azure/custom）
- 模型名称
- API Base URL

### 3. 运行评估

```bash
cd /path/to/Mind-ParaWorld
bash experiments/custom/evaluate/run_judge_evaluation.sh
```

按提示选择要评估的 Prompt 和模型。

## 命令行参数

### vLLM 版本

```bash
python experiments/custom/vllm/async_inference.py \
    --input data.jsonl \              # 输入文件
    --output results.jsonl \          # 输出文件
    --prompt-file my_prompt.py \      # Prompt 文件
    --model models \                  # 模型名称
    --api-base http://localhost:8000/v1 \  # API 地址
    --max-concurrent-turns 32 \       # 最大并发 Turn 数
    --max-turns-per-sample 8 \        # 每样本最大轮数
    --enable-thinking                 # 启用 thinking 模式（可选）
    --parser default                  # tool call 解析器（可选，见下文）
    --parallel-tool-calls             # 启用并行工具调用（可选，见下文）
```

### API 版本

API 版本支持两种工具调用模式，通过 `--tool-mode` 参数控制：

**文本模式（默认，与 vLLM 对齐）**

```bash
python experiments/custom/api/async_inference.py \
    --provider openai \               # API 提供商（openai/azure/custom）
    --model gpt-4o \                  # 模型名称
    --input data.jsonl \              # 输入文件
    --output results.jsonl \          # 输出文件
    --prompt-file my_prompt.py \      # Prompt 文件
    --tool-mode text \                # 文本解析模式（默认）
    --parser default \                # tool call 解析器（与 vllm 版本一致）
    --parallel-tool-calls \           # 启用并行工具调用（可选，见下文）
    --qps 10 \                        # QPS 限制
    --qpm 600 \                       # QPM 限制（可选）
    --max-concurrent-turns 100 \      # 最大并发 Turn 数
    --max-turns-per-sample 32 \       # 每样本最大轮次
    --resume                          # 断点续传
```

**原生模式（商业 API 推荐）**

```bash
python experiments/custom/api/async_inference.py \
    --provider openai \
    --model gpt-4o \
    --input data.jsonl \
    --output results.jsonl \
    --prompt-file my_prompt.py \
    --tool-mode native \              # API 原生 function calling
    --qps 10 \
    --max-concurrent-turns 100 \
    --max-turns-per-sample 32 \
    --resume
```

| 模式 | `--tool-mode` | 说明 |
|------|--------------|------|
| 文本模式 | `text`（默认）| System Prompt 约定格式，parser 从文本中提取工具调用；与 vLLM 版本保持一致，适合跨部署对比评估 |
| 原生模式 | `native` | 传入 `tools` 参数，从 `tool_calls` 字段读取结果；由 API provider 完成解析，无需 `<answer>` 标签，适合评估商业模型真实能力 |

> **注意**：`--parser` 参数仅在 `--tool-mode text` 时生效，原生模式下忽略此参数。

## 输入数据格式

输入 JSONL 文件格式：

```json
{
  "index": 0,
  "prompt": [
    {"role": "user", "content": "你的问题"}
  ],
  "answer": "标准答案",
  "extra_info": {}  // 可选元数据，不需要时传空对象
}
```

## 输出数据格式

输出 JSONL 文件格式：

```json
{
  "index": 0,
  "messages": [...],
  "full_messages": [...],
  "final_answer": "模型预测答案",
  "ground_truth": "标准答案",
  "status": "finished",
  "trajectory_log": {...},
  "judge_judgment": "Correct",
  "judge_reason": "判断理由"
}
```

## 与其他 Setting 的区别

| 特性 | Setting A | Setting B | Setting C | Custom |
|------|-----------|-----------|-----------|--------|
| 工具调用 | ❌ | ✅ | ✅ | ✅ |
| System Prompt | 固定 | 预定义选项 | 无 | 完全自定义 |
| 使用场景 | Oracle 测试 | Guided Search | Unguided | 自定义实验 |

## Prompt 编写指南

### 必须包含的元素

1. **工具定义**：说明可用的工具及其用法
2. **输出格式**：定义思考和答案的输出格式
3. **行为指导**：说明何时使用工具、如何回答问题

### 示例模板

```python
system_prompt = """你是一个ReAct范式的agent...

## 可用工具

web_search: ...

## 输出格式

<|thought_start|>
你的思考过程
<|thought_end|>

<|action_start|>
{"name": "web_search", "arguments": {"query": "..."}}
<|action_end|>

<answer>
你的最终答案
</answer>

## 注意事项

1. 每次调用工具后等待结果返回
2. 给出答案后结束对话
3. 不要拒绝回答问题
"""
```

### 并行工具调用

通过 `--parallel-tool-calls` 标志，框架可以识别模型在单轮中输出的多个工具调用块，将它们并行执行后合并为一条用户消息返回，减少对话轮数。所有内置 parser 均已支持此功能。

**前提条件**

| 条件 | 要求 |
|---|---|
| 解析器 | 所有内置 parser 均支持多 tool_call 提取（见下方 parser 表格）；`kimi_k2` 因函数名不在文本中，所有并行调用以 `web_search` 兜底 |
| System Prompt | 必须明确告知模型可以在同一轮输出多个工具调用，格式示例须与所选 parser 匹配 |

**消息格式**

启用后，单轮 N 个并行调用产生的 messages 结构如下：

```
[assistant]  <tool_call>{"query": "A"}</tool_call>
             <tool_call>{"query": "B"}</tool_call>

[user]       搜索结果：A → ...

             搜索结果：B → ...
```

N 个结果用 `\n\n` 分隔合并为单条 user 消息；若只检测到 1 个 `<tool_call>`，自动退化为普通调用，行为与不加 flag 完全一致。

**快速开始**

仓库已提供配套 Prompt 示例：

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

# API（文本模式）
python experiments/custom/api/async_inference.py \
    --prompt-file experiments/custom/prompts/parallel_search_prompt.py \
    --provider custom --base-url http://localhost:8000/v1 \
    --model <model> \
    --tool-mode text --parser default \
    --parallel-tool-calls \
    --input data/your_data.jsonl \
    --output results/parallel/inference_results.jsonl
```

**Prompt 编写要点**

1. **格式声明**：明确声明所选 parser 对应的工具调用格式（如 `default` parser 使用 `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`）
2. **并行说明**：告知模型可在同一轮输出多个工具调用块，示意什么情况下并行（独立查询）
3. **串行说明**：告知模型当后续查询依赖前一步结果时，分轮发出
4. **格式示例**：给出一个包含多个工具调用块的完整输出示例

详见 `prompts/parallel_search_prompt.py`（`default` parser 示例）。

## 自定义工具注册

`custom` 场景支持在 `web_search` 之外注册任意工具。框架默认只加载 `web_search`（`WorldModelWebSearchTool`），可以通过以下两种方式替换或扩展。

### 实现一个自定义工具

继承 `ToolBase`，定义 `name` 类属性，并实现 `execute_async()`（推荐）或 `execute()`：

```python
from verl.workers.agent.tool_envs import ToolBase

class MyCustomTool(ToolBase):
    name = "my_tool"

    def __init__(self, *args, **kwargs):
        super().__init__(
            name=self.name,
            description="描述你的工具功能",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "输入参数"}
                },
                "required": ["query"]
            }
        )

    async def execute_async(self, action_string: str, **kwargs) -> tuple:
        # kwargs 中会自动传入：
        #   agent_trajectory: str  当前完整轨迹
        #   world_truth: dict      当前样本的世界真相
        # 工具按需取用，不需要的参数忽略即可

        args = ...  # 从 action_string 里解析参数
        result = ...  # 工具执行逻辑
        obs = f"\n<|im_start|>user\n{result}\n<|im_end|>\n<|im_start|>assistant\n"
        return obs, 0.0, False, {}

    def reset(self, **kwargs):
        pass  # 每个样本开始前重置工具内部状态

    def execute(self, *args, **kwargs):
        pass  # 仅供同步场景使用，async 场景实现 execute_async 即可
```

### 参数校验（可选）

`ToolBase` 提供了 `validate_args(args: dict)` 辅助方法，根据 `__init__` 中定义的 `parameters` JSON Schema 校验参数是否合法。**这个方法由工具作者在自己的 `execute_async` / `execute` 内主动调用，框架不会自动触发。**

```python
async def execute_async(self, action_string: str, **kwargs) -> tuple:
    args = ...  # 解析参数
    valid, msg = self.validate_args(args)
    if not valid:
        obs = f"\n<|im_start|>user\nError: {msg}\n<|im_end|>\n<|im_start|>assistant\n"
        return obs, 0.0, False, {"error": msg}
    # 参数合法，继续执行
    ...
```

### 注册工具

**方式一：构造时传入**（推荐，一次性替换整个工具集）

```python
from verl.workers.agent.envs.agent_eval import AgentEval
from verl.workers.agent.envs.tools.world_model_web_search_tool import WorldModelWebSearchTool

agent = AgentEval(tools=[
    WorldModelWebSearchTool(),      # 保留默认的 web_search
    MyCustomTool(),       # 新增自定义工具
])
```

**方式二：构造后动态注册**（适合按条件添加工具）

```python
agent = AgentEval()
agent.register_tool(MyCustomTool())
```

注册后工具名称会自动出现在分发路由里，agent 在 `<tool_call>` 中使用对应的 `name` 即可触发调用。

### 切换 tool call 解析器

框架内置六种解析器，对应不同模型的文本输出格式：

| 名称 | 适用模型 | tool call 格式 | 并行调用 |
|---|---|---|---|
| `default` | Hermes / NousResearch / Qwen3 / 本框架默认 | `<tool_call>{"name":..., "arguments":{...}}</tool_call>` | ✅ |
| `deepseek` | DeepSeek-V2 / V3 / R1 | `<\|tool▁calls▁begin\|>...<\|tool▁sep\|>name\n```json\n{...}\n```...` | ✅ |
| `glm4` | GLM-4 / GLM-Z1 | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>` | ✅ |
| `minimax` | MiniMax-M2.5 | `<minimax:tool_call><invoke name="..."><parameter name="k">v</parameter></invoke></minimax:tool_call>` | ✅ |
| `qwen35` | Qwen3.5 | `<tool_call>\n<function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>\n</tool_call>` | ✅ |
| `kimi_k2` | Kimi-K2 | `<\|tool_call_begin\|>call_id<\|tool_call_argument_begin\|>{...}<\|tool_call_end\|>`（⚠️ 函数名不在文本中，默认兜底为 `web_search`，多工具场景见 `parsers.py` 文档）| ✅ ⚠️ |

**重要**：Parser 与 System Prompt 是绑定关系。切换 parser 时，System Prompt 中的工具调用格式示例必须与之匹配，否则模型会按错误格式输出。

命令行使用：

```bash
python experiments/custom/vllm/async_inference.py \
    --prompt-file prompts/my_glm_prompt.py \
    --parser glm4 \
    ...
```

代码中使用：

```python
agent = AgentEval(parser="glm4")
# 或直接传实例
from verl.workers.agent.parsers import GLM4Parser
agent = AgentEval(parser=GLM4Parser())
```

如需支持其他模型格式，继承 `ToolCallParser` 后注册到 `PARSER_REGISTRY` 即可：

```python
from verl.workers.agent.parsers import ToolCallParser, PARSER_REGISTRY

class MyModelParser(ToolCallParser):
    name = "my_model"

    def extract_action(self, text):
        # 解析并归一化为 {"name": "...", "arguments": {...}}
        ...

    def extract_answer(self, text):
        ...

PARSER_REGISTRY["my_model"] = MyModelParser
```

然后通过 `--parser my_model` 或 `AgentEval(parser="my_model")` 使用。

---

## 常见问题

### Q: 如何调试 Prompt？

A: 先用小数据集测试，观察模型行为：

```bash
# 提取前10条数据
head -n 10 data/your_data.jsonl > data/test.jsonl

# 运行推理
python experiments/custom/vllm/async_inference.py \
    --input data/test.jsonl \
    --output results/test.jsonl \
    --prompt-file prompts/my_prompt.py \
    --model models \
    --api-base http://localhost:8000/v1
```

### Q: 如何对比不同 Prompt 的效果？

A: 使用不同的 Prompt 文件运行推理，结果会保存在不同目录：

```
custom_results/
├── prompt_v1/
│   └── model_name/
│       └── inference_results.jsonl
├── prompt_v2/
│   └── model_name/
│       └── inference_results.jsonl
```

### Q: 如何使用已有的 Prompt？

A: 可以直接使用 `setting_B_guided` 中的 Prompt：

```python
# prompts/fewshot_prompt.py
from experiments.setting_B_guided.prompt import fewshot_prompt as system_prompt
```

或者复制内容到新文件中。

## 最佳实践

1. **命名规范**：Prompt 文件命名要清晰，如 `web_search_optimized.py`
2. **版本管理**：记录每次 Prompt 修改的原因和效果
3. **对比实验**：保持其他变量不变，只改变 Prompt
4. **断点续传**：推理被中断时可重新运行，会自动跳过已处理样本

