# Mind-ParaWorld (MPW)

**面向 Tool-Integrated-Reasoning Agent 的轻量模块化异步推理框架，支持本地 vLLM 与商业 API 双后端，内置 Mind-ParaWorld (MPW) Benchmark 提供完整的 agentic search 评测环境。**

这个仓库解决的核心问题是：**如何在大规模、多轮、带工具调用的场景下，轻量高效地跑通一个 agent 的推理流程？** 框架提供 turn 级别的异步调度、可插拔的工具与 parser、统一的轨迹记录，推理结果以轨迹 + 预测答案的形式落盘，方便对接任意下游评测流程。MPW Benchmark 在此基础上提供了一套完整的 agentic search 评测环境，通过构造与训练数据隔离的 Parallel World 搜索场景，为搜索型 agent 提供端到端的能力度量基准。

> 论文：[Evaluating the Search Agent in a Parallel World](https://arxiv.org/abs/2603.04751)

---

## 项目概览 / Why this repo

**首先是一个轻量、模块化的 agent 推理框架。** 你可以带上自己的数据集、system prompt 和工具策略，直接完成大规模多轮 tool-use 推理，结果以轨迹 + 预测答案的形式落盘，可在 `evaluation/` 下构建自己的评测流程。框架支持：

- 任意 JSONL 格式数据集，一行一个样本
- 本地 vLLM 服务或任意 OpenAI 兼容 API
- 6 种内置 tool-call parser（Qwen系列、DeepSeek系列、GLM-4系列、MiniMax系列、Kimi-K2系列）
- 可插拔工具：内置 `web_search`、`FlashRAGSearchTool`、`PythonCodeInterpreterTool`，也可自定义并一键注册
- 并行工具调用（单轮多个 tool_call 同时执行）
- 断点续传（`--resume`）
- 内置 Claude Code Skills，可由 AI 编程助手交互式引导完成全套评测配置，无需手动阅读文档

**同时内置 MPW Benchmark 评测环境。** 如果你需要在受控的 Parallel World 场景中评测搜索型 agent，框架提供三种标准设置（Oracle / Guided / Unguided）的完整推理与评测脚本，以及配套的 FCR、Hit Rate 等指标计算工具。

如果你关心以下任一场景，这个仓库都可以直接作为起点：

- 在自己的数据集上对任意多轮 tool-use agent 完成大规模异步推理
- 对比不同 system prompt 策略或模型系列的 agent 能力
- 在 MPW Parallel World 环境下对比 guided / unguided / oracle 三种设置
- 基于已有结果文件做离线评测、主表生成与误差分析

---

## 用 AI 助手完成推理/评测配置

框架内置了两个 [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills-tutorial)，可以由 Claude Code 或兼容的 AI 助手（如 OpenClaw）交互式地引导你完成评测配置，**无需亲自阅读文档**。把这个仓库克隆下来之后，直接在 Claude Code 中运行：

| Skill | 适用场景 |
|---|---|
| `/mpw-custom` | 在任意数据集上配置并运行自定义 agent 推理 |
| `/mpw-bench` | 复现论文结果，或运行 MPW Setting A / B / C 评测 |

助手会逐步询问数据格式、模型部署方式、parser 选择等信息，最终生成一条可直接运行的推理命令。整个流程不需要你提前了解框架结构，**让 agent 助手帮你完成 agent 评测。**

---

## 核心能力

### 1) Turn-level async scheduling

推理入口采用 **turn 级别的异步调度**，而非样本级串行 rollout。全局任务队列动态分配并发 turn，某个样本在等待工具或模型响应时，其它样本的后续 turn 可以立即推进，整体吞吐相比串行方案提升显著。该机制在 `experiments/custom/vllm/async_inference.py` 中有完整实现。

### 2) Agent / Tool / Environment 解耦

`verl/` 下的核心组件：

- **`verl/sample_state.py`**：`SampleState` 管理单样本状态；`StateManager` 负责全局进度追踪与结果导出。
- **`verl/workers/agent/tool_envs.py`**：`ToolBase` 定义工具抽象，`ToolMeta` 自动注册子类。
- **`verl/workers/agent/envs/agent_eval.py`**：`AgentEval` 负责解析工具调用、执行工具、收集轨迹日志，支持可插拔工具列表与多种 parser。
- **`verl/workers/agent/envs/tools/world_model_web_search_tool.py`**：`WorldModelWebSearchTool` 通过 world model 模拟搜索引擎，支持 endpoint 池、缓存与异步重试。

### 3) 完整实验闭环

推理 → LLM-as-Judge 评测 → 主表生成 → 定量分析，各阶段均有独立脚本，结果文件格式统一，可按需接入任意环节。

---

## 系统组成 / 仓库结构

```text
Mind-ParaWorld/
├── data/                         # 数据示例与格式说明
├── experiments/                  # 推理入口
│   ├── setting_A_oracle/         # MPW Oracle：直接提供事实，单轮
│   ├── setting_B_guided/         # MPW Guided：多轮 + 查询分解引导
│   ├── setting_C_unguided/       # MPW Unguided：多轮标准 ReAct
│   └── custom/                   # 自定义数据集推理入口
├── evaluation/                   # 评测流程（用户自定义或接入 MPW judge）
├── verl/                         # agent-loop 核心组件
│   ├── sample_state.py
│   └── workers/agent/
│       ├── tool_envs.py
│       └── envs/
│           ├── agent_eval.py
│           └── tools/world_model_web_search_tool.py
└── analysis/                     # 定量分析、失败归因、主表与可视化
```

如果从工程视角快速理解这个仓库，推荐优先阅读：

- `experiments/custom/vllm/async_inference.py`
- `verl/sample_state.py`
- `verl/workers/agent/envs/agent_eval.py`
- `verl/workers/agent/envs/tools/world_model_web_search_tool.py`

---

## 快速开始

### 1) 基础环境

```bash
pip install openai httpx pillow tqdm
export PYTHONPATH=$(pwd)
```

### 2) 数据

框架接受 JSONL 格式，每行一个样本：

```json
{"index": 0, "prompt": [{"role": "user", "content": "你的问题"}], "answer": "参考答案", "extra_info": {}}
```

`data/mpw_bench.jsonl` 包含三条示例，可直接用于格式参考或冒烟测试。详见 [`data/README_CN.md`](data/README_CN.md)。

---

## 常见工作流

完整参数说明见 [`experiments/README_CN.md`](experiments/README_CN.md)。

### 自定义 agent 评测（通用入口）

对开源社区用户而言，最直接的起点是带上自己的数据，直接使用 `custom/` 下的推理脚本，无需任何 MPW 相关依赖：

```bash
# vLLM 后端
python experiments/custom/vllm/async_inference.py \
  --input data/your_data.jsonl \
  --output results/my_model/inference_results.jsonl \
  --model <model_name> \
  --api-base http://<vllm_host>:8000/v1 \
  --parser default \
  --max-concurrent-turns 64 \
  --max-turns-per-sample 16
```

使用商业 API 时切换到 `experiments/custom/api/async_inference.py`，参数相同。评测结果由 `experiments/custom/evaluate/llm-as-judge.py` 处理。

### 运行 MPW Setting C（Unguided，MPW-bench 专用）

以下流程**仅适用于 MPW Benchmark 评测**，需要额外部署 World Model 以提供模拟搜索环境：

```bash
# 部署世界模型后设置 endpoint
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

### 生成主结果表

```bash
python analysis/utils/generate_main_table.py \
  --inputs results/model1/evaluated_results.jsonl results/model2/evaluated_results.jsonl \
  --labels model1 model2 \
  --output main_table.md
```

---

## MPW Benchmark 介绍

MPW 的核心设定是：把 agent 放进一个发生于未来的 **Parallel World** 搜索环境中评测，构建模型认知隔离的虚拟搜索环境，而不是直接依赖开放真实 Web。

基本思路是：

1. 为每个问题维护一组 agent-unaware 的 atomic facts
2. agent 通过 `web_search` 查询逐步检索这些 facts
3. 最终依据 facts 覆盖情况与 Judge 评估结果判定回答质量

benchmark 共 **19 个类别、1608 个问题**，数据集计划通过 HuggingFace Gated Dataset 发布（详见 License 说明）。每个样本的 `extra_info.world_truth_info.atomic_facts` 提供 Parallel World 的原子事实集合。

### 三种评测设置

| Setting | 作用 | 特征 |
|---|---|---|
| **A — Oracle** | 评估信息充分条件下的整合与推理能力 | 直接提供 atomic facts，单轮回答 |
| **B — Guided** | 评估"提示引导的查询分解"对 agent 的帮助 | 多轮 ReAct + guided prompt |
| **C — Unguided** | 评估端到端 agent 能力 | 多轮 ReAct + 标准 prompt |

---

## 评测指标

| 指标 | 含义 |
|---|---|
| **Pass@1** | 最终答案被 Judge 判为正确的比例 |
| **FCR** | Fact Coverage Rate，命中的 atomic facts 占比 |
| **Hit Rate / Hit Precision** | 工具调用的命中效率 |
| **Avg Turns** | 每个样本平均对话轮数 |

主表统计 `Pass@1 | FCR | Hit Rate | Avg Turns`，聚合逻辑见 `analysis/utils/generate_main_table.py`。

---

## 结果分析

`analysis/` 覆盖从结果表到失败归因、定量统计与可视化的完整后续分析，详见 [`analysis/README_CN.md`](analysis/README_CN.md)。

```
analysis/
├── core/      # 综合分析、错误分析、多模型比较
├── tools/     # 失败归因、定量分析、综合报告、可视化
├── utils/     # 指标计算、难度划分、主表生成、Excel 转换
└── scripts/   # 批处理脚本与专题绘图脚本
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

**代码框架**采用 [Apache License 2.0](LICENSE) 开源协议。

**MPW Benchmark 数据集**暂不随代码仓库一同公开发布，原因如下：

**内容审查**：数据集基于现实世界实体名称构建虚构的平行世界事件，我们正在对涉及真实个人与组织的内容进行系统性审查，以确保不存在可能引发歧义或争议的表述，审查完成后将通过受控渠道发布。

数据集计划以 **Gated Dataset** 形式发布于 HuggingFace。

---

**学术研究声明**

本项目及 MPW Benchmark 数据集仅供学术研究使用。数据集中涉及的所有事件均为虚构，设定于与现实隔离的平行世界，不代表真实发生的事实，不构成对任何真实个人、组织或机构的陈述、评价或预测。作者不对数据集在非学术用途下的使用及其后果承担责任。
