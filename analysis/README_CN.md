# Analysis 目录说明

本目录提供 MPW / agentic search 实验结果的离线分析工具，分为四层：

1. **core/**：基础统计与错误分析
2. **utils/**：可复用的指标计算、主表生成与格式转换工具
3. **research/**：论文专题分析与可视化脚本
4. **scripts/**：批量执行包装脚本

---

## 目录结构

```text
analysis/
├── core/
│   ├── comprehensive_analysis.py
│   ├── error_analysis.py
│   └── model_comparison.py
├── research/
│   ├── fcr_vs_toolcalls_analysis.py
│   ├── plot_fcr_vs_pass1_relation.py
│   └── scripts/
│       ├── plot_fcr_vs_toolcalls.py
│       ├── plot_fcr_vs_toolcalls_cohort_and_examples.py
│       ├── plot_fcr_vs_toolcalls_truncated.py
│       ├── report_nk_for_fcr_toolcalls.py
│       └── run_fcr_vs_toolcalls_plots.sh
├── tools/
│   └── quantitative_analysis_settingB_C.py
├── utils/
│   ├── calculate_metrics.py
│   ├── convert_to_excel.py
│   ├── difficulty_utils.py
│   ├── generate_main_table.py
│   └── metrics_calculator.py
└── scripts/
    ├── batch_calculate_difficulty_pass1.sh
    ├── batch_quantitative_analysis_settingB.sh
    ├── batch_quantitative_analysis_settingB_fewshot.sh
    ├── batch_quantitative_analysis_settingB_guidance.sh
    ├── calculate_difficulty_pass1.py
    └── generate_settingC_summary.py
```

---

## 一、`core/`：基础分析层

### `core/comprehensive_analysis.py`

对单个结果文件做综合统计，输出整体成功率、按类别表现、工具调用分布、FCR 等基础指标。适用于快速了解单个结果文件的整体情况或初步 sanity check。

### `core/error_analysis.py`

分析推理轨迹中的错误现象，关注工具调用格式错误、world model 响应解析失败、错误恢复能力等。适用于判断失败来源（检索、格式、环境响应、执行链路）。

### `core/model_comparison.py`

汇总多个模型的评测统计结果，输出横向对比报告。依赖预先生成的统计文件，而非直接处理原始轨迹。

---

## 二、`tools/`：定量分析主入口

### `tools/quantitative_analysis_settingB_C.py`

Setting B / Setting C 共用的定量分析脚本，直接从 `trajectory_log` 中提取指标，适配以 `trajectory_log`、`answer`、`actual_turns` 为主的结果字段组织方式。

---

## 三、`utils/`：可复用工具层

### `utils/difficulty_utils.py`

统一难度分级逻辑，根据原子事实数量划分 Easy / Medium / Hard，兼容多种样本结构。

### `utils/metrics_calculator.py`

统一指标计算模块，提供整体统计、类别统计、难度统计、工具调用统计等基础接口，适合被其他分析脚本复用。

### `utils/calculate_metrics.py`

从结果文件离线重算指标，可用于补齐 `metrics` 信息或统一指标口径。

### `utils/generate_main_table.py`

按难度分级生成主表，核心指标包括 `Pass@1 / FCR / Hit Rate / Avg Turns`，输出 Markdown 或 CSV。

```bash
python analysis/utils/generate_main_table.py \
  --inputs results/model1/evaluated_results.jsonl results/model2/evaluated_results.jsonl \
  --labels model1 model2 \
  --output main_table.md
```

### `utils/convert_to_excel.py`

将 Markdown 主表转换为 Excel 文件：

```bash
python analysis/utils/convert_to_excel.py main_table.md main_table.xlsx
```

---

## 四、`research/`：研究专题分析层

承接论文分析和专题可视化内容，通常不是主入口，但对于复现特定图表仍有参考价值。

### `research/fcr_vs_toolcalls_analysis.py`

面向 Setting C，统计工具调用次数 vs FCR 的关系，生成后续绘图所需的中间 JSON 数据。

### `research/plot_fcr_vs_pass1_relation.py`

绘制单个模型在 Setting B + Setting C 中的样本级 FCR → Pass@1 关系图，通常将 Setting A 作为上界参考。

### `research/scripts/`

一组专题绘图脚本：FCR vs tool calls 曲线、截断曲线、cohort 固定样本集分析、`n(k)` 表输出等。`run_fcr_vs_toolcalls_plots.sh` 将上述脚本串成完整的专题分析链。

---

## 五、根目录 `scripts/`：批量执行与轻量统计

### 批量执行类

- `batch_quantitative_analysis_settingB.sh`
- `batch_quantitative_analysis_settingB_fewshot.sh`
- `batch_quantitative_analysis_settingB_guidance.sh`
- `batch_calculate_difficulty_pass1.sh`

面向现有结果目录做批量执行，一次性处理多个模型或多个 prompt 子目录。均通过 `SCRIPT_DIR` / `PROJECT_ROOT` 推导仓库根目录，不依赖本地绝对路径。

### `scripts/calculate_difficulty_pass1.py`

按难度分级统计 Pass@1，支持 Setting A / B / C 的多种结果结构。适合生成仅关注 difficulty-Pass@1 的辅助表。

### `scripts/generate_settingC_summary.py`

生成 Setting C 专题汇总表。与 `utils/generate_main_table.py` 的区别在于：前者在定量分析结果基础上整理专题汇总，后者直接从原始结果文件生成正式主表。

---

## 推荐使用路径

**主分析链**
```
tools/quantitative_analysis_settingB_C.py
  → utils/generate_main_table.py
  → utils/convert_to_excel.py
```

**基础支持**
```
core/  ·  utils/difficulty_utils.py  ·  utils/metrics_calculator.py
```

**论文专题**
```
research/  ·  scripts/generate_settingC_summary.py
```
