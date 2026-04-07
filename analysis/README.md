# Analysis Directory

This directory provides offline analysis tools for MPW / agentic search experiment results, organized into four layers:

1. **core/**: Basic statistics and error analysis
2. **utils/**: Reusable metric calculation, main table generation, and format conversion tools
3. **research/**: Paper-specific analysis and visualization scripts
4. **scripts/**: Batch execution wrapper scripts

---

## Directory Structure

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

## I. `core/`: Basic Analysis Layer

### `core/comprehensive_analysis.py`

Performs comprehensive statistics on a single result file, outputting overall success rate, performance by category, tool call distribution, FCR, and other basic metrics. Suitable for quickly understanding the overall situation of a single result file or preliminary sanity check.

### `core/error_analysis.py`

Analyzes error phenomena in reasoning trajectories, focusing on tool call format errors, world model response parsing failures, error recovery capabilities, etc. Suitable for determining failure sources (retrieval, format, environment response, execution chain).

### `core/model_comparison.py`

Aggregates evaluation statistics from multiple models, outputs horizontal comparison reports. Depends on pre-generated statistics files rather than directly processing raw trajectories.

---

## II. `tools/`: Quantitative Analysis Main Entry

### `tools/quantitative_analysis_settingB_C.py`

Shared quantitative analysis script for Setting B / Setting C, extracts metrics directly from `trajectory_log`, adapted to result field organization centered on `trajectory_log`, `answer`, `actual_turns`.

---

## III. `utils/`: Reusable Tools Layer

### `utils/difficulty_utils.py`

Unified difficulty stratification logic, divides Easy / Medium / Hard based on atomic fact count, compatible with various sample structures.

### `utils/metrics_calculator.py`

Unified metric calculation module, provides basic interfaces for overall statistics, category statistics, difficulty statistics, tool call statistics, etc., suitable for reuse by other analysis scripts.

### `utils/calculate_metrics.py`

Offline recalculation of metrics from result files, can be used to fill in missing `metrics` information or unify metric definitions.

### `utils/generate_main_table.py`

Generates main table by difficulty level, core metrics include `Pass@1 / FCR / Hit Rate / Avg Turns`, outputs Markdown or CSV.

```bash
python analysis/utils/generate_main_table.py \
  --inputs results/model1/evaluated_results.jsonl results/model2/evaluated_results.jsonl \
  --labels model1 model2 \
  --output main_table.md
```

### `utils/convert_to_excel.py`

Convert Markdown main table to Excel file:

```bash
python analysis/utils/convert_to_excel.py main_table.md main_table.xlsx
```

---

## IV. `research/`: Research Topic Analysis Layer

Contains paper analysis and topic visualization content, usually not the main entry point but still valuable for reproducing specific charts.

### `research/fcr_vs_toolcalls_analysis.py`

For Setting C, statistics on tool call count vs. FCR relationship, generates intermediate JSON data needed for subsequent plotting.

### `research/plot_fcr_vs_pass1_relation.py`

Plots single model's sample-level FCR → Pass@1 relationship in Setting B + Setting C, usually with Setting A as upper bound reference.

### `research/scripts/`

A set of topic plotting scripts: FCR vs tool calls curves, truncation curves, cohort fixed sample analysis, `n(k)` table output, etc. `run_fcr_vs_toolcalls_plots.sh` chains the above scripts into a complete topic analysis pipeline.

---

## V. Root `scripts/`: Batch Execution and Lightweight Statistics

### Batch Execution

- `batch_quantitative_analysis_settingB.sh`
- `batch_quantitative_analysis_settingB_fewshot.sh`
- `batch_quantitative_analysis_settingB_guidance.sh`
- `batch_calculate_difficulty_pass1.sh`

Batch execution for existing result directories, processing multiple models or multiple prompt subdirectories at once. All use `SCRIPT_DIR` / `PROJECT_ROOT` to derive repository root, not depending on local absolute paths.

### `scripts/calculate_difficulty_pass1.py`

Statistics on Pass@1 by difficulty level, supports various result structures for Setting A / B / C. Suitable for generating auxiliary tables focusing only on difficulty-Pass@1.

### `scripts/generate_settingC_summary.py`

Generates Setting C topic summary table. The difference from `utils/generate_main_table.py`: the former organizes topic summaries based on quantitative analysis results, the latter generates formal main tables directly from raw result files.

---

## Recommended Usage Paths

**Main Analysis Chain**
```
tools/quantitative_analysis_settingB_C.py
  → utils/generate_main_table.py
  → utils/convert_to_excel.py
```

**Basic Support**
```
core/  ·  utils/difficulty_utils.py  ·  utils/metrics_calculator.py
```

**Paper Topics**
```
research/  ·  scripts/generate_settingC_summary.py
```
