#!/usr/bin/env bash
set -euo pipefail

# 【Setting C】FCR vs toolcalls：端到端统计 + 画图
#
# 功能
# - Step 1: 跑 analysis/fcr_vs_toolcalls_analysis.py 生成 fcr_vs_toolcalls_data.json
# - Step 2: 打印 n(k) 分布
# - Step 3: 交互式输入每个模型的 cohort-k（samples with effective_calls >= k）并绘制图一
# - Step 4: 交互式询问是否需要绘制图二（单模型截断 FCR + 累计 Hit Precision），为每个模型指定 k 值
#
# 用法示例
#   1. 直接运行（使用脚本内硬编码的模型列表）
#      bash analysis/scripts/run_fcr_vs_toolcalls_plots.sh
#
#   2. 命令行覆盖模型列表
#      bash analysis/scripts/run_fcr_vs_toolcalls_plots.sh --models MindWatcher_round2,minimax-m2_1
#
# 说明
# - MODELS 可在下方第29行直接修改，逗号分隔的模型目录名
# - 脚本会在 $MODELS_DIR/<model>/re_evaluated_results.jsonl 读取数据
# - cohort-k 和 pick-k 均为交互式输入（看完 n(k) 数据后再决定）

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ANALYSIS_PY="$ROOT/analysis/fcr_vs_toolcalls_analysis.py"
PLOT_COHORT_PY="$ROOT/analysis/scripts/plot_fcr_vs_toolcalls_cohort_and_examples.py"
PLOT_TRUNC_PY="$ROOT/analysis/scripts/plot_fcr_vs_toolcalls_truncated.py"
REPORT_NK_PY="$ROOT/analysis/scripts/report_nk_for_fcr_toolcalls.py"
JSON_PATH="$ROOT/fcr_vs_toolcalls_data.json"

# 在此填写要分析的模型列表（逗号分隔），例如 "minimax-m2_1,MindWatcher_round2"
MODELS="minimax-m2_1"
MODELS_DIR="$ROOT/setting_c_results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"; shift 2;;
    --models-dir)
      MODELS_DIR="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2;;
  esac
done

if [[ -z "$MODELS" ]]; then
  echo "ERROR: MODELS is empty. Please set MODELS variable in the script or use --models" >&2
  exit 2
fi

# 根据模型列表生成唯一标识（用于文件命名）
MODELS_TAG=$(echo "$MODELS" | tr ',' '_')
JSON_PATH="$ROOT/fcr_vs_toolcalls_data_${MODELS_TAG}.json"
OUT_DIR="$ROOT/analysis_results/fcr_vs_toolcalls_${MODELS_TAG}"
mkdir -p "$OUT_DIR"

echo "模型列表: $MODELS"
echo "数据文件: $JSON_PATH"
echo "输出目录: $OUT_DIR"
echo ""

# Step 1: 分析数据（如果文件已存在，询问是否复用）
if [[ -f "$JSON_PATH" ]]; then
  echo "[1/4] 检测到已存在的分析数据: $JSON_PATH"
  python3 - <<PY
import sys
try:
    tty = open('/dev/tty', 'r')
except OSError:
    print("错误：无法打开 /dev/tty 进行交互输入", file=sys.stderr)
    sys.exit(1)

print("是否复用已有数据？(y/n，回车默认复用): ", end='', flush=True)
s = tty.readline().strip().lower()
tty.close()

if s in ['n', 'no']:
    print("将重新分析数据")
    sys.exit(1)  # 退出码1表示需要重新分析
else:
    print("复用已有数据")
    sys.exit(0)  # 退出码0表示复用
PY

  if [[ $? -eq 0 ]]; then
    echo "跳过分析步骤"
  else
    echo "重新运行分析: $ANALYSIS_PY"
    python3 "$ANALYSIS_PY" --models "$MODELS" --models-dir "$MODELS_DIR" --max-effective-calls 32 --output "$JSON_PATH"
  fi
else
  echo "[1/4] 运行分析: $ANALYSIS_PY"
  python3 "$ANALYSIS_PY" --models "$MODELS" --models-dir "$MODELS_DIR" --max-effective-calls 32 --output "$JSON_PATH"
fi

echo ""
echo "[2/4] 打印 n(k) 分布表"
python3 "$REPORT_NK_PY" --input "$JSON_PATH" --max-k 32

echo "[3/4] 交互式选择每个模型的 cohort-k"
# 交互式输入：为每个模型指定 cohort-k（samples with calls >= k）
# 注意：使用 /dev/tty 读取输入以支持 heredoc 脚本中的交互
python3 - <<PY
import json
import sys
from pathlib import Path

json_path = Path("$JSON_PATH")
data = json.loads(json_path.read_text(encoding='utf-8'))
models = list(data.keys())

print("\n请为每个模型输入 cohort-k 值（整数，k>=1）")
print("含义：只保留 effective_calls >= k 的样本用于该模型的统计")
print("直接回车跳过该模型（不生成图一）\n")

# 打开真正的终端进行交互（heredoc 中 stdin 不可用）
try:
    tty = open('/dev/tty', 'r')
except OSError:
    print("错误：无法打开 /dev/tty 进行交互输入", file=sys.stderr)
    sys.exit(1)

chosen = {}
for m in models:
    while True:
        print(f"  {m} cohort-k（回车跳过）: ", end='', flush=True)
        s = tty.readline().strip()
        if s == "":
            print(f"    已跳过 {m}")
            break
        try:
            k = int(s)
            if k < 1:
                raise ValueError
            chosen[m] = k
            print(f"    {m} 将使用 cohort-k={k}")
            break
        except Exception:
            print("    输入无效，请输入 >= 1 的整数，或回车跳过")

tty.close()

if not chosen:
    print("\n未选择任何模型，跳过图一生成\n")
    sys.exit(0)

print("\n已选择的模型 cohort-k:")
for m,k in chosen.items():
    print(f"  {m}: calls>={k}")

# 把选择写到临时文件，供后续绘图步骤读取
out = Path("$OUT_DIR") / "chosen_cohort_k.json"
out.write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"已保存配置到: {out}\n")
PY

echo "[3/4] Plot cohort marginal newfacts (per-model cohorts)"
python3 - <<PY
import json
from pathlib import Path
from analysis.scripts import plot_fcr_vs_toolcalls_cohort_and_examples as mod

json_path = Path("$JSON_PATH")
out_dir = Path("$OUT_DIR")
min_calls = 0
chosen_path = out_dir / "chosen_cohort_k.json"

cohort_k = json.loads(chosen_path.read_text(encoding='utf-8'))
data = json.loads(json_path.read_text(encoding='utf-8'))

# 画满 1..32（不截断）
per_model_kmax = {m: 32 for m in cohort_k.keys()}

mod.plot_cohort_curves(
    data,
    out_dir,
    min_calls=min_calls,
    per_model_kmax=per_model_kmax,
    per_model_min_calls=cohort_k,
)
print(f"Saved cohort figures into: {out_dir}")
PY

echo "[4/4] 为选定模型生成图二（可选）"
# 交互式询问是否需要生成图二（truncated FCR + hit precision）
python3 - <<PY
import json
import sys
from pathlib import Path

json_path = Path("$JSON_PATH")
data = json.loads(json_path.read_text(encoding='utf-8'))
models = list(data.keys())

print("\n是否需要生成图二（截断 FCR + 累计 Hit Precision）？")
print("为每个模型指定一个 k 值（cohort: 只包含 effective_calls >= k 的样本）")
print("或者直接回车跳过该模型\n")

# 打开真正的终端进行交互
try:
    tty = open('/dev/tty', 'r')
except OSError:
    print("错误：无法打开 /dev/tty 进行交互输入", file=sys.stderr)
    sys.exit(1)

picks = []
for m in models:
    while True:
        print(f"  {m} k 值（回车跳过）: ", end='', flush=True)
        s = tty.readline().strip()
        if s == "":
            print(f"    已跳过 {m}")
            break
        try:
            k = int(s)
            if k < 1:
                raise ValueError
            picks.append((m, k))
            print(f"    将为 {m} 生成 k={k} 的截断 FCR 图")
            break
        except Exception:
            print("    输入无效，请输入 >= 1 的整数，或回车跳过")

tty.close()

if not picks:
    print("\n未选择任何模型生成图二，已跳过\n")
else:
    print(f"\n正在生成 {len(picks)} 个截断 FCR 图...\n")
    import subprocess
    for m, k in picks:
        out = Path("$OUT_DIR") / f"truncated_fcr_{m}_k_{k}.png"
        cmd = [
            "python3", "$PLOT_TRUNC_PY",
            "--input", "$JSON_PATH",
            "--model", m,
            "--k", str(k),
            "--out", str(out)
        ]
        print(f"  - {m} k={k} -> {out}")
        subprocess.run(cmd, check=True)
PY

echo "完成！输出目录: $OUT_DIR"