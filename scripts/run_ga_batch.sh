#!/usr/bin/env bash

# 批量运行 GA 优化实验（适用于 Linux 环境）
#
# 使用方式示例：
#   chmod +x scripts/run_ga_batch.sh
#   ./scripts/run_ga_batch.sh               # 使用脚本内置的 generations/pop
#   GA_GENERATIONS=8 GA_POP=24 ./scripts/run_ga_batch.sh
#   ./scripts/run_ga_batch.sh 4 12          # 通过位置参数覆盖 generations/pop
#
# 实验组合：若干 (algo × pop × generations × strategy_yaml)，串行执行。
# 每个实验的结果会被整理到 outputs/ga_experiments/<exp_name>/ 下。

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 1

mkdir -p outputs/ga_experiments

GEN_OVERRIDE_ENV=${GA_GENERATIONS:-}
POP_OVERRIDE_ENV=${GA_POP:-}
GEN_OVERRIDE_ARG=${1:-}
POP_OVERRIDE_ARG=${2:-}

FAILED_LOG="outputs/ga_experiments/failed_experiments.log"
: > "$FAILED_LOG"

run_one() {
  local algo="$1"
  local strategy="$2"
  local base_gen="$3"
  local base_pop="$4"

  local gen="$base_gen"
  local pop="$base_pop"

  # 优先使用命令行参数，其次环境变量，最后落回实验自身默认值
  if [[ -n "$GEN_OVERRIDE_ARG" ]]; then
    gen="$GEN_OVERRIDE_ARG"
  elif [[ -n "$GEN_OVERRIDE_ENV" ]]; then
    gen="$GEN_OVERRIDE_ENV"
  fi

  if [[ -n "$POP_OVERRIDE_ARG" ]]; then
    pop="$POP_OVERRIDE_ARG"
  elif [[ -n "$POP_OVERRIDE_ENV" ]]; then
    pop="$POP_OVERRIDE_ENV"
  fi

  local exp_name="algo_${algo}_strategy_${strategy}_g${gen}_p${pop}"
  local exp_dir="outputs/ga_experiments/${exp_name}"

  echo "[INFO] Running experiment: ${exp_name}" | tee -a "$FAILED_LOG"

  local attempt
  local success=0
  for attempt in 1 2 3; do
    echo "[INFO]  Attempt ${attempt} for ${exp_name}" | tee -a "$FAILED_LOG"
    python scripts/run_ga_optimization.py \
      --config "src/configs/strategies/${strategy}.yaml" \
      --strategy "${strategy}" \
      --algo "${algo}" \
      --generations "${gen}" \
      --pop "${pop}"
    status=$?
    if [[ $status -eq 0 ]]; then
      success=1
      break
    else
      echo "[WARN]  Attempt ${attempt} failed for ${exp_name} (status=${status})" | tee -a "$FAILED_LOG"
    fi
  done

  if [[ "$success" -ne 1 ]]; then
    echo "[ERROR] Experiment ${exp_name} failed after 3 attempts" | tee -a "$FAILED_LOG"
    return
  fi

  mkdir -p "$exp_dir"
  if [[ -f outputs/results/pareto_candidates.csv ]]; then
    mv outputs/results/pareto_candidates.csv "${exp_dir}/pareto_candidates.csv"
  fi
  if [[ -f outputs/results/best_moga_fl_config.yaml ]]; then
    mv outputs/results/best_moga_fl_config.yaml "${exp_dir}/best_moga_fl_config.yaml"
  fi

  echo "[INFO] Experiment ${exp_name} finished and results stored in ${exp_dir}" | tee -a "$FAILED_LOG"
}

# 设计若干实验组合（6–12 个左右），覆盖不同算法和策略 YAML
# (algo, strategy, generations, pop)
run_one "nsga3"      "hybrid_opt" 4 12
run_one "moga_fl"    "hybrid_opt" 6 16
run_one "moead"      "hybrid_opt" 6 16
run_one "pymoo_nsga3" "hybrid_opt" 6 24
run_one "nsga3"      "sync"       4 12
run_one "nsga3"      "async"      4 12
run_one "moga_fl"    "sync"       6 16
run_one "moead"      "async"      6 16
