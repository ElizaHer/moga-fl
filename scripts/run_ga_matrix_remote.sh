#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/moga-fl
source /root/miniconda3/etc/profile.d/conda.sh
conda activate moga-fl

run_tag=20260308_moga_fl_upgrade
out_dir=outputs/fl_comp/${run_tag}/ga_moga_fl
mkdir -p "$out_dir"

base_cfg=src/configs/strategies/hybrid_opt.yaml
seeds=(42 43 44)
prefs=(time energy fairness)
algos=(moga_fl nsga3 moead)

# CPU-safe budget for full matrix execution stability.
pop=2
gen=1
rounds=3

ledger="$out_dir/ga_run_ledger.csv"
if [ ! -f "$ledger" ]; then
  echo "timestamp,algo,seed,preference,status,log,pareto_csv,best_yaml,note" > "$ledger"
fi

for pref in "${prefs[@]}"; do
  cfg="$out_dir/hybrid_opt_pref_${pref}.yaml"
  PYTHONPATH=. python - <<PY
import yaml
p="$base_cfg"
out="$cfg"
with open(p, "r", encoding="utf-8") as f:
    c=yaml.safe_load(f) or {}
c.setdefault("eval", {})["preference"] = "$pref"
# Force CPU resources to avoid CUDA OOM in long matrix runs
c.setdefault("fl", {}).setdefault("client_resources", {})["num_gpus"] = 0.0
c.setdefault("fl", {}).setdefault("client_resources", {})["num_cpus"] = 2
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
print(out)
PY

  for seed in "${seeds[@]}"; do
    for algo in "${algos[@]}"; do
      ts=$(date +"%Y-%m-%dT%H:%M:%S")
      tag="${algo}_seed${seed}_pref${pref}"
      log="$out_dir/${tag}.log"
      pareto="$out_dir/pareto_${tag}.csv"
      best="$out_dir/best_${tag}.yaml"

      set +e
      PYTHONPATH=. python -u scripts/run_ga_optimization.py \
        --config "$cfg" \
        --algo "$algo" \
        --strategy hybrid_opt \
        --pop "$pop" \
        --generations "$gen" \
        --num-rounds "$rounds" \
        > "$log" 2>&1
      rc=$?
      set -e

      if [ $rc -eq 0 ] && [ -f outputs/results/pareto_candidates.csv ] && [ -f outputs/results/best_moga_fl_config.yaml ]; then
        cp outputs/results/pareto_candidates.csv "$pareto"
        cp outputs/results/best_moga_fl_config.yaml "$best"
        echo "$ts,$algo,$seed,$pref,success,$log,$pareto,$best,rc=$rc" >> "$ledger"
      else
        echo "$ts,$algo,$seed,$pref,failed,$log,,,rc=$rc" >> "$ledger"
      fi
    done
  done
done

PYTHONPATH=. python - <<PY
import csv
from pathlib import Path
root = Path("$out_dir")
ledger = root / "ga_run_ledger.csv"
summary = root / "ga_summary.md"
rows = list(csv.DictReader(ledger.open("r", encoding="utf-8"))) if ledger.exists() else []
ok = [r for r in rows if r.get("status") == "success"]
fail = [r for r in rows if r.get("status") != "success"]
lines = []
lines.append(f"# GA Summary ({root.name})")
lines.append("")
lines.append(f"- Total runs: {len(rows)}")
lines.append(f"- Success: {len(ok)}")
lines.append(f"- Failed: {len(fail)}")
lines.append("")
lines.append("## Success Runs")
for r in ok:
    lines.append(f"- {r['algo']} seed={r['seed']} pref={r['preference']} -> {r['pareto_csv']}")
if fail:
    lines.append("")
    lines.append("## Failed Runs")
    for r in fail:
        lines.append(f"- {r['algo']} seed={r['seed']} pref={r['preference']} log={r['log']} note={r['note']}")
summary.write_text("\n".join(lines), encoding="utf-8")
print(summary)
PY
