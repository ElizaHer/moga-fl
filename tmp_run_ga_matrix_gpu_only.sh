#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/moga-fl

unset FORCE_CPU || true
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
PY_BIN=/root/miniconda3/envs/moga-fl/bin/python

out_dir="outputs/fl_comp/20260308_moga_fl_upgrade/ga_moga_fl/run_20260309_012405_gpu_only"
mkdir -p "$out_dir"

base_cfg=src/configs/strategies/hybrid_opt.yaml
seeds=(42 43 44)
prefs=(time energy fairness)
algos=(moga_fl nsga3 moead)

A_POP=8; A_GEN=2; A_ROUNDS=8; A_BATCH=256; A_FRAC=0.4; A_CLIENT_GPU=0.2; A_CLIENT_CPU=2
B_POP=6; B_GEN=2; B_ROUNDS=6; B_BATCH=192; B_FRAC=0.35; B_CLIENT_GPU=0.15; B_CLIENT_CPU=2
C_POP=4; C_GEN=1; C_ROUNDS=4; C_BATCH=128; C_FRAC=0.3; C_CLIENT_GPU=0.1; C_CLIENT_CPU=2

ledger="$out_dir/ga_run_ledger.csv"
if [ ! -f "$ledger" ]; then
  echo "timestamp,algo,seed,preference,attempt,profile,status,rc,log,pareto_csv,best_yaml,note" > "$ledger"
fi

OOM_PAT='CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory'

make_cfg() {
  local pref="$1"; local prof="$2"; local cfg_out="$3"
  local pop gen rounds batch frac cgpu ccpu
  if [ "$prof" = "A" ]; then
    pop=$A_POP; gen=$A_GEN; rounds=$A_ROUNDS; batch=$A_BATCH; frac=$A_FRAC; cgpu=$A_CLIENT_GPU; ccpu=$A_CLIENT_CPU
  elif [ "$prof" = "B" ]; then
    pop=$B_POP; gen=$B_GEN; rounds=$B_ROUNDS; batch=$B_BATCH; frac=$B_FRAC; cgpu=$B_CLIENT_GPU; ccpu=$B_CLIENT_CPU
  else
    pop=$C_POP; gen=$C_GEN; rounds=$C_ROUNDS; batch=$C_BATCH; frac=$C_FRAC; cgpu=$C_CLIENT_GPU; ccpu=$C_CLIENT_CPU
  fi

  "$PY_BIN" - <<PY
import yaml
p="$base_cfg"
out="$cfg_out"
with open(p, "r", encoding="utf-8") as f:
    c=yaml.safe_load(f) or {}
c.setdefault("eval", {})["preference"] = "$pref"
c.setdefault("dataset", {})["batch_size"] = int($batch)
c.setdefault("fl", {})["num_rounds"] = int($rounds)
c.setdefault("fl", {})["fraction_fit"] = float($frac)
cr = c.setdefault("fl", {}).setdefault("client_resources", {})
cr["num_gpus"] = float($cgpu)
cr["num_cpus"] = int($ccpu)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
PY

  echo "$pop,$gen,$rounds"
}

run_one_attempt() {
  local algo="$1"; local seed="$2"; local pref="$3"; local attempt="$4"; local profile="$5"
  local ts tag log pareto best cfg pop gen rounds meta rc note
  ts=$(date +"%Y-%m-%dT%H:%M:%S")
  tag="${algo}_seed${seed}_pref${pref}_a${attempt}_${profile}"
  log="$out_dir/${tag}.log"
  pareto="$out_dir/pareto_${algo}_seed${seed}_pref${pref}.csv"
  best="$out_dir/best_${algo}_seed${seed}_pref${pref}.yaml"
  cfg="$out_dir/hybrid_opt_pref_${pref}_${profile}.yaml"

  meta=$(make_cfg "$pref" "$profile" "$cfg")
  pop=$(echo "$meta" | cut -d',' -f1)
  gen=$(echo "$meta" | cut -d',' -f2)
  rounds=$(echo "$meta" | cut -d',' -f3)

  set +e
  "$PY_BIN" -u /root/autodl-tmp/moga-fl/scripts/run_ga_optimization.py \
    --config "$cfg" \
    --algo "$algo" \
    --strategy hybrid_opt \
    --pop "$pop" \
    --generations "$gen" \
    --num-rounds "$rounds" \
    > "$log" 2>&1
  rc=$?
  set -e

  if grep -Eqi "$OOM_PAT" "$log"; then
    note="rc=$rc;oom=1"
  else
    note="rc=$rc;oom=0"
  fi

  if [ $rc -eq 0 ] && [ -f outputs/results/pareto_candidates.csv ] && [ -f outputs/results/best_moga_fl_config.yaml ]; then
    cp outputs/results/pareto_candidates.csv "$pareto"
    cp outputs/results/best_moga_fl_config.yaml "$best"
    echo "$ts,$algo,$seed,$pref,$attempt,$profile,success,$rc,$log,$pareto,$best,$note" >> "$ledger"
    return 0
  fi

  echo "$ts,$algo,$seed,$pref,$attempt,$profile,failed,$rc,$log,,,$note" >> "$ledger"
  return 1
}

for pref in "${prefs[@]}"; do
  for seed in "${seeds[@]}"; do
    for algo in "${algos[@]}"; do
      if run_one_attempt "$algo" "$seed" "$pref" 1 A; then
        continue
      fi
      log1="$out_dir/${algo}_seed${seed}_pref${pref}_a1_A.log"
      if grep -Eqi "$OOM_PAT" "$log1"; then
        if run_one_attempt "$algo" "$seed" "$pref" 2 B; then
          continue
        fi
        log2="$out_dir/${algo}_seed${seed}_pref${pref}_a2_B.log"
        if grep -Eqi "$OOM_PAT" "$log2"; then
          run_one_attempt "$algo" "$seed" "$pref" 3 C || true
        fi
      fi
    done
  done
done

"$PY_BIN" - <<PY
import csv
from pathlib import Path
import pandas as pd
root = Path("$out_dir")
ledger = root / "ga_run_ledger.csv"
summary = root / "ga_summary.md"
rows = list(csv.DictReader(ledger.open("r", encoding="utf-8"))) if ledger.exists() else []
latest = {}
for r in rows:
    k = (r.get("algo"), r.get("seed"), r.get("preference"))
    a = int(r.get("attempt") or 0)
    if k not in latest or a >= int(latest[k].get("attempt") or 0):
        latest[k] = r
final = list(latest.values())
ok = [r for r in final if r.get("status") == "success"]
fail = [r for r in final if r.get("status") != "success"]

def load_best(path):
    if not path:
        return None
    p=Path(path)
    if not p.exists():
        return None
    df=pd.read_csv(p)
    req=["acc","time","energy","comm_cost","fairness"]
    if df.empty or any(c not in df.columns for c in req):
        return None
    row=df.sort_values(["acc","fairness","time","energy","comm_cost"], ascending=[False,False,True,True,True]).iloc[0]
    return {k: float(row[k]) for k in req}

def ge3(a,b):
    w=0
    if a["acc"] > b["acc"]: w+=1
    if a["fairness"] > b["fairness"]: w+=1
    if a["time"] < b["time"]: w+=1
    if a["energy"] < b["energy"]: w+=1
    if a["comm_cost"] < b["comm_cost"]: w+=1
    return w>=3

dom=[]
for seed in ["42","43","44"]:
    for pref in ["time","energy","fairness"]:
        m=latest.get(("moga_fl",seed,pref))
        n=latest.get(("nsga3",seed,pref))
        d=latest.get(("moead",seed,pref))
        if not (m and n and d):
            dom.append(f"- seed={seed} pref={pref}: insufficient records")
            continue
        mo=load_best(m.get("pareto_csv",""))
        no=load_best(n.get("pareto_csv",""))
        do=load_best(d.get("pareto_csv",""))
        if not (mo and no and do):
            dom.append(f"- seed={seed} pref={pref}: missing valid objective csv")
            continue
        hit = ge3(mo,no) or ge3(mo,do)
        dom.append(f"- seed={seed} pref={pref}: moga_fl_>=3obj_vs_ablation={'yes' if hit else 'no'}")

lines=[]
lines.append(f"# GA Summary ({root.name})")
lines.append("")
lines.append("- Total jobs: 27")
lines.append(f"- Final success: {len(ok)}")
lines.append(f"- Final failed: {len(fail)}")
lines.append(f"- OOM failed: {sum('oom=1' in (r.get('note') or '') for r in fail)}")
lines.append(f"- Non-OOM failed: {sum('oom=1' not in (r.get('note') or '') for r in fail)}")
lines.append("")
lines.append("## Final Status By Job")
for r in sorted(final, key=lambda x:(x.get('preference',''), x.get('seed',''), x.get('algo',''))):
    lines.append(f"- {r.get('algo')} seed={r.get('seed')} pref={r.get('preference')} status={r.get('status')} profile={r.get('profile')} log={r.get('log')}")
lines.append("")
lines.append("## Dominance Check (moga_fl vs ablations, same seed+pref)")
lines.extend(dom)
summary.write_text("\n".join(lines), encoding="utf-8")
print(summary)
PY

echo "RUN_DONE $out_dir"