# Matrix Experiment TODO (Authoritative)

Last updated: 2026-03-03 (rev2)
Project: `moga-fl`

## 1. Background
- We compare `hybrid_opt` against baselines under two wireless settings: `wsn` and `jitter`.
- Prior runs showed weak hybrid advantage, partly due to mixed/invalid outputs.
- This TODO is the single source of truth for future reruns.
- If context is missing, read `todo.md` and `agent.md` first.

## 2. Mandatory Experiment Matrix
- Common factors:
  - `alpha=0.5`
  - `seed=42`
  - `algorithm=fedprox`
  - `num_rounds=60`
- `wsn` jobs:
  - `hybrid_opt` (use inv=true config)
  - `bandwidth_first` (use inv=true config)
  - `energy_first` (use inv=true config)
- `jitter` jobs:
  - `hybrid_opt` with `bridge_invariants.enable=false`
  - `hybrid_opt` with `bridge_invariants.enable=true`
  - `sync` (use inv=true config)
  - `async` (use inv=true config)
  - `bridge_free` (use inv=true config)
  - `bandwidth_first` (use inv=true config)
  - `energy_first` (use inv=true config)

Total jobs: 11

## 3. Success/Failure Rule (Critical)
- A run is **successful** only if the run log contains a final line like:
  - `Metrics CSV: outputs\hybrid_metrics\hybrid_opt\xxxx.csv`
- **Do not judge while process is still running.**
- Success/failure is evaluated **only after one attempt finishes** (normal exit, non-zero exit, timeout, or manual kill).
- If process has not ended yet, keep waiting; do not trigger rerun.
- After process ends, if that line is missing, treat as failed and rerun.
- Never treat `.err` warnings alone as failure if log has valid `Metrics CSV` line.
- After success, copy that exact CSV into job output folder.

## 4. File Preservation Rule
- Never delete existing `.csv/.log/.err`.
- Never overwrite existing files.
- Every rerun must create a new attempt index (`attemptN`).
- Keep a chronological ledger: `attempt_ledger.csv` with fields:
  - `timestamp,tag,attempt,status,log,err,metrics_csv,copied_csv,note`

## 5. Latest CSV Rule for Final Stats
- For each job, choose the **latest successful** `copied_csv` by ledger timestamp.
- If ledger is unavailable, fallback to latest CSV by file time.
- Validate selected CSV: last round must be `>=60`.

## 6. Tuning Loop Requirement
- Baseline target: in `jitter`, hybrid should generally outperform non-hybrid baselines in accuracy/loss stability while keeping energy/time reasonable.
- If hybrid is not clearly better:
  - adjust controller thresholds/hysteresis/bridge duration,
  - adjust scheduler weights,
  - adjust FedBuff (`buffer_size/min_updates/interval/staleness_alpha/max_staleness`),
  - adjust wireless/channel/energy related params in `hybrid_opt.yaml`,
  - adjust FL training params (`num_rounds/local_epochs/lr/fraction_fit`),
  - optionally adjust `fedprox_mu`.
- Record tuned values before and after rerun.

## 6.1 Aggressive Tuning Playbook (Use `hybrid_opt.yaml` Fully)
- `dataset.alpha`:
  - effect: stronger non-IID stress influences hybrid advantage.
  - bold range: `0.2, 0.3, 0.5` (keep official matrix at `0.5`, but use lower alpha for stress validation runs).
- `dataset.batch_size`:
  - effect: larger batch stabilizes local updates, may improve FedProx behavior.
  - bold range: `64, 128, 256`.
- `fl.local_epochs`:
  - effect: larger local drift can hurt sync/baselines under jitter; hybrid+prox may gain if tuned.
  - bold range: `1 -> 2 -> 3` (check overfitting/drift).
- `fl.lr`:
  - effect: too high causes instability, too low hides strategy differences.
  - bold range: `0.0005, 0.001, 0.002`.
- `fl.fraction_fit` and `scheduler.selection_top_k`:
  - effect: control participation width and fairness/communication tradeoff.
  - bold actions:
    - fixed top-k regime: `selection_top_k=4/5/6` with stable fraction.
    - broad regime: `fraction_fit=0.6~0.8` and `selection_top_k=0`.
- `algorithm.fedprox_mu`:
  - effect: suppresses client drift in non-IID; can boost hybrid robustness.
  - bold range: `0.01, 0.02, 0.05, 0.1`.
- `wireless.base_snr_db`, `wireless.per_k`, `wireless.block_fading_intensity`:
  - effect: directly controls packet-loss hardness and mode switching pressure.
  - bold stress profile:
    - harder jitter: `base_snr_db 8->6`, `per_k 1.0->0.8`, `block_fading_intensity 1.0->1.4`.
  - bold easier profile:
    - `base_snr_db 8->10`, `per_k 1.0->1.2`.
- `wireless.bandwidth_budget_mb_per_round`:
  - effect: controls upload bottleneck severity.
  - bold range: `8, 12, 16`.
- `wireless.tx_power_watts`, `compute_power_watts`, `compute_rate_samples_per_sec`:
  - effect: changes energy pressure and bridge invariants activation.
  - bold stress profile: higher power + lower compute rate to amplify energy constraints.
- `energy.initial_client_energy` and `energy.client_initial_energies`:
  - effect: heterogeneity can reveal fairness/energy management advantages.
  - bold action:
    - use explicit vector for 10 clients (mixed high/low), not only scalar default.
- `scheduler.fair_window_size`:
  - effect: short window reacts fast, long window stabilizes fairness trend.
  - bold range: `4, 8, 12`.
- `scheduler.weights.channel_w/data_w/fair_w/energy_w`:
  - effect: participation preference shifts between quality, data value, fairness, energy.
  - bold profiles:
    - quality-first: `0.45/0.35/0.15/0.05`
    - fairness-first: `0.20/0.20/0.45/0.15`
    - energy-safe: `0.25/0.20/0.20/0.35`
- `controller.semi_sync_wait_ratio`:
  - effect: higher ratio includes slower clients (fairness↑, latency↑).
  - bold range: `0.65, 0.75, 0.85, 0.9`.
- `controller.gate_thresholds`, `hysteresis_margin`, `window_size`:
  - effect: mode switch sensitivity/stability.
  - bold profiles:
    - async-sensitive: `to_async=0.55,to_semi=0.45,margin=0.02`.
    - stability-first: `to_async=0.68,to_semi=0.38,margin=0.06,window=8`.
- `controller.bridge_rounds`, `min_rounds_between_switch`:
  - effect: transition smoothness vs responsiveness.
  - bold range: `bridge_rounds 2/4/6`, `min_rounds_between_switch 2/4/6`.
- `controller.gate_weights.per/fairness/energy`:
  - effect: determines why switching happens.
  - bold profiles:
    - PER-driven: `0.65/0.20/0.15`
    - fairness-recovery: `0.35/0.45/0.20`
    - energy-protective: `0.35/0.20/0.45`
- `controller.bandwidth_rebalance.low_energy_factor/high_energy_factor`:
  - effect: dynamic communication budget squeeze/release.
  - bold range: low factor `0.5~0.9`, high factor `1.0~1.2`.
- `controller.bridge_invariants.*`:
  - effect: bridge period intervention intensity.
  - bold actions:
    - relaxed invariants for performance: larger budgets, higher thresholds.
    - strict invariants for safety: lower budgets, lower thresholds, stronger downweight/throttle.
- `fedbuff.buffer_size/min_updates_to_aggregate/async_agg_interval`:
  - effect: async aggregation freshness vs variance.
  - bold range: `buffer_size 6~20`, `min_updates 3~10`, `interval 1~3`.
- `fedbuff.staleness_alpha/max_staleness`:
  - effect: stale update penalty and acceptance horizon.
  - bold range: `alpha 0.8~2.0`, `max_staleness 4~12`.

## 6.2 Recommended High-Impact Tuning Bundles
- Bundle A (Hybrid advantage in jitter):
  - `fedprox_mu=0.05`, `semi_sync_wait_ratio=0.85`, `to_async=0.65`, `to_semi=0.38`,
  - `bridge_rounds=4`, `min_rounds_between_switch=4`,
  - `fedbuff: buffer=10,min_updates=5,interval=1,alpha=1.5,max_staleness=6`,
  - scheduler weights use quality-first profile.
- Bundle B (Fairness recovery):
  - fairness-first scheduler weights,
  - gate weights fairness-recovery profile,
  - `fair_window_size=8`,
  - enable `client_initial_energies` heterogeneity.
- Bundle C (Energy-constrained stress):
  - lower initial energy and stricter invariants,
  - energy-safe scheduler profile,
  - `bandwidth_budget_mb_per_round=8`,
  - compare hybrid inv=false vs inv=true gap.

## 6.3 Notification Requirement (Per Single Job)
- For each single experiment job, send WeChat notifications:
  - at job start: include experiment tag and key parameters,
  - at job end: include tag, rounds, final accuracy, final loss, chosen csv path, and success/failure.
- Required script template:
```python
import requests
headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjkwMjkwNiwidXVpZCI6ImE5NDViOTgzNWQwOGY4YjUiLCJpc19hZG1pbiI6ZmFsc2UsImJhY2tzdGFnZV9yb2xlIjoiIiwiaXNfc3VwZXJfYWRtaW4iOmZhbHNlLCJzdWJfbmFtZSI6IiIsInRlbmFudCI6ImF1dG9kbCIsInVwayI6IiJ9.sw0UVJSDJ8CZ7O6cT9_OQOfWQKi6TwW9JQVB0FGp8SMQueBsN-0mEETOV857BNZBouh2yy-MwJQL7VRE1u0XSg"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "实验开始/结束",
                         "name": "实验名称",
                         "content": "eg. Epoch=100. Acc=90.2"
                     }, headers=headers)
print(resp.content.decode())
```
- Integrate this call into runner start/end hooks; do not send only once for the whole matrix.

## 7. Output Artifacts
- `matrix_manifest.csv` (11 selected final CSVs)
- `attempt_ledger.csv` (all attempts, success/failure)
- `analysis/B_matrix_tuned/summary_runs.csv`
- `analysis/B_matrix_tuned/summary_by_strategy.csv`
- Plot images from `generate_comparison_plots.py` under `analysis/plots`

## 7.1 Expected-Result Archival + Git Commit (Mandatory)
- When results meet expectation (hybrid advantage is clear under target setting), archive both:
  - tuned parameter files (yaml/config/scripts),
  - corresponding selected result files (csv/summary/plots/manifest/ledger),
  into a dedicated folder, e.g.:
  - `outputs/fl_comp/<run_tag>/accepted/`
- The archival folder must be self-contained and include:
  - `README.md` with run tag, key params, why accepted, key metrics.
- After archival, create a git commit including these accepted params/results for downstream processing.
- Suggested commit message:
  - `auto(exp): archive accepted hybrid matrix results and tuned configs (<run_tag>)`

## 8. Script Self-Repair Policy
- If runner scripts mis-handle strategy metric folders or success criteria, agent must patch scripts.
- Prior known pitfall:
  - some strategies still write metrics under `outputs/hybrid_metrics/hybrid_opt`
  - therefore, rely on log-extracted `Metrics CSV` path, not hardcoded strategy folder.
- Patch is allowed and expected when needed.

## 9. Completion Checklist
- 11/11 jobs have valid final CSV.
- Every selected CSV passes round-count validation.
- Ledger and manifest are consistent.
- Plots are generated from final selected matrix directory.
- Final summary explicitly states which config version and tuning was used.
- Accepted results/params are copied into `accepted/` folder.
- Accepted artifacts are committed to git.

## 10. Server Access
- Passwordless server connection command:
  - `ssh -p 14863 root@connect.bjb2.seetacloud.com`

## 11. Latest Run Log (2026-03-03)
- Run#1: `outputs/fl_comp/20260303_014057/B_matrix_tuned`
- Run#2 (tuned A2): `outputs/fl_comp/20260303_034927/B_matrix_tuned_A2`
- Acceptance archive: `outputs/fl_comp/20260303_034927/accepted/`
- Round validation: selected CSV final round all `>=60` in both runs.
- Retry stats: both runs achieved `11/11` success with single attempt per job.
- Current status: `hybrid_jitter_invFalse` is strongest in jitter accuracy; `hybrid_jitter_invTrue` still underperforms top baselines and needs further bridge-invariants tuning.
