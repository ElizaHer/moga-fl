# Agent Handoff Prompt (Run Matrix + Tune + Plot)

You are taking over FL matrix experiments in `moga-fl` with incomplete context.
Before doing anything, read:
1. `todo.md` (authoritative plan)
2. `docs/algorithms_explained.md` (method context)
3. `run_experiments.ps1` and local runner scripts

If memory/context is lost later, re-read `todo.md` and `agent.md`.

---

## Primary Objective
Run a complete, reproducible matrix experiment for hybrid-vs-baselines, with retry logic, parameter tuning, and correct plotting, while preserving all history files.

## Hard Constraints
- Do not delete existing `.csv/.log/.err` files.
- Do not overwrite existing attempt files.
- Add new attempts only (`attemptN`).
- Keep chronological attempt records in `attempt_ledger.csv`.
- Final stats must use latest successful CSV per job (by ledger timestamp).

## Experiment Matrix
Use exactly the 11 jobs in `todo.md`.
Common factors: `alpha=0.5`, `seed=42`, `algorithm=fedprox`, `num_rounds=60`.

## Success Criterion for One Attempt
A run attempt is successful only when run `.log` ends with a line matching:
`Metrics CSV: outputs\\hybrid_metrics\\hybrid_opt\\xxxx.csv`
Then copy this exact CSV into job output folder with unique name:
`<tag>_attempt<k>_<csv_basename>.csv`

Do not evaluate success/failure while the process is still running.
Evaluate only after this attempt ends (normal exit, non-zero exit, timeout, or kill).
If no `Metrics CSV:` line appears after attempt end, mark failed and rerun.
Do not reject success only because `.err` contains warnings.

## Required Execution Procedure
1. Ensure dedicated run directory under `outputs/fl_comp/<run_tag>/`.
2. Prepare tuned config files (`inv_false` and `inv_true`).
3. Integrate per-job notification hooks (start/end) into the runner.
4. Execute matrix jobs with retry (max 3 attempts per job).
5. Append each attempt to ledger with status and file paths.
6. After all jobs complete, build `matrix_manifest.csv` from latest successful CSV per tag.
7. Validate each selected CSV has final round >= 60.
8. Generate summaries with `analyze_results.py`.
9. Generate plots with `generate_comparison_plots.py` using the new matrix dir.

## Tuning Guidance
If hybrid advantage is weak, tune and rerun matrix:
- dataset: `alpha`, `batch_size`
- fl: `local_epochs`, `lr`, `fraction_fit`, `num_rounds`
- wireless: `base_snr_db`, `per_k`, `block_fading_intensity`, `bandwidth_budget_mb_per_round`, `tx_power_watts`, `compute_power_watts`, `compute_rate_samples_per_sec`
- algorithm: `fedprox_mu`
- energy: `initial_client_energy`, `client_initial_energies` (vector length must equal `num_clients`)
- scheduler: `selection_top_k`, `fair_window_size`, weights (`channel_w`, `data_w`, `fair_w`, `energy_w`)
- controller: `semi_sync_wait_ratio`, `window_size`, `gate_thresholds`, `hysteresis_margin`, `bridge_rounds`, `min_rounds_between_switch`, `gate_weights`, `bandwidth_rebalance`
- bridge invariants (`inv_true`): budgets, thresholds, downweight/throttle/extend settings
- fedbuff: `buffer_size`, `min_updates_to_aggregate`, `async_agg_interval`, `staleness_alpha`, `max_staleness`

Record before/after parameter values in notes.

Prefer bold bundle-style tuning (A/B/C) from `todo.md` instead of tiny single-parameter moves.

## Notification Hook (Must Implement)
For every single job attempt:
- On start: send message with tag + key params.
- On end: send message with tag + status + rounds + final accuracy + final loss + selected csv.

Use this script payload:
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

Implementation detail:
- Build helper function `send_wechat(title, name, content)` in runner.
- Call it at job start and job end.
- End-content must include metrics parsed from result csv/log.

## Script Repair Permission
You are allowed to patch scripts if they are incorrect.
Known issue pattern:
- strategy-specific metrics directories may not match real output path.
- Use log-extracted `Metrics CSV` as source of truth.

If script behavior conflicts with `todo.md`, patch scripts and continue.

## Final Deliverables
- `todo.md` and `agent.md` preserved and updated
- `attempt_ledger.csv`
- `matrix_manifest.csv`
- `summary_runs.csv` and `summary_by_strategy.csv`
- comparison plots under analysis output
- brief report: selected latest CSVs, failures/retries, tuned parameters, whether hybrid improvement is achieved

## Accepted Result Handling (Mandatory)
If results are considered expected/qualified:
1. Copy tuned params + selected result artifacts into a dedicated archival folder:
   - `outputs/fl_comp/<run_tag>/accepted/`
2. Ensure archival folder contains:
   - configs/yaml used,
   - selected csv files,
   - summary files,
   - plots,
   - manifest + ledger,
   - `README.md` describing acceptance rationale and key metrics.
3. Commit accepted artifacts to git for downstream processing.
   - Example commit message:
   - `feat(exp): archive accepted hybrid matrix results and tuned configs (<run_tag>)`

## Server Login (Passwordless)
Use this command to connect server:
`ssh -p 49692 root@connect.bjb2.seetacloud.com`
