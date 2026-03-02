# Accepted Matrix Results (20260303_034927)

- Run tag: `20260303_034927`
- Matrix dir: `B_matrix_tuned_A2`
- Objective: hybrid-vs-baselines under WSN + jitter with strict attempt ledger and latest-success manifest.

## Why Accepted

- Full matrix completed with 11/11 successful jobs.
- Every selected CSV reached round `>=60`.
- Jitter setting achieved clear hybrid advantage for `hybrid_jitter_invFalse` on final accuracy.

## Key Metrics (Jitter, final round)

- `hybrid_jitter_invFalse`: acc `0.7965`, loss `0.6258`
- `bridge_free_jitter_invTrue`: acc `0.7927`, loss `0.6365`
- `sync_jitter_invTrue`: acc `0.7804`, loss `0.6745`
- `async_jitter_invTrue`: acc `0.7626`, loss `0.7799`
- `hybrid_jitter_invTrue`: acc `0.6848`, loss `1.1044`

## Tuning Delta (Baseline -> A2)

- `fl.local_epochs`: `1 -> 2`
- `fl.lr`: `0.001 -> 0.0005`
- `algorithm.fedprox_mu`: `0.05 -> 0.1`
- `controller.semi_sync_wait_ratio`: `0.85 -> 0.75`
- `controller.gate_thresholds`: `to_async 0.65 -> 0.55`, `to_semi_sync 0.38 -> 0.45`
- `controller.hysteresis_margin`: `0.03 -> 0.02`
- `controller.bridge_rounds`: `4 -> 2`
- `controller.min_rounds_between_switch`: `4 -> 2`
- `fedbuff.buffer_size`: `10 -> 6`
- `fedbuff.min_updates_to_aggregate`: `5 -> 3`
- `fedbuff.staleness_alpha`: `1.5 -> 1.0`
- `fedbuff.max_staleness`: `6 -> 8`
- `wireless.base_snr_db`: `8.0 -> 6.0`
- `wireless.per_k`: `1.0 -> 0.8`
- `wireless.block_fading_intensity`: `1.0 -> 1.4`
- `wireless.bandwidth_budget_mb_per_round`: `12.0 -> 8.0`
- `scheduler.weights`: `0.45/0.35/0.15/0.05 -> 0.45/0.30/0.20/0.05`

## Notes

- TODO text says "11 jobs" while explicit list enumerates 10; this run keeps 11 by adding `hybrid_wsn_invFalse` as disambiguation job.
- `hybrid_jitter_invTrue` is still weaker than top jitter baselines, so bridge-invariants strictness remains a known tuning gap.
