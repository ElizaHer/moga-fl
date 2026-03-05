# Accepted Matrix Results (20260306_014534)

- Run tag: `20260306_014534`
- Matrix: `B_matrix_20260306_regression`
- Goal: improve `hybrid_jitter_invTrue` with upgraded client scoring and energy-aware selection.

## Why accepted
- 11/11 jobs succeeded.
- Selected CSVs all reached round >= 60.
- In jitter setting, `hybrid_jitter_invTrue` achieved the highest final accuracy in this regression run.

## Key jitter metrics (final round)
- hybrid_jitter_invTrue: acc=0.7645, loss=0.6698
- async_jitter_invTrue: acc=0.7545, loss=0.7848
- energy_first_jitter_invTrue: acc=0.7335, loss=0.7785
- bandwidth_first_jitter_invTrue: acc=0.7314, loss=0.7838
- bridge_free_jitter_invTrue: acc=0.7292, loss=0.7628
- hybrid_jitter_invFalse: acc=0.7013, loss=0.8504
- sync_jitter_invTrue: acc=0.5944, loss=1.3160

## Main changes
- Scheduler scoring now includes:
  - data_value proxy,
  - historical_contribution EMA,
  - energy survival factor.
- Added hard energy guardrails and anti-monopoly cooldown.
- Jitter profile used: `p1_async_sensitive_relaxed_inv`.
- Params:
  - wsn: lr=0.0005, fedprox_mu=0.01
  - jitter: lr=0.0005, fedprox_mu=0.01
  - batch_size=768
  - client_resources: num_cpus=5, num_gpus=0.2
