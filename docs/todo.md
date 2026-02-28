## Experiment Plan (Hybrid vs Baselines, Reduced)
1. Goals: compare hybrid vs baselines on accuracy, loss stability, energy, upload time, fairness; validate bridge invariants; measure robustness across wireless settings.
2. Methods: strategies `hybrid_opt`, `sync`, `async`, `bridge_free`, `bandwidth_first`, `energy_first`.
3. Factors:
- Wireless: `simulated_mode = jitter` and `wsn`.
- Non-IID: `alpha = 0.5` only.
- Seed: 1 seed (e.g., 42).
- Algorithm: choose best between `fedavg` and `fedprox` for `hybrid_opt` under `wsn`, `bridge_invariants.enable=false`, `num_rounds=100`.
4. Bridge invariants:
- `hybrid_opt` runs both `bridge_invariants.enable=false` and `true`.
- All baselines run with `bridge_invariants.enable=false`.
5. Metrics: accuracy (final, AUC), loss (final, variance), energy (per round, total), upload time (per round, total), fairness (Jain mean, min), client exhaustion (count, first round), mode usage (semi-sync/bridge/async ratio, switch count).
6. Analysis: compare strategies on both wireless settings; if `hybrid_opt` is not clearly better, tune params and re-run; record pre/post settings and results.
7. Outputs: per-metric line plots (all strategies); summary tables; Pareto scatter (total upload time vs final accuracy).
