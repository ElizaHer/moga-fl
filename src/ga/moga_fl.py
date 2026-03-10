from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .constraints import penalty, repair
from .moead import MOEAD
from .nsga3 import NSGA3
from .objectives import evaluate_solution
from .pareto import non_dominated_set


class MOGAFLController:
    """Improved MOGA-FL controller.

    - Joint chromosome encodes scheduler weights/top-k, bandwidth factor,
      staleness decay, bridge trigger thresholds, and compression ratio.
    - Two-stage co-evolution: NSGA-III (distribution) -> MOEA/D (convergence).
    - Island migration to reduce premature convergence.
    - Feasible-region local search near energy/bandwidth/staleness constraints.
    - Multi-fidelity evaluation: early low-fidelity and late high-fidelity.
    - Elite memory cache to avoid repeated expensive evaluations.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        low_fidelity_eval: Callable[[Dict[str, Any]], Dict[str, float]],
        high_fidelity_eval: Callable[[Dict[str, Any]], Dict[str, float]] | None = None,
        pop_size: int = 20,
        n_islands: int = 2,
    ) -> None:
        self.cfg = cfg
        self.low_eval = low_fidelity_eval
        self.high_eval = high_fidelity_eval or low_fidelity_eval
        self.pop_size = pop_size
        self.n_islands = max(2, n_islands)
        self._cache_low: Dict[Tuple[Tuple[str, float], ...], Dict[str, float]] = {}
        self._cache_high: Dict[Tuple[Tuple[str, float], ...], Dict[str, float]] = {}

    def _key(self, ind: Dict[str, Any]) -> Tuple[Tuple[str, float], ...]:
        fixed = repair(ind)
        out: List[Tuple[str, float]] = []
        for k in sorted(fixed.keys()):
            v = fixed[k]
            if isinstance(v, (int, float)):
                out.append((k, float(round(float(v), 6))))
        return tuple(out)

    def _random_individual(self) -> Dict[str, Any]:
        fl = self.cfg.get("fl", {}) if isinstance(self.cfg.get("fl", {}), dict) else {}
        num_clients = int(fl.get("num_clients", 10))
        ind: Dict[str, Any] = {
            "energy_w": float(np.random.uniform(0.05, 0.55)),
            "channel_w": float(np.random.uniform(0.05, 0.55)),
            "data_w": float(np.random.uniform(0.05, 0.55)),
            "fair_w": float(np.random.uniform(0.05, 0.55)),
            "selection_top_k": int(np.random.randint(1, max(2, num_clients + 1))),
            "bandwidth_alloc_factor": float(np.random.uniform(0.6, 1.4)),
            "staleness_alpha": float(np.random.uniform(0.5, 2.2)),
            "bridge_to_async": float(np.random.uniform(0.55, 0.80)),
            "bridge_to_semi_sync": float(np.random.uniform(0.20, 0.50)),
            "compression_ratio": float(np.random.uniform(0.45, 1.0)),
        }
        return repair(ind)

    def _with_penalty(self, m: Dict[str, float]) -> Dict[str, float]:
        pen = penalty(self.cfg, m)
        out = dict(m)
        out["time"] = out.get("time", 0.0) + 0.5 * pen
        out["energy"] = out.get("energy", 0.0) + pen
        out["comm_cost"] = out.get("comm_cost", out.get("time", 0.0)) + 0.25 * pen
        return out

    def _low_fidelity_metrics(self, ind: Dict[str, Any]) -> Dict[str, float]:
        k = self._key(ind)
        if k in self._cache_low:
            return dict(self._cache_low[k])
        m = evaluate_solution(self.low_eval, repair(ind))
        out = self._with_penalty(m)
        self._cache_low[k] = dict(out)
        return out

    def _high_fidelity_metrics(self, ind: Dict[str, Any]) -> Dict[str, float]:
        k = self._key(ind)
        if k in self._cache_high:
            return dict(self._cache_high[k])
        m = evaluate_solution(self.high_eval, repair(ind))
        out = self._with_penalty(m)
        self._cache_high[k] = dict(out)
        return out

    def _run_nsga3_island(self, generations: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        init_pop = [self._random_individual() for _ in range(self.pop_size)]
        opt = NSGA3(self.cfg, self.low_eval, pop_size=self.pop_size)
        return opt.run(generations=generations, init_pop=init_pop)

    def _run_moead_island(
        self,
        generations: int,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        opt = MOEAD(self.cfg, self.low_eval, pop_size=self.pop_size)
        return opt.run(generations=generations, init_pop=init_pop)

    def _feasible_local_search(
        self,
        elites: List[Dict[str, Any]],
        elite_metrics: List[Dict[str, float]],
        n_neighbors: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate neighbors near feasibility boundaries for higher-quality feasible solutions."""
        neighbors: List[Dict[str, Any]] = []
        controller = self.cfg.get("controller", {}) if isinstance(self.cfg.get("controller", {}), dict) else {}
        bridge_inv = controller.get("bridge_invariants", {}) if isinstance(controller.get("bridge_invariants", {}), dict) else {}
        e_budget = float(bridge_inv.get("energy_budget_round", 120.0))
        t_budget = float(bridge_inv.get("upload_time_budget_round", 3.0))

        for ind, m in zip(elites, elite_metrics):
            for _ in range(n_neighbors):
                c = repair(dict(ind))
                # Generic small perturbation around elite.
                for k in ["energy_w", "channel_w", "data_w", "fair_w"]:
                    c[k] = float(c.get(k, 0.25) + np.random.normal(0.0, 0.03))
                c["selection_top_k"] = int(c.get("selection_top_k", 1)) + int(np.random.randint(-1, 2))
                c["bridge_to_async"] = float(c.get("bridge_to_async", 0.58) + np.random.normal(0.0, 0.01))
                c["bridge_to_semi_sync"] = float(c.get("bridge_to_semi_sync", 0.42) + np.random.normal(0.0, 0.01))
                c["staleness_alpha"] = float(c.get("staleness_alpha", 1.0) + np.random.normal(0.0, 0.04))
                c["compression_ratio"] = float(c.get("compression_ratio", 1.0) + np.random.normal(0.0, 0.02))
                c["bandwidth_alloc_factor"] = float(c.get("bandwidth_alloc_factor", 1.0) + np.random.normal(0.0, 0.04))

                # Constraint-guided nudging.
                if float(m.get("energy", 0.0)) > e_budget:
                    c["compression_ratio"] = float(c.get("compression_ratio", 1.0) - 0.03)
                    c["selection_top_k"] = int(c.get("selection_top_k", 1) - 1)
                    c["bandwidth_alloc_factor"] = float(c.get("bandwidth_alloc_factor", 1.0) - 0.03)
                if float(m.get("time", 0.0)) > t_budget:
                    c["bandwidth_alloc_factor"] = float(c.get("bandwidth_alloc_factor", 1.0) + 0.04)
                    c["bridge_to_async"] = float(c.get("bridge_to_async", 0.58) + 0.01)
                if float(m.get("comm_cost", 0.0)) > float(m.get("time", 0.0)):
                    c["compression_ratio"] = float(c.get("compression_ratio", 1.0) - 0.03)
                if float(m.get("fairness", 0.0)) < 0.75:
                    c["fair_w"] = float(c.get("fair_w", 0.25) + 0.03)
                    c["selection_top_k"] = int(c.get("selection_top_k", 1) + 1)
                neighbors.append(repair(c))
        return neighbors

    def run(self, generations: int = 8) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        # 1) Stage-1 island: NSGA-III for distributed exploration.
        nsga_pop, nsga_metrics = self._run_nsga3_island(generations=max(2, generations // 2))

        # 2) Migration: move NSGA-III non-dominated elites to MOEA/D island.
        nd_idx = non_dominated_set(nsga_metrics)
        migrants = [nsga_pop[i] for i in nd_idx]
        while len(migrants) < min(6, self.pop_size // 2):
            migrants.append(self._random_individual())

        # 3) Stage-2 island: MOEA/D for neighborhood convergence.
        moead_init: List[Dict[str, Any]] = list(migrants)
        while len(moead_init) < self.pop_size:
            moead_init.append(self._random_individual())
        moead_pop, moead_metrics = self._run_moead_island(
            generations=max(2, generations - generations // 2),
            init_pop=moead_init,
        )

        # 4) Island merge and low-fidelity Pareto filtering.
        all_pop = nsga_pop + moead_pop
        all_low_metrics = nsga_metrics + moead_metrics
        nd_idx_all = non_dominated_set(all_low_metrics)
        elite_pop = [all_pop[i] for i in nd_idx_all]
        elite_low_metrics = [all_low_metrics[i] for i in nd_idx_all]

        # 5) Feasible-region local search around elites.
        neighbors = self._feasible_local_search(elite_pop, elite_low_metrics, n_neighbors=2)

        # 6) Multi-fidelity final scoring: high-fidelity for elites + neighbors.
        final_pop: List[Dict[str, Any]] = []
        final_metrics: List[Dict[str, float]] = []
        for ind in elite_pop + neighbors:
            m = self._high_fidelity_metrics(ind)
            final_pop.append(repair(ind))
            final_metrics.append(m)

        nd_idx_final = non_dominated_set(final_metrics)
        pareto_pop = [final_pop[i] for i in nd_idx_final]
        pareto_metrics = [final_metrics[i] for i in nd_idx_final]
        return pareto_pop, pareto_metrics
