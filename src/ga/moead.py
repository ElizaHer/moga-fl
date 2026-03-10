from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .constraints import penalty, repair
from .objectives import evaluate_solution


def _to_min_objective_vector(m: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            -float(m["acc"]),
            float(m["time"]),
            float(m["energy"]),
            float(m.get("comm_cost", m["time"])),
            -float(m["fairness"]),
        ],
        dtype=np.float64,
    )


class MOEAD:
    """MOEA/D style optimizer for MOGA-FL with 5 objectives."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        sim_runner: Callable[[Dict[str, Any]], Dict[str, float]],
        pop_size: int = 20,
    ) -> None:
        self.cfg = cfg
        self.sim_runner = sim_runner
        self.pop_size = pop_size
        self.n_obj = 5
        self.weights = self._init_weights(pop_size, self.n_obj)
        self.neighbors = self._init_neighbors(k=max(3, pop_size // 5))

    def _init_weights(self, n: int, n_obj: int) -> np.ndarray:
        w = np.random.rand(n, n_obj)
        return w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)

    def _init_neighbors(self, k: int) -> List[np.ndarray]:
        dist = np.linalg.norm(self.weights[:, None, :] - self.weights[None, :, :], axis=2)
        return [np.argsort(dist[i])[:k] for i in range(self.pop_size)]

    def _random_individual(self) -> Dict[str, Any]:
        fl = self.cfg.get("fl", {}) if isinstance(self.cfg.get("fl", {}), dict) else {}
        num_clients = int(fl.get("num_clients", 10))
        ind = {
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

    def _evaluate(self, p: Dict[str, Any]) -> Dict[str, float]:
        p_fixed = repair(p)
        metrics = evaluate_solution(self.sim_runner, p_fixed)
        pen = penalty(self.cfg, metrics)
        out = dict(metrics)
        out["time"] = out.get("time", 0.0) + 0.5 * pen
        out["energy"] = out.get("energy", 0.0) + pen
        out["comm_cost"] = out.get("comm_cost", out.get("time", 0.0)) + 0.25 * pen
        return out

    def _scalarize_tchebycheff(self, vec: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
        return float(np.max(w * np.abs(vec - z)))

    def _mate_and_mutate(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        c: Dict[str, Any] = {}
        for k in a:
            if k == "selection_top_k":
                c[k] = int(round(0.5 * float(a[k]) + 0.5 * float(b[k]) + np.random.normal(0.0, 0.6)))
            else:
                c[k] = 0.5 * float(a[k]) + 0.5 * float(b[k]) + np.random.normal(0.0, 0.03)
        return repair(c)

    def run(
        self,
        generations: int = 10,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        pop: List[Dict[str, Any]] = list(init_pop) if init_pop is not None else []
        if not pop:
            pop = [self._random_individual() for _ in range(self.pop_size)]
        if len(pop) < self.pop_size:
            pop.extend([self._random_individual() for _ in range(self.pop_size - len(pop))])
        pop = pop[: self.pop_size]

        metrics = [self._evaluate(p) for p in pop]
        obj = np.asarray([_to_min_objective_vector(m) for m in metrics], dtype=np.float64)
        z = obj.min(axis=0)

        for _ in range(generations):
            for i in range(self.pop_size):
                nb = self.neighbors[i]
                a_idx, b_idx = np.random.choice(nb, 2, replace=False)
                child = self._mate_and_mutate(pop[int(a_idx)], pop[int(b_idx)])
                m_child = self._evaluate(child)
                v_child = _to_min_objective_vector(m_child)
                z = np.minimum(z, v_child)
                for j in nb:
                    j_int = int(j)
                    v_j = _to_min_objective_vector(metrics[j_int])
                    if self._scalarize_tchebycheff(v_child, self.weights[j_int], z) <= self._scalarize_tchebycheff(
                        v_j, self.weights[j_int], z
                    ):
                        pop[j_int] = dict(child)
                        metrics[j_int] = dict(m_child)
        return pop, metrics
