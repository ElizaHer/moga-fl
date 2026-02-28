from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .constraints import penalty, repair
from .objectives import evaluate_solution


class MOEAD:
    """Simplified MOEA/D style optimizer.

    在本工程中，它也是 MOGA-FL 的一个子算法，用于加速多目标搜索的
    收敛。个体编码与 Flower 异构 FL 的 GA 搜索空间一致：

    - ``energy_w, channel_w, data_w, fair_w``
    - ``selection_top_k``
    - ``staleness_alpha``
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        sim_runner: Callable[[Dict[str, Any]], Dict[str, float]],
        pop_size: int = 20,
    ) -> None:
        self.cfg = cfg
        self.sim_runner = sim_runner
        self.pop_size = pop_size
        self.weights = self._init_weights(pop_size)

    def _init_weights(self, n: int) -> List[np.ndarray]:
        w: List[np.ndarray] = []
        for _ in range(n):
            a = np.random.rand(4)
            a = a / (a.sum() + 1e-9)
            w.append(a)
        return w

    def _random_individual(self) -> Dict[str, Any]:
        fl = self.cfg.get("fl", {}) if isinstance(self.cfg.get("fl", {}), dict) else {}
        num_clients = int(fl.get("num_clients", 10))
        return {
            "energy_w": float(np.random.uniform(0.1, 0.4)),
            "channel_w": float(np.random.uniform(0.1, 0.4)),
            "data_w": float(np.random.uniform(0.1, 0.4)),
            "fair_w": float(np.random.uniform(0.1, 0.4)),
            "selection_top_k": int(np.random.randint(1, max(2, num_clients // 2))),
            "staleness_alpha": float(np.random.uniform(0.8, 1.4)),
        }

    def scalarize(self, metrics: Dict[str, float], w: np.ndarray) -> float:
        # Normalize crudely into a single scalar: maximize acc/fairness, minimize time/energy
        acc = float(metrics["acc"])
        fair = float(metrics["fairness"])
        time = float(metrics["time"])
        energy = float(metrics["energy"])
        return float(-(w[0] * acc + w[1] * fair) + (w[2] * time + w[3] * energy))

    def _evaluate(self, p: Dict[str, Any]) -> Dict[str, float]:
        p_fixed = repair(p)
        metrics = evaluate_solution(self.sim_runner, p_fixed)
        pen = penalty(self.cfg, metrics)
        out = dict(metrics)
        out["energy"] = out.get("energy", 0.0) + pen
        return out

    def run(
        self,
        generations: int = 10,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        pop: List[Dict[str, Any]] = [] if init_pop is None else list(init_pop)
        if not pop:
            pop = [self._random_individual() for _ in range(self.pop_size)]

        metrics: List[Dict[str, float]] = [self._evaluate(p) for p in pop]
        for _ in range(generations):
            for i in range(self.pop_size):
                a_idx, b_idx = np.random.choice(self.pop_size, 2, replace=False)
                child = dict(pop[i])
                for k in child:
                    if k == "selection_top_k":
                        val = 0.5 * float(pop[a_idx][k]) + 0.5 * float(pop[b_idx][k])
                        val += np.random.normal(0.0, 0.5)
                        child[k] = val
                    else:
                        child[k] = (
                            float(pop[a_idx][k]) + float(pop[b_idx][k])
                        ) / 2.0 + np.random.normal(0.0, 0.01)
                m_child = self._evaluate(child)
                if self.scalarize(m_child, self.weights[i]) < self.scalarize(metrics[i], self.weights[i]):
                    pop[i] = child
                    metrics[i] = m_child
        return pop, metrics
