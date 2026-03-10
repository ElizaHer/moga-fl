from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from .constraints import penalty, repair
from .objectives import evaluate_solution
from .pareto import dominates, non_dominated_set


def _to_min_objective_vector(m: Dict[str, float]) -> np.ndarray:
    """Convert mixed objectives to minimization vector."""
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


class NSGA3:
    """NSGA-III style optimizer with reference-direction niching.

    Decision variables:
    - scheduler weights: ``energy_w, channel_w, data_w, fair_w``
    - ``selection_top_k``
    - ``bandwidth_alloc_factor``
    - ``staleness_alpha``
    - ``bridge_to_async``, ``bridge_to_semi_sync``
    - ``compression_ratio``
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
        self.ref_dirs = self._init_reference_directions(pop_size, n_obj=5)

    def _init_reference_directions(self, n: int, n_obj: int) -> np.ndarray:
        # Lightweight simplex sampling as reference directions.
        dirs = np.random.rand(max(4 * n, n), n_obj)
        dirs = dirs / np.clip(dirs.sum(axis=1, keepdims=True), 1e-12, None)
        return dirs

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

    def init_pop(self) -> List[Dict[str, Any]]:
        return [self._random_individual() for _ in range(self.pop_size)]

    def crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        c: Dict[str, Any] = {}
        for k in a:
            if k == "selection_top_k":
                c[k] = int(round(0.5 * float(a[k]) + 0.5 * float(b[k])))
            else:
                c[k] = 0.5 * float(a[k]) + 0.5 * float(b[k])
        return repair(c)

    def mutate(self, p: Dict[str, Any]) -> Dict[str, Any]:
        q: Dict[str, Any] = dict(p)
        for k in ["energy_w", "channel_w", "data_w", "fair_w"]:
            q[k] = float(q.get(k, 0.25) + np.random.normal(0.0, 0.03))
        q["selection_top_k"] = int(q.get("selection_top_k", 1)) + int(np.random.randint(-1, 2))
        q["bandwidth_alloc_factor"] = float(q.get("bandwidth_alloc_factor", 1.0) + np.random.normal(0.0, 0.05))
        q["staleness_alpha"] = float(q.get("staleness_alpha", 1.0) + np.random.normal(0.0, 0.06))
        q["bridge_to_async"] = float(q.get("bridge_to_async", 0.58) + np.random.normal(0.0, 0.02))
        q["bridge_to_semi_sync"] = float(q.get("bridge_to_semi_sync", 0.42) + np.random.normal(0.0, 0.02))
        q["compression_ratio"] = float(q.get("compression_ratio", 1.0) + np.random.normal(0.0, 0.03))
        return repair(q)

    def _evaluate(self, p: Dict[str, Any]) -> Dict[str, float]:
        p_fixed = repair(p)
        metrics = evaluate_solution(self.sim_runner, p_fixed)
        pen = penalty(self.cfg, metrics)
        out = dict(metrics)
        # Apply soft penalty on minimization objectives.
        out["time"] = out.get("time", 0.0) + 0.5 * pen
        out["energy"] = out.get("energy", 0.0) + pen
        out["comm_cost"] = out.get("comm_cost", out.get("time", 0.0)) + 0.25 * pen
        return out

    def _fast_non_dominated_sort(self, metrics: Sequence[Dict[str, float]]) -> List[List[int]]:
        n = len(metrics)
        dominates_to: List[List[int]] = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts: List[List[int]] = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if dominates(metrics[i], metrics[j]):
                    dominates_to[i].append(j)
                elif dominates(metrics[j], metrics[i]):
                    dominated_count[i] += 1
            if dominated_count[i] == 0:
                fronts[0].append(i)

        f = 0
        while f < len(fronts) and fronts[f]:
            nxt: List[int] = []
            for i in fronts[f]:
                for j in dominates_to[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        nxt.append(j)
            if nxt:
                fronts.append(nxt)
            f += 1
        return fronts

    def _normalize_objectives(self, vectors: np.ndarray) -> np.ndarray:
        vmin = vectors.min(axis=0)
        vmax = vectors.max(axis=0)
        return (vectors - vmin) / np.clip(vmax - vmin, 1e-12, None)

    def _associate_to_ref_dirs(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # cosine distance to reference direction on normalized minimization vectors
        x = self._normalize_objectives(vectors)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x_unit = x / np.clip(x_norm, 1e-12, None)
        r = self.ref_dirs / np.clip(np.linalg.norm(self.ref_dirs, axis=1, keepdims=True), 1e-12, None)
        cos_sim = x_unit @ r.T
        best_ref = np.argmax(cos_sim, axis=1)
        # perpendicular distance proxy
        dist = np.sqrt(np.clip(1.0 - np.max(cos_sim, axis=1) ** 2, 0.0, 1.0))
        return best_ref, dist

    def _niching_select(self, idxs: List[int], metrics: Sequence[Dict[str, float]], remain: int) -> List[int]:
        if remain <= 0 or not idxs:
            return []
        vecs = np.asarray([_to_min_objective_vector(metrics[i]) for i in idxs], dtype=np.float64)
        assoc, dist = self._associate_to_ref_dirs(vecs)
        selected: List[int] = []
        niche_count = np.zeros(len(self.ref_dirs), dtype=np.int32)
        available = set(range(len(idxs)))
        while len(selected) < remain and available:
            ref_order = np.argsort(niche_count)
            picked = False
            for ref_id in ref_order:
                cands = [j for j in available if assoc[j] == ref_id]
                if not cands:
                    continue
                best_local = min(cands, key=lambda j: dist[j])
                selected.append(idxs[best_local])
                available.remove(best_local)
                niche_count[ref_id] += 1
                picked = True
                break
            if not picked:
                break
        if len(selected) < remain:
            fallback = [idxs[j] for j in sorted(list(available), key=lambda j: dist[j])]
            selected.extend(fallback[: remain - len(selected)])
        return selected[:remain]

    def _next_generation(
        self,
        pop: List[Dict[str, Any]],
        metrics: List[Dict[str, float]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        fronts = self._fast_non_dominated_sort(metrics)
        chosen: List[int] = []
        for front in fronts:
            if len(chosen) + len(front) <= self.pop_size:
                chosen.extend(front)
            else:
                need = self.pop_size - len(chosen)
                chosen.extend(self._niching_select(front, metrics, need))
                break
        return [pop[i] for i in chosen], [metrics[i] for i in chosen]

    def run(
        self,
        generations: int = 10,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        pop: List[Dict[str, Any]] = list(init_pop) if init_pop is not None else self.init_pop()
        if not pop:
            pop = self.init_pop()

        metrics = [self._evaluate(p) for p in pop]
        pop, metrics = self._next_generation(pop, metrics)
        for _ in range(generations):
            offspring: List[Dict[str, Any]] = []
            while len(offspring) < self.pop_size:
                a_idx, b_idx = np.random.choice(len(pop), 2, replace=False)
                child = self.crossover(pop[a_idx], pop[b_idx])
                child = self.mutate(child)
                offspring.append(child)
            off_metrics = [self._evaluate(c) for c in offspring]
            mix_pop = pop + offspring
            mix_metrics = metrics + off_metrics
            pop, metrics = self._next_generation(mix_pop, mix_metrics)

        nd_idx = non_dominated_set(metrics)
        return [pop[i] for i in nd_idx], [metrics[i] for i in nd_idx]
