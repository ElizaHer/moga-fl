from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from .constraints import penalty, repair
from .objectives import evaluate_solution
from .pareto import non_dominated_set


class NSGA3:
    """Simplified NSGA-III style optimizer.

    在本工程中，它既可以单独使用，也会被 MOGA-FL 控制器调用，作为
    多目标搜索的基础子算法之一。个体编码与 Flower 异构 FL 的可调参数
    对齐，仅包含：

    - ``energy_w, channel_w, data_w, fair_w``：调度评分权重
    - ``selection_top_k``：每轮选多少客户端
    - ``staleness_alpha``：FedBuff 陈旧度加权的指数
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

    # ---------------------------- 初始化与个体 ----------------------------
    def _random_individual(self) -> Dict[str, Any]:
        fl = self.cfg.get("fl", {}) if isinstance(self.cfg.get("fl", {}), dict) else {}
        num_clients = int(fl.get("num_clients", 10))
        ind: Dict[str, Any] = {
            "energy_w": float(np.random.uniform(0.1, 0.4)),
            "channel_w": float(np.random.uniform(0.1, 0.4)),
            "data_w": float(np.random.uniform(0.1, 0.4)),
            "fair_w": float(np.random.uniform(0.1, 0.4)),
            "selection_top_k": int(np.random.randint(1, max(2, num_clients // 2))),
            "staleness_alpha": float(np.random.uniform(0.8, 1.4)),
        }
        return ind

    def init_pop(self) -> List[Dict[str, Any]]:
        return [self._random_individual() for _ in range(self.pop_size)]

    # ---------------------------- 变异与交叉 ----------------------------
    def crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        c: Dict[str, Any] = {}
        for k in a:
            if k == "selection_top_k":
                c[k] = 0.5 * float(a[k]) + 0.5 * float(b[k])
            else:
                c[k] = 0.5 * float(a[k]) + 0.5 * float(b[k])
        return c

    def mutate(self, p: Dict[str, Any]) -> Dict[str, Any]:
        q: Dict[str, Any] = dict(p)
        for k in ["energy_w", "channel_w", "data_w", "fair_w"]:
            q[k] = float(q.get(k, 0.25) + np.random.normal(0.0, 0.02))
        # Top-K 做离散步长扰动
        k_old = int(q.get("selection_top_k", 1))
        k_new = k_old + int(np.random.randint(-1, 2))
        q["selection_top_k"] = k_new
        # staleness_alpha 做小范围高斯扰动
        q["staleness_alpha"] = float(q.get("staleness_alpha", 1.0) + np.random.normal(0.0, 0.05))
        return q

    # ---------------------------- 评估 ----------------------------
    def _evaluate(self, p: Dict[str, Any]) -> Dict[str, float]:
        p_fixed = repair(p)
        metrics = evaluate_solution(self.sim_runner, p_fixed)
        pen = penalty(self.cfg, metrics)
        out = dict(metrics)
        out["energy"] = out.get("energy", 0.0) + pen
        return out

    # ---------------------------- 主循环 ----------------------------
    def run(
        self,
        generations: int = 10,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """Run a small NSGA-III loop.

        参数
        ------
        generations:
            迭代轮数（通常较小，因为外层 MOGA-FL 还会做多轮调用）。
        init_pop:
            若给定，则从外部提供的初始种群开始（用于岛屿模型中的迁移）。
        """
        pop: List[Dict[str, Any]] = list(init_pop) if init_pop is not None else self.init_pop()
        if not pop:
            pop = self.init_pop()

        sols = [self._evaluate(p) for p in pop]
        for _ in range(generations):
            # Selection: keep non-dominated + offspring filling
            nd_idx = non_dominated_set(sols)
            new_pop = [pop[i] for i in nd_idx]
            while len(new_pop) < self.pop_size:
                a_idx, b_idx = np.random.choice(len(pop), 2, replace=False)
                child = self.crossover(pop[a_idx], pop[b_idx])
                child = self.mutate(child)
                new_pop.append(child)
            pop = new_pop
            sols = [self._evaluate(p) for p in pop]

        # Return final non-dominated set
        nd_idx = non_dominated_set(sols)
        return [pop[i] for i in nd_idx], [sols[i] for i in nd_idx]
