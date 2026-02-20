from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .constraints import penalty, repair


class MogaFLProblem(Problem):
    """pymoo Problem 封装：多目标优化 MOGA-FL 调度参数。

    变量包含：
    - energy_w, channel_w, data_w, fair_w: 多指标评分权重；
    - bwcost_w: 带宽成本权重（通常为负）；
    - selection_top_k: 每轮选中客户端数量（整数，将通过 repair 取整）；
    - hysteresis: 调度防抖系数；
    - staleness_alpha: FedBuff 陈旧度加权指数。
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        eval_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    ) -> None:
        self.cfg = cfg
        self.eval_fn = eval_fn
        num_clients = cfg["clients"]["num_clients"]

        xl = np.array(
            [
                0.0,  # energy_w
                0.0,  # channel_w
                0.0,  # data_w
                0.0,  # fair_w
                -0.3,  # bwcost_w
                1.0,  # selection_top_k (下界)
                0.0,  # hysteresis
                0.5,  # staleness_alpha
            ],
            dtype=float,
        )
        xu = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                -0.05,
                float(num_clients),
                0.2,
                2.0,
            ],
            dtype=float,
        )

        super().__init__(n_var=8, n_obj=4, n_constr=0, xl=xl, xu=xu)

    # ---------------- 工具方法 -----------------
    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """将连续向量解码为参数字典，并做一次 repair。"""

        vec = np.asarray(x, dtype=float).ravel()
        params = {
            "energy_w": float(vec[0]),
            "channel_w": float(vec[1]),
            "data_w": float(vec[2]),
            "fair_w": float(vec[3]),
            "bwcost_w": float(vec[4]),
            "selection_top_k": float(vec[5]),
            "hysteresis": float(vec[6]),
            "staleness_alpha": float(vec[7]),
        }
        return repair(params)

    # ---------------- 目标函数 -----------------
    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore[override]
        X = np.atleast_2d(X)
        F_list: List[np.ndarray] = []
        for x in X:
            params = self.decode(x)
            metrics = self.eval_fn(params)
            pen = penalty(self.cfg, metrics)
            energy_p = float(metrics["energy"]) + float(pen)
            f1 = -float(metrics["acc"])        # maximize acc
            f2 = -float(metrics["fairness"])   # maximize fairness
            f3 = float(metrics["time"])        # minimize time
            f4 = energy_p                       # minimize energy (with penalty)
            F_list.append(np.array([f1, f2, f3, f4], dtype=float))
        out["F"] = np.vstack(F_list)


class MogaFLOptimizer:
    """基于 pymoo-NSGA2 的多目标 GA 封装。"""

    def __init__(
        self,
        cfg: Dict[str, Any],
        low_eval: Callable[[Dict[str, Any]], Dict[str, float]],
        high_eval: Callable[[Dict[str, Any]], Dict[str, float]] | None = None,
        pop_size: int = 16,
    ) -> None:
        self.cfg = cfg
        self.low_eval = low_eval
        self.high_eval = high_eval or low_eval
        self.pop_size = int(pop_size)

    def run(self, generations: int = 6) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        problem = MogaFLProblem(self.cfg, self.low_eval)
        algorithm = NSGA2(pop_size=self.pop_size)
        termination = get_termination("n_gen", generations)

        res = minimize(problem, algorithm, termination, verbose=False)

        pop_params: List[Dict[str, Any]] = [problem.decode(x) for x in res.X]
        metrics: List[Dict[str, float]] = [self.high_eval(p) for p in pop_params]
        return pop_params, metrics
