from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .constraints import penalty, repair
from .nsga3 import NSGA3
from .objectives import evaluate_solution
from .moead import MOEAD
from .pareto import non_dominated_set


class MOGAFLController:
    """MOGA-FL：多目标遗传联邦调度控制器。

    设计目标：
    - 同时利用 **NSGA-III** 与 **MOEA/D** 两种多目标进化策略；
    - 通过“岛屿模型”（不同子种群、不同算法）和简单迁移，提高搜索效率；
    - 在精英解周围做局部搜索（Memetic/微调），专门微调评分权重与 Top-K 等关键参数；
    - 支持多保真评估：先用低保真（少轮训练）粗筛，再对少量精英解做高保真复评；
    - 输出一组 Pareto 候选解，并交给上层脚本进行偏好化部署（时间/公平/能耗优先）。

    个体编码与 HybridWirelessStrategy 中可控参数保持一致，仅包含：
    - ``energy_w, channel_w, data_w, fair_w``：调度评分权重
    - ``selection_top_k``：每轮选多少客户端
    - ``staleness_alpha``：FedBuff 陈旧度加权指数
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
        # 这里的 eval 函数被视为“仿真 runner”，其返回值会再经过
        # :func:`evaluate_solution` 做键名统一。
        self.low_eval = low_fidelity_eval
        self.high_eval = high_fidelity_eval or low_fidelity_eval
        self.pop_size = pop_size
        # 至少两个岛屿：一个 NSGA-III，一个 MOEA/D
        self.n_islands = max(2, n_islands)

    # ------------------------ 个体与评估 ------------------------
    def _random_individual(self) -> Dict[str, Any]:
        """采样一个参数个体：评分权重、Top-K 与陈旧度权重等。"""
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

    def _low_fidelity_metrics(self, ind: Dict[str, Any]) -> Dict[str, float]:
        ind_fixed = repair(ind)
        m = evaluate_solution(self.low_eval, ind_fixed)
        pen = penalty(self.cfg, m)
        out = dict(m)
        out["energy"] = out.get("energy", 0.0) + pen
        return out

    def _high_fidelity_metrics(self, ind: Dict[str, Any]) -> Dict[str, float]:
        ind_fixed = repair(ind)
        m = evaluate_solution(self.high_eval, ind_fixed)
        pen = penalty(self.cfg, m)
        out = dict(m)
        out["energy"] = out.get("energy", 0.0) + pen
        return out

    # ------------------------ 岛屿 + 迁移 ------------------------
    def _run_nsga3_island(self, generations: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        init_pop = [self._random_individual() for _ in range(self.pop_size)]
        opt = NSGA3(self.cfg, self.low_eval, pop_size=self.pop_size)
        pop, metrics = opt.run(generations=generations, init_pop=init_pop)
        return pop, metrics

    def _run_moead_island(
        self,
        generations: int,
        init_pop: List[Dict[str, Any]] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        opt = MOEAD(self.cfg, self.low_eval, pop_size=self.pop_size)
        pop, metrics = opt.run(generations=generations, init_pop=init_pop)
        return pop, metrics

    # ------------------------ 局部搜索（Memetic 微调） ------------------------
    def _local_search(self, elites: List[Dict[str, Any]], n_neighbors: int = 2) -> List[Dict[str, Any]]:
        """在精英个体附近做小范围扰动，微调权重、Top-K 与 staleness_alpha。"""
        neighbors: List[Dict[str, Any]] = []
        fl = self.cfg.get("fl", {}) if isinstance(self.cfg.get("fl", {}), dict) else {}
        num_clients = int(fl.get("num_clients", 10))

        for e in elites:
            for _ in range(n_neighbors):
                c: Dict[str, Any] = dict(e)
                # 小范围高斯扰动权重
                for k in ["energy_w", "channel_w", "data_w", "fair_w"]:
                    c[k] = float(c.get(k, 0.25) + np.random.normal(0.0, 0.05))
                # 归一化正向权重，保持总和为 1 左右
                pos_keys = ["energy_w", "channel_w", "data_w", "fair_w"]
                s = sum(max(1e-6, float(c[k])) for k in pos_keys)
                for k in pos_keys:
                    c[k] = float(max(1e-6, float(c[k])) / s)
                # Top-K 轻微调整
                k_old = int(c.get("selection_top_k", 1))
                k_new = int(np.clip(k_old + np.random.randint(-1, 2), 1, max(2, num_clients)))
                c["selection_top_k"] = k_new
                # staleness_alpha 小范围扰动
                sa_old = float(c.get("staleness_alpha", 1.0))
                sa_new = float(np.clip(sa_old + np.random.normal(0.0, 0.05), 0.5, 2.0))
                c["staleness_alpha"] = sa_new
                neighbors.append(c)
        return neighbors

    # ------------------------ 主流程：MOGA-FL ------------------------
    def run(self, generations: int = 8) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
        """运行改进型 MOGA-FL，返回一组 Pareto 候选解及其高保真指标。

        为了适配低算力环境，这里默认 generations 较小，且内部只做少量局部搜索
        与高保真评估。
        """
        # 1）岛屿1：NSGA-III 粗搜索
        nsga_pop, nsga_metrics = self._run_nsga3_island(generations=max(2, generations // 2))

        # 2）根据 NSGA-III 结果，挑选部分精英个体迁移到 MOEA/D 岛屿
        nd_idx = non_dominated_set(nsga_metrics)
        elites = [nsga_pop[i] for i in nd_idx]
        # 若精英数量不足，则补若干随机个体
        while len(elites) < min(4, self.pop_size // 2):
            elites.append(self._random_individual())

        # 3）岛屿2：MOEA/D，从精英 + 随机个体出发继续优化
        moead_init_pop: List[Dict[str, Any]] = list(elites)
        while len(moead_init_pop) < self.pop_size:
            moead_init_pop.append(self._random_individual())
        moead_pop, moead_metrics = self._run_moead_island(
            generations=max(2, generations - generations // 2), init_pop=moead_init_pop
        )

        # 4）岛屿间“迁移”：简单地将两岛的解合并，形成一个大候选集
        all_pop = nsga_pop + moead_pop
        all_metrics_low = nsga_metrics + moead_metrics

        # 5）在低保真候选的非支配前沿上做局部搜索（Memetic）
        nd_idx_all = non_dominated_set(all_metrics_low)
        elite_pop = [all_pop[i] for i in nd_idx_all]
        neighbors = self._local_search(elite_pop, n_neighbors=2)

        # 6）对“精英 + 邻域”做高保真评估
        final_pop: List[Dict[str, Any]] = []
        final_metrics: List[Dict[str, float]] = []
        for ind in elite_pop + neighbors:
            m = self._high_fidelity_metrics(ind)
            final_pop.append(ind)
            final_metrics.append(m)

        # 再次取高保真结果的非支配集，作为最终 MOGA-FL 输出
        nd_idx_final = non_dominated_set(final_metrics)
        pareto_pop = [final_pop[i] for i in nd_idx_final]
        pareto_metrics = [final_metrics[i] for i in nd_idx_final]
        return pareto_pop, pareto_metrics
