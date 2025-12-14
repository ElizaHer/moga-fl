import numpy as np
from typing import List, Dict, Any, Callable
from .pareto import non_dominated_set
from .constraints import penalty, repair

class NSGA3:
    """Simplified NSGA-III style optimizer.

    在本工程中，它既可以单独使用，也会被 MOGA-FL 控制器调用
    （问题1：多目标遗传优化的基础子算法之一）。"""

    def __init__(self, cfg: Dict[str, Any], eval_fn: Callable[[Dict[str, Any]], Dict[str, float]], pop_size=20):
        self.cfg = cfg
        self.eval_fn = eval_fn
        self.pop_size = pop_size

    def init_pop(self) -> List[Dict[str, Any]]:
        pop = []
        for _ in range(self.pop_size):
            p = {
                'energy_w': np.random.uniform(0.1, 0.4),
                'channel_w': np.random.uniform(0.1, 0.4),
                'data_w': np.random.uniform(0.1, 0.4),
                'fair_w': np.random.uniform(0.1, 0.4),
                'bwcost_w': -np.random.uniform(0.05, 0.2),
                'selection_top_k': np.random.randint(2, max(3, self.cfg['clients']['num_clients']//2)),
                'hysteresis': np.random.uniform(0.02, 0.1),
                'staleness_alpha': np.random.uniform(0.8, 1.4),
            }
            pop.append(p)
        return pop

    def crossover(self, a, b):
        c = {}
        for k in a:
            c[k] = 0.5*a[k] + 0.5*b[k]
        return c

    def mutate(self, p):
        for k in p:
            p[k] += np.random.normal(0, 0.02)
        return p

    def evaluate(self, p):
        p = repair(p)
        metrics = self.eval_fn(p)
        pen = penalty(self.cfg, metrics)
        metrics['acc'] -= 0.0*pen
        metrics['energy'] += pen
        return metrics

    def run(self, generations=10, init_pop: List[Dict[str, Any]] | None = None):
        """Run a small NSGA-III loop.

        参数
        ------
        generations: 迭代轮数（通常较小，因为外层 MOGA-FL 还会做多轮调用）。
        init_pop: 若给定，则从外部提供的初始种群开始（用于岛屿模型中的迁移）。
        """
        pop = init_pop if init_pop is not None else self.init_pop()
        sols = [self.evaluate(p) for p in pop]
        for g in range(generations):
            # Selection: keep non-dominated + random
            nd_idx = non_dominated_set(sols)
            new_pop = [pop[i] for i in nd_idx]
            while len(new_pop) < self.pop_size:
                a, b = np.random.choice(len(pop), 2, replace=False)
                child = self.crossover(pop[a], pop[b])
                child = self.mutate(child)
                new_pop.append(child)
            pop = new_pop
            sols = [self.evaluate(p) for p in pop]
        # Return final non-dominated set
        nd_idx = non_dominated_set(sols)
        return [pop[i] for i in nd_idx], [sols[i] for i in nd_idx]
