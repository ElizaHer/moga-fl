import numpy as np
from typing import List, Dict, Any, Callable

class MOEAD:
    """Simplified MOEA/D style optimizer.

    在本工程中，它也是 MOGA-FL 的一个子算法，用于加速收敛
    （问题1：多目标遗传优化的基础子算法之一）。"""

    def __init__(self, eval_fn: Callable[[Dict[str, Any]], Dict[str, float]], pop_size=20):
        self.eval_fn = eval_fn
        self.pop_size = pop_size
        self.weights = self._init_weights(pop_size)

    def _init_weights(self, n):
        w = []
        for i in range(n):
            a = np.random.rand(4)
            a = a / (a.sum()+1e-9)
            w.append(a)
        return w

    def scalarize(self, metrics: Dict[str, float], w):
        # Normalize crude
        acc = metrics['acc']
        fair = metrics['fairness']
        time = metrics['time']
        energy = metrics['energy']
        return -(w[0]*acc + w[1]*fair) + (w[2]*time + w[3]*energy)

    def run(self, generations=10, init_pop: List[Dict[str, Any]] | None = None):
        pop = [] if init_pop is None else list(init_pop)
        if not pop:
            for _ in range(self.pop_size):
                p = {
                    'energy_w': np.random.uniform(0.1, 0.4),
                    'channel_w': np.random.uniform(0.1, 0.4),
                    'data_w': np.random.uniform(0.1, 0.4),
                    'fair_w': np.random.uniform(0.1, 0.4),
                    'bwcost_w': -np.random.uniform(0.05, 0.2),
                    'selection_top_k': np.random.randint(2, 8),
                    'hysteresis': np.random.uniform(0.02, 0.1),
                    'staleness_alpha': np.random.uniform(0.8, 1.4),
                }
                pop.append(p)
        metrics = [self.eval_fn(p) for p in pop]
        for g in range(generations):
            for i in range(self.pop_size):
                a, b = np.random.choice(self.pop_size, 2, replace=False)
                child = pop[i].copy()
                for k in child:
                    child[k] = (pop[a][k] + pop[b][k]) / 2.0 + np.random.normal(0,0.01)
                m_child = self.eval_fn(child)
                if self.scalarize(m_child, self.weights[i]) < self.scalarize(metrics[i], self.weights[i]):
                    pop[i] = child
                    metrics[i] = m_child
        return pop, metrics
