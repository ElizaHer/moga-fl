from __future__ import annotations

from typing import Tuple, List, Dict, Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .problem import FLHyperparamProblem


def run_nsga2(
    pop_size: int = 12,
    n_gen: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用 NSGA-II 对联邦学习超参数做一个小规模多目标搜索。

    返回：
    - X: (n_solutions, n_var) 的决策变量数组；
    - F: (n_solutions, n_obj) 的目标值数组。
    """

    problem = FLHyperparamProblem()
    algorithm = NSGA2(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=False,
    )
    return res.X, res.F
