from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .constraints import repair
from .objectives import evaluate_solution
from .sim_runner_flower import make_flower_sim_runner


def _energy_budget(cfg: Dict[str, Any]) -> float:
    controller = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
    bridge_inv = controller.get("bridge_invariants", {}) if isinstance(controller.get("bridge_invariants", {}), dict) else {}
    if "energy_budget_round" in bridge_inv:
        try:
            return float(bridge_inv["energy_budget_round"])
        except (TypeError, ValueError):
            pass
    wireless = cfg.get("wireless", {}) if isinstance(cfg.get("wireless", {}), dict) else {}
    tx_power = float(wireless.get("tx_power_watts", 1.0))
    compute_power = float(wireless.get("compute_power_watts", 8.0))
    return float((tx_power + compute_power) * 10.0)


def run_pymoo_nsga3(
    cfg: Dict[str, Any],
    strategy_name: str,
    round_scale: float,
    pop_size: int,
    generations: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]]]:
    """Use pymoo NSGA-III on the same chromosome/objectives as MOGA-FL."""

    from pymoo.algorithms.moo.nsga3 import NSGA3 as PymooNSGA3
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.factory import get_reference_directions
    from pymoo.optimize import minimize

    sim_runner = make_flower_sim_runner(cfg, strategy_name=strategy_name, round_scale=round_scale)
    budget = _energy_budget(cfg)

    fl = cfg.get("fl", {}) if isinstance(cfg.get("fl", {}), dict) else {}
    num_clients = int(fl.get("num_clients", 10))

    class GAFLProblem(ElementwiseProblem):
        def __init__(self) -> None:
            xl = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.5, 0.50, 0.10, 0.40], dtype=float)
            xu = np.array([1.0, 1.0, 1.0, 1.0, float(num_clients), 1.4, 2.5, 0.90, 0.70, 1.00], dtype=float)
            super().__init__(n_var=10, n_obj=5, n_constr=1, xl=xl, xu=xu)

        def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore[override]
            params: Dict[str, Any] = {
                "energy_w": float(x[0]),
                "channel_w": float(x[1]),
                "data_w": float(x[2]),
                "fair_w": float(x[3]),
                "selection_top_k": int(round(float(x[4]))),
                "bandwidth_alloc_factor": float(x[5]),
                "staleness_alpha": float(x[6]),
                "bridge_to_async": float(x[7]),
                "bridge_to_semi_sync": float(x[8]),
                "compression_ratio": float(x[9]),
            }
            params = repair(params)
            metrics = evaluate_solution(sim_runner, params)

            acc = float(metrics["acc"])
            fairness = float(metrics["fairness"])
            time = float(metrics["time"])
            energy = float(metrics["energy"])
            comm_cost = float(metrics.get("comm_cost", time))

            out["F"] = np.array([-acc, time, energy, comm_cost, -fairness], dtype=float)
            out["G"] = np.array([energy - budget], dtype=float)

    problem = GAFLProblem()
    ref_dirs = get_reference_directions("energy", 5, pop_size)
    algorithm = PymooNSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    seed = int(fl.get("seed", 42))
    res = minimize(
        problem,
        algorithm,
        ("n_gen", int(generations)),
        seed=seed,
        verbose=False,
    )

    pop: List[Dict[str, Any]] = []
    metrics_list: List[Dict[str, float]] = []

    X = np.atleast_2d(res.X) if res.X is not None else np.empty((0, 10))
    for x in X:
        params: Dict[str, Any] = {
            "energy_w": float(x[0]),
            "channel_w": float(x[1]),
            "data_w": float(x[2]),
            "fair_w": float(x[3]),
            "selection_top_k": int(round(float(x[4]))),
            "bandwidth_alloc_factor": float(x[5]),
            "staleness_alpha": float(x[6]),
            "bridge_to_async": float(x[7]),
            "bridge_to_semi_sync": float(x[8]),
            "compression_ratio": float(x[9]),
        }
        params = repair(params)
        pop.append(params)
        metrics_list.append(evaluate_solution(sim_runner, params))

    return pop, metrics_list
