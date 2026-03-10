from __future__ import annotations

from typing import Any, Callable, Dict


def evaluate_solution(
    sim_runner: Callable[[Dict[str, Any]], Dict[str, float]],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate a GA individual and return normalized multi-objective metrics.

    This helper standardizes the objective interface for all optimizers.

    The underlying ``sim_runner`` is responsible for running a (possibly
    low-fidelity) FL simulation and must return a dict containing the
    aggregated objectives. We normalize the key names to:

    - ``acc``       : global accuracy (higher is better)
    - ``time``      : training time / latency proxy (lower is better)
    - ``energy``    : per-round energy consumption (lower is better)
    - ``comm_cost`` : communication payload/cost proxy (lower is better)
    - ``fairness``  : Jain index or similar (higher is better)
    """
    res = sim_runner(params)

    # Be slightly defensive: fall back to a few legacy names if needed, but the
    # new Flower runner already returns the canonical keys used below.
    acc = res.get("acc")
    if acc is None:
        acc = res.get("accuracy") or res.get("accuracy_mean") or 0.0

    time = res.get("time")
    if time is None:
        time = res.get("comm_time") or res.get("comm_time_mean") or 0.0

    energy = res.get("energy")
    if energy is None:
        energy = res.get("energy_mean") or 0.0

    comm_cost = res.get("comm_cost")
    if comm_cost is None:
        comm_cost = res.get("communication_cost") or res.get("comm_cost_mean") or time

    fairness = res.get("fairness")
    if fairness is None:
        fairness = res.get("jain") or res.get("jain_mean") or 0.0

    return {
        "acc": float(acc),
        "time": float(time),
        "energy": float(energy),
        "comm_cost": float(comm_cost),
        "fairness": float(fairness),
    }
