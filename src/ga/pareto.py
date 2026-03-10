from typing import List, Dict


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    # Maximize: acc, fairness. Minimize: time, energy, comm_cost.
    comm_a = float(a.get("comm_cost", a.get("time", 0.0)))
    comm_b = float(b.get("comm_cost", b.get("time", 0.0)))
    better_or_equal = (
        float(a["acc"]) >= float(b["acc"])
        and float(a["fairness"]) >= float(b["fairness"])
        and float(a["time"]) <= float(b["time"])
        and float(a["energy"]) <= float(b["energy"])
        and comm_a <= comm_b
    )
    strictly_better = (
        float(a["acc"]) > float(b["acc"])
        or float(a["fairness"]) > float(b["fairness"])
        or float(a["time"]) < float(b["time"])
        or float(a["energy"]) < float(b["energy"])
        or comm_a < comm_b
    )
    return better_or_equal and strictly_better


def non_dominated_set(solutions: List[Dict[str, float]]) -> List[int]:
    nd = []
    for i in range(len(solutions)):
        dom = False
        for j in range(len(solutions)):
            if i != j and dominates(solutions[j], solutions[i]):
                dom = True
                break
        if not dom:
            nd.append(i)
    return nd
