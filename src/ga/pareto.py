from typing import List, Dict


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    # a dominates b if better or equal in all, and strictly better in one
    better_or_equal = (
        a['acc'] >= b['acc'] and
        a['fairness'] >= b['fairness'] and
        a['time'] <= b['time'] and
        a['energy'] <= b['energy']
    )
    strictly_better = (
        a['acc'] > b['acc'] or
        a['fairness'] > b['fairness'] or
        a['time'] < b['time'] or
        a['energy'] < b['energy']
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
