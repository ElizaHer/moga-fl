from typing import Dict, Any

def penalty(cfg: Dict[str, Any], metrics: Dict[str, float]) -> float:
    # Simple penalty if energy exceeds some budget or time too high
    pen = 0.0
    energy_budget = cfg['wireless']['tx_power_watts'] * 5.0  # heuristic
    if metrics['energy'] > energy_budget:
        pen += (metrics['energy'] - energy_budget)
    return pen


def repair(params: Dict[str, Any]) -> Dict[str, Any]:
    params['selection_top_k'] = max(1, int(params['selection_top_k']))
    params['hysteresis'] = float(min(0.2, max(0.0, params['hysteresis'])))
    return params

