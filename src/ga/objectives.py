from typing import Dict, Any

# Multi-objective: maximize accuracy, minimize time/energy cost, maximize fairness

def evaluate_solution(sim_runner, params: Dict[str, Any]) -> Dict[str, float]:
    # Simulate a short run with given params and return objectives
    res = sim_runner(params)
    # Objectives
    return {
        'acc': res['accuracy_mean'],
        'time': res['comm_time_mean'],
        'fairness': res['jain_mean'],
        'energy': res['energy_mean'],
    }
