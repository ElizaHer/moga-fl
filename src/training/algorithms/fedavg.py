from typing import Dict, Any, List
import copy


def aggregate_fedavg(global_state: Dict[str, Any], client_states: List[Dict[str, Any]], weights: List[float]):
    # Weighted average of model parameters
    avg_state = copy.deepcopy(global_state)
    total = sum(weights) + 1e-9
    for k in avg_state.keys():
        avg_state[k] = sum(w * cs[k] for cs, w in zip(client_states, weights)) / total
    return avg_state
