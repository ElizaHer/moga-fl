from typing import Dict, Any, List
import torch
from .algorithms.fedavg import aggregate_fedavg
from .algorithms.fedbuff import FedBuffBuffer

class Aggregator:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        fb = cfg['training'].get('fedbuff', {'enabled': False})
        self.buffer = FedBuffBuffer(fb.get('buffer_size', 32), fb.get('staleness_alpha', 1.0)) if fb.get('enabled', False) else None

    def aggregate(self, global_state: Dict[str, Any], client_states: List[Dict[str, Any]], weights: List[float], staleness_list: List[int] = None):
        if self.buffer is None:
            return aggregate_fedavg(global_state, client_states, weights)
        else:
            # Push into buffer and aggregate
            for sd, st in zip(client_states, staleness_list or [0]*len(client_states)):
                self.buffer.push(sd, st)
            return self.buffer.aggregate(global_state)
