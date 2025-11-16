from typing import Dict, Any, List
import copy

class FedBuffBuffer:
    def __init__(self, buffer_size: int, staleness_alpha: float):
        self.buffer_size = buffer_size
        self.alpha = staleness_alpha
        self.entries = []  # list of (state_dict, staleness)

    def push(self, state_dict: Dict[str, Any], staleness: int):
        self.entries.append((copy.deepcopy(state_dict), staleness))
        if len(self.entries) > self.buffer_size:
            self.entries.pop(0)

    def aggregate(self, global_state: Dict[str, Any]):
        if not self.entries:
            return global_state
        total_w = 0.0
        out = copy.deepcopy(global_state)
        for k in out.keys():
            out[k] = 0.0
        for sd, stale in self.entries:
            w = 1.0 / (1.0 + stale)**self.alpha
            total_w += w
            for k in out.keys():
                out[k] += w * sd[k]
        for k in out.keys():
            out[k] = out[k] / max(1e-9, total_w)
        self.entries.clear()
        return out
