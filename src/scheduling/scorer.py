from typing import Dict, Any
import numpy as np

class ClientScorer:
    def __init__(self, cfg: Dict[str, Any], num_clients: int, ledger):
        w = cfg['scheduling']['weights']
        self.weights = w
        self.num_clients = num_clients
        self.ledger = ledger

    def normalize(self, arr):
        a = np.array(arr, dtype=float)
        if np.allclose(a.max(), a.min()):
            return np.ones_like(a)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    def score(self, energy_avail, channel_quality, data_value, bandwidth_cost):
        # Gather metrics
        energy_n = self.normalize(energy_avail)
        channel_n = self.normalize(channel_quality)
        data_n = self.normalize(data_value)
        bwc_n = self.normalize(bandwidth_cost)
        fairness = np.array([self.ledger.get(i) for i in range(self.num_clients)])
        fairness_n = self.normalize(fairness)
        # Weighted sum (bandwidth_cost expected negative weight)
        s = (
            self.weights.get('energy', 0)*energy_n +
            self.weights.get('channel', 0)*channel_n +
            self.weights.get('data_value', 0)*data_n +
            self.weights.get('fairness_debt', 0)*fairness_n +
            self.weights.get('bandwidth_cost', 0)*bwc_n
        )
        return s
