from __future__ import annotations

from typing import Any, Dict

import numpy as np


class ClientScorer:
    """Multi-metric client scoring.

    能量充足度、信道质量、数据价值、公平债务与带宽成本等指标先归一化，
    再按配置权重做加权求和，输出每个客户端的综合得分。
    """

    def __init__(self, cfg: Dict[str, Any], num_clients: int, ledger: "FairnessDebtLedger") -> None:
        w = cfg.get("scheduling", {}).get("weights", {})
        self.weights = w
        self.num_clients = num_clients
        self.ledger = ledger

    @staticmethod
    def _normalize(arr):
        a = np.array(arr, dtype=float)
        if np.allclose(a.max(), a.min()):
            return np.ones_like(a)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    def score(self, energy_avail, channel_quality, data_value, bandwidth_cost):
        energy_n = self._normalize(energy_avail)
        channel_n = self._normalize(channel_quality)
        data_n = self._normalize(data_value)
        bwc_n = self._normalize(bandwidth_cost)
        fairness = np.array([self.ledger.get(i) for i in range(self.num_clients)])
        fairness_n = self._normalize(fairness)

        s = (
            self.weights.get("energy", 0) * energy_n
            + self.weights.get("channel", 0) * channel_n
            + self.weights.get("data_value", 0) * data_n
            + self.weights.get("fairness_debt", 0) * fairness_n
            + self.weights.get("bandwidth_cost", 0) * bwc_n
        )
        return s
