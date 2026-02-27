from typing import Any, Deque, Dict, List, Tuple
from collections import deque
import copy
import numpy as np


class FedBuffState:
    def __init__(
        self,
        alpha: float,
        max_staleness: int,
        buffer_size: int,
        min_updates: int,
        async_agg_interval: int,
    ) -> None:
        self.alpha = alpha
        self.max_staleness = max_staleness
        self.buffer_size = buffer_size
        self.min_updates = min_updates
        self.async_agg_interval = async_agg_interval

        # entries: (params, staleness, num_examples)
        self.entries: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        self.updates_since_last_agg = 0
        self.last_agg_round = 0
        self.last_age_round = 0

    def age(self, current_round: int) -> None:
        """按轮次差值 old->new 进行 aging，避免重复过度老化。"""
        if current_round <= self.last_age_round:
            return
        delta = current_round - self.last_age_round
        if delta <= 0:
            return
        aged: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        for params, staleness, num_examples in self.entries:
            s = staleness + delta
            if s <= self.max_staleness:
                aged.append((params, s, num_examples))
        self.entries = aged
        self.last_age_round = current_round

    def push(self, params: List[np.ndarray], num_examples: int) -> None:
        self.entries.append((params, 0, num_examples))
        self.updates_since_last_agg += 1

    def should_aggregate(self, server_round: int) -> bool:
        if len(self.entries) >= self.buffer_size:
            return True
        if self.updates_since_last_agg >= self.min_updates:
            return True
        if (
            self.async_agg_interval > 0
            and server_round - self.last_agg_round >= self.async_agg_interval
            and len(self.entries) > 0
        ):
            return True
        return False

    def aggregate(self, fallback_params: List[np.ndarray], server_round: int) -> List[np.ndarray]:
        if not self.entries:
            return fallback_params

        sum_weights = 0.0
        agg = [arr.copy() for arr in fallback_params]
        for i in range(len(agg)):
            if np.issubdtype(agg[i].dtype, np.floating):
                agg[i] = np.zeros_like(agg[i], dtype=np.float32)

        for params, staleness, num_examples in self.entries:
            staleness_weight = 1.0 / ((1.0 + float(staleness)) ** self.alpha)
            w = staleness_weight * float(num_examples)
            sum_weights += w
            for idx, arr in enumerate(params):
                if np.issubdtype(agg[idx].dtype, np.floating):
                    agg[idx] += w * arr.astype(np.float32, copy=False)

        out: List[np.ndarray] = []
        for arr in agg:
            if np.issubdtype(arr.dtype, np.floating):
                out.append(arr / max(sum_weights, 1e-12))
            else:
                out.append(arr)

        self.entries.clear()
        self.updates_since_last_agg = 0
        self.last_agg_round = server_round
        return out
