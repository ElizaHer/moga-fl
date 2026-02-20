from __future__ import annotations

from typing import List

import numpy as np


class TopKSelector:
    """Top-K selector with simple hysteresis-based debounce.

    - 每轮按得分排序取前 K 个；
    - 若上一轮某客户端得分与当前边界差不多，则保留它以减少抖动；
    - 维护有限长度历史以便后续统计 Jain 公平指数。
    """

    def __init__(self, k: int, sliding_window: int = 5, hysteresis: float = 0.05) -> None:
        self.k = int(k)
        self.window = int(sliding_window)
        self.hysteresis = float(hysteresis)
        self.history: List[List[int]] = []

    def select(self, scores: np.ndarray) -> List[int]:
        idx = np.argsort(scores)[::-1]
        top = idx[: self.k].tolist()
        if self.history:
            prev = self.history[-1]
            boundary = scores[idx[self.k - 1]] if self.k > 0 else 0.0
            for p in prev:
                if p not in top and scores[p] >= boundary - self.hysteresis:
                    top[-1] = int(p)
        self.history.append(top)
        if len(self.history) > self.window:
            self.history.pop(0)
        return top
