from typing import List, Dict
import numpy as np

class TopKSelector:
    def __init__(self, k: int, sliding_window: int = 5, hysteresis: float = 0.05):
        self.k = k
        self.window = sliding_window
        self.hysteresis = hysteresis
        self.history = []

    def select(self, scores: np.ndarray) -> List[int]:
        idx = np.argsort(scores)[::-1]
        top = idx[:self.k].tolist()
        # Debounce: keep some previous if score differences are small
        if self.history:
            prev = self.history[-1]
            boundary = scores[idx[self.k-1]] if self.k > 0 else 0
            for p in prev:
                if p not in top and scores[p] >= boundary - self.hysteresis:
                    # swap in
                    top[-1] = p
        self.history.append(top)
        if len(self.history) > self.window:
            self.history.pop(0)
        return top
