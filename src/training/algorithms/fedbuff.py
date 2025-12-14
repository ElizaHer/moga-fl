from typing import Dict, Any, List
import copy

class FedBuffBuffer:
    def __init__(self, buffer_size: int, staleness_alpha: float, max_staleness: int | None = None):
        """简化版 FedBuff 缓冲队列。

        - buffer_size: 最大缓冲更新条数；
        - staleness_alpha: 陈旧度衰减指数；
        - max_staleness: 若不为 None，则超过该陈旧度的更新会被丢弃。

        对应问题3：高丢包场景下的异步聚合（FedBuff 风格）。
        """
        self.buffer_size = buffer_size
        self.alpha = staleness_alpha
        self.max_staleness = max_staleness
        self.entries = []  # list of (state_dict, staleness)

    def push(self, state_dict: Dict[str, Any], staleness: int):
        self.entries.append((copy.deepcopy(state_dict), int(staleness)))
        if len(self.entries) > self.buffer_size:
            # 丢弃最旧的一条
            self.entries.pop(0)

    def age_entries(self, delta: int = 1):
        """让缓冲中的条目整体“变旧”，用于跨轮累积陈旧度。"""
        if delta <= 0 or not self.entries:
            return
        new_entries = []
        for sd, stale in self.entries:
            new_stale = stale + delta
            if self.max_staleness is not None and new_stale > self.max_staleness:
                # 超过最大陈旧度则直接丢弃
                continue
            new_entries.append((sd, new_stale))
        self.entries = new_entries

    def aggregate(self, global_state: Dict[str, Any]):
        if not self.entries:
            return global_state
        total_w = 0.0
        out = copy.deepcopy(global_state)
        for k in out.keys():
            out[k] = 0.0
        for sd, stale in self.entries:
            w = 1.0 / (1.0 + stale) ** self.alpha
            total_w += w
            for k in out.keys():
                out[k] += w * sd[k]
        for k in out.keys():
            out[k] = out[k] / max(1e-9, total_w)
        # 清空缓冲，等待下一批异步更新
        self.entries.clear()
        return out
