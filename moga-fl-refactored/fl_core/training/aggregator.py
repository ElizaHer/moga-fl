from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


class FedBuffBuffer:
    """FedBuff-style buffer operating on serialized model parameters (1D tensors).

    该版本与原工程逻辑等价，但直接基于 FedLab 的序列化参数向量实现，
    便于与 :class:`fedlab.core.model_maintainer.ModelMaintainer` 对接。
    """

    def __init__(self, buffer_size: int, staleness_alpha: float, max_staleness: int | None = None) -> None:
        self.buffer_size = int(buffer_size)
        self.alpha = float(staleness_alpha)
        self.max_staleness = max_staleness
        self.entries: List[tuple[torch.Tensor, int]] = []  # list of (params, staleness)

    def push(self, params: torch.Tensor, staleness: int) -> None:
        self.entries.append((params.detach().clone(), int(staleness)))
        if len(self.entries) > self.buffer_size:
            self.entries.pop(0)

    def age_entries(self, delta: int = 1) -> None:
        if delta <= 0 or not self.entries:
            return
        new_entries: List[tuple[torch.Tensor, int]] = []
        for params, stale in self.entries:
            new_stale = stale + delta
            if self.max_staleness is not None and new_stale > self.max_staleness:
                continue
            new_entries.append((params, new_stale))
        self.entries = new_entries

    def aggregate(self, global_params: torch.Tensor) -> torch.Tensor:
        if not self.entries:
            return global_params
        out = torch.zeros_like(global_params)
        total_w = 0.0
        for params, stale in self.entries:
            w = 1.0 / (1.0 + stale) ** self.alpha
            total_w += w
            out += w * params
        out = out / max(1e-9, total_w)
        self.entries.clear()
        return out


class MogaAggregator:
    """Aggregator supporting sync / semi_sync / async / bridge modes.

    输入与输出均为 FedLab 样式的“序列化模型参数向量” (:class:`torch.Tensor`)。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        fb = cfg.get("training", {}).get("fedbuff", {})
        self.mode = cfg.get("training", {}).get("sync_mode", "sync")
        self.buffer: Optional[FedBuffBuffer] = None
        self.buffer_size = fb.get("buffer_size", 32)
        if fb.get("enabled", False):
            self.buffer = FedBuffBuffer(
                self.buffer_size,
                fb.get("staleness_alpha", 1.0),
                max_staleness=fb.get("max_staleness", None),
            )
        self.async_min_updates = fb.get("min_updates_to_aggregate", max(1, self.buffer_size // 2))
        self.async_agg_interval = fb.get("async_agg_interval", None)
        self.updates_since_last_agg = 0
        self.last_agg_round: Optional[int] = None

    @staticmethod
    def _fedavg_aggregate(
        global_params: torch.Tensor,
        client_params: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        if not client_params:
            return global_params
        total = float(sum(weights)) + 1e-9
        out = torch.zeros_like(global_params)
        for w, p in zip(weights, client_params):
            out += float(w) * p
        return out / total

    def aggregate(
        self,
        global_params: torch.Tensor,
        client_params: List[torch.Tensor],
        weights: List[float],
        *,
        staleness_list: Optional[List[int]] = None,
        round_idx: Optional[int] = None,
        mode: Optional[str] = None,
        bridge_weight: float = 0.5,
    ) -> torch.Tensor:
        """Aggregate client updates according to current mode.

        当未启用 FedBuff 或处于 sync / semi_sync 模式时，退化为普通 FedAvg；
        在 async / bridge 模式下使用 FedBuff 缓冲与陈旧度加权聚合，并在 bridge
        模式中对同步结果与异步结果做线性插值。
        """

        current_mode = mode or self.mode
        print(f"[sync mode] {current_mode}")

        # 纯同步 / 半同步：直接 FedAvg
        if self.buffer is None or current_mode in ("sync", "semi_sync"):
            print("[(semi) sync agg]")
            if not client_params:
                return global_params
            return self._fedavg_aggregate(global_params, client_params, weights)

        # -------- 异步 / 桥接：FedBuff 缓冲 --------
        if round_idx is not None:
            if self.last_agg_round is None:
                self.last_agg_round = round_idx
            else:
                delta = max(0, round_idx - self.last_agg_round)
                if delta > 0:
                    self.buffer.age_entries(delta)
                    self.last_agg_round = round_idx

        # push 新更新
        if client_params:
            if staleness_list is None:
                staleness_list = [0] * len(client_params)
            for p, st in zip(client_params, staleness_list):
                self.buffer.push(p, st)
            self.updates_since_last_agg += len(client_params)

        # 判断是否触发一次异步聚合
        should_agg = False
        if len(self.buffer.entries) >= self.buffer_size:
            should_agg = True
        if self.updates_since_last_agg >= self.async_min_updates:
            should_agg = True
        if self.async_agg_interval is not None and round_idx is not None:
            if self.last_agg_round is None or (
                round_idx - self.last_agg_round
            ) >= self.async_agg_interval:
                should_agg = True

        async_result = global_params
        if should_agg:
            async_result = self.buffer.aggregate(global_params)
            self.updates_since_last_agg = 0
            self.last_agg_round = round_idx

        if current_mode == "async":
            print("[async agg]")
            return async_result

        # bridge：同步结果与异步结果加权混合
        print(f"[bridge agg] bridge weight: {bridge_weight}")
        if not client_params:
            return async_result

        sync_result = self._fedavg_aggregate(global_params, client_params, weights)
        w = float(np.clip(bridge_weight, 0.0, 1.0))
        mixed = (1.0 - w) * sync_result + w * async_result
        return mixed
