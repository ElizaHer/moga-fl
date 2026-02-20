from __future__ import annotations

from typing import Any, Dict


class BandwidthAllocator:
    """Per-round bandwidth budgeting helper.

    与原实现保持等价：给定每轮总带宽预算，按被选中客户端均分，并提供
    基于 ``payload_mb / allocated_mb`` 的粗略发送时间估计。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.budget_mb: float = cfg["wireless"].get("bandwidth_budget_mb_per_round", 10.0)

    def allocate_uniform(self, selected_clients):
        if len(selected_clients) == 0:
            return {}
        per_client_mb = self.budget_mb / len(selected_clients)
        return {cid: per_client_mb for cid in selected_clients}

    def estimate_tx_time(self, payload_mb: float, allocated_mb: float) -> float:
        """Approximate transmission time as payload / allocated_bandwidth."""
        if allocated_mb <= 1e-9:
            return 1e9
        return payload_mb / allocated_mb
