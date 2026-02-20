from __future__ import annotations

from typing import Any, Dict


class FairnessDebtLedger:
    """Track per-client fairness debt over rounds.

    与原工程等价：被选中客户端“还债”，未选中客户端“加债”，并限制在
    ``[0, max_debt]`` 区间内，用于在调度评分中提升长期未参与客户端的权重。
    """

    def __init__(self, cfg: Dict[str, Any], num_clients: int) -> None:
        fc = cfg.get("scheduling", {}).get("fairness_ledger", {})
        self.debt_increase: float = fc.get("debt_increase", 0.05)
        self.repay_rate: float = fc.get("repay_rate", 0.1)
        self.max_debt: float = fc.get("max_debt", 1.0)
        self.debt: Dict[int, float] = {i: 0.0 for i in range(num_clients)}

    def on_round_end(self, selected_clients) -> None:
        for i in self.debt:
            if i in selected_clients:
                # repay
                self.debt[i] = max(0.0, self.debt[i] - self.repay_rate)
            else:
                # increase
                self.debt[i] = min(self.max_debt, self.debt[i] + self.debt_increase)

    def get(self, cid: int) -> float:
        return self.debt.get(cid, 0.0)
