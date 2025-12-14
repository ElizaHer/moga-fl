from typing import Dict, Any

class FairnessDebtLedger:
    def __init__(self, cfg: Dict[str, Any], num_clients: int):
        fc = cfg['scheduling'].get('fairness_ledger', {})
        self.debt_increase = fc.get('debt_increase', 0.05)
        self.repay_rate = fc.get('repay_rate', 0.1)
        self.max_debt = fc.get('max_debt', 1.0)
        self.debt = {i: 0.0 for i in range(num_clients)}

    def on_round_end(self, selected_clients):
        for i in self.debt:
            if i in selected_clients:
                # repay
                self.debt[i] = max(0.0, self.debt[i] - self.repay_rate)
            else:
                # increase
                self.debt[i] = min(self.max_debt, self.debt[i] + self.debt_increase)

    def get(self, cid: int) -> float:
        return self.debt.get(cid, 0.0)
