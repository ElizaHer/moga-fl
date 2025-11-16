from typing import Dict, Any

class BandwidthAllocator:
    def __init__(self, cfg: Dict[str, Any]):
        self.budget_mb = cfg['wireless'].get('bandwidth_budget_mb_per_round', 10.0)

    def allocate_uniform(self, selected_clients):
        if len(selected_clients) == 0:
            return {}
        per_client_mb = self.budget_mb / len(selected_clients)
        return {cid: per_client_mb for cid in selected_clients}

    def estimate_tx_time(self, payload_mb: float, allocated_mb: float) -> float:
        # Simplified: time proportional to payload / allocation
        if allocated_mb <= 1e-9:
            return 1e9
        return payload_mb / allocated_mb
