from typing import Dict, Any, List
import numpy as np


class BandwidthAllocator:
    def __init__(self, cfg: Dict[str, Any]):
        self.budget_mb = cfg.get('bandwidth_budget_mb_per_round', 10.0)
        self.beta = float(cfg.get('bandwidth_alloc_beta', 0.5))
        self.eps = 1e-9

    def allocate_uniform(self, selected_clients):
        if len(selected_clients) == 0:
            return {}
        per_client_mb = self.budget_mb / len(selected_clients)
        return {cid: per_client_mb for cid in selected_clients}

    def allocate_ratios(self, wireless_stats: Dict[int, Dict[str, float]], selected_clients: List[int]) -> Dict[int, float]:
        """Allocate normalized bandwidth ratios for selected clients.

        Input:
        - wireless_stats: {cid: {"snr_db"/"snr_lin"/"per"...}}
        - selected_clients: top-k client ids for current round

        Output:
        - {cid: ratio} where sum(ratio)=1 over selected_clients.
        """
        if len(selected_clients) == 0:
            return {}

        raw_scores: Dict[int, float] = {}
        for cid in selected_clients:
            stats = wireless_stats.get(cid, {})
            snr_lin = float(stats.get("snr_lin", 0.0))
            if snr_lin <= 0.0:
                snr_db = float(stats.get("snr_db", 0.0))
                snr_lin = float(10.0 ** (snr_db / 10.0))
            per = float(stats.get("per", 0.0))
            link_quality = max(self.eps, (1.0 - per))
            raw_scores[cid] = max(self.eps, (snr_lin ** self.beta) * link_quality)

        total = float(np.sum(list(raw_scores.values())))
        if total <= self.eps:
            uniform = 1.0 / float(len(selected_clients))
            return {cid: uniform for cid in selected_clients}
        return {cid: float(raw_scores[cid] / total) for cid in selected_clients}

    def allocate_by_stats(self, wireless_stats: Dict[int, Dict[str, float]], selected_clients: List[int]) -> Dict[int, float]:
        """Allocate actual bandwidth (MB) for selected clients using channel-aware ratios."""
        ratios = self.allocate_ratios(wireless_stats, selected_clients)
        return {cid: float(self.budget_mb * r) for cid, r in ratios.items()}

    def estimate_tx_time(self, payload_mb: float, allocated_mb: float) -> float:
        # Simplified: time proportional to payload / allocation
        if allocated_mb <= 1e-9:
            return 1e9
        return payload_mb / allocated_mb
