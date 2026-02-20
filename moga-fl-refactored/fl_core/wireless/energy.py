from __future__ import annotations

from typing import Any, Dict


class EnergyEstimator:
    """Approximate communication and computation energy consumption."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        wcfg = cfg.get("wireless", {})
        self.tx_power = float(wcfg.get("tx_power_watts", 1.0))
        self.compute_power = float(wcfg.get("compute_power_watts", 10.0))
        self.compute_rate = float(wcfg.get("compute_rate_samples_per_sec", 2000))

    def comm_energy(self, tx_time_sec: float) -> float:
        return self.tx_power * tx_time_sec

    def compute_energy(self, samples: int) -> float:
        # Approximate compute time by samples / compute_rate
        t = samples / max(1.0, self.compute_rate)
        return self.compute_power * t
