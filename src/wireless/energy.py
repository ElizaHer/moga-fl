from typing import Dict, Any

class EnergyEstimator:
    def __init__(self, cfg: Dict[str, Any]):
        wcfg = cfg['wireless']
        self.tx_power = wcfg.get('tx_power_watts', 1.0)
        self.compute_power = wcfg.get('compute_power_watts', 10.0)
        self.compute_rate = wcfg.get('compute_rate_samples_per_sec', 2000)

    def comm_energy(self, tx_time_sec: float) -> float:
        return self.tx_power * tx_time_sec

    def compute_energy(self, samples: int) -> float:
        # Approximate time = samples / compute_rate
        t = samples / max(1, self.compute_rate)
        return self.compute_power * t
