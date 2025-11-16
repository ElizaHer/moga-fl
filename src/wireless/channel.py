import numpy as np
from typing import Dict, Any

class ChannelSimulator:
    def __init__(self, cfg: Dict[str, Any], num_clients: int):
        self.intensity = cfg['wireless'].get('block_fading_intensity', 1.0)
        self.base_snr_db = cfg['wireless'].get('base_snr_db', 12.0)
        self.per_k = cfg['wireless'].get('per_k', 1.0)
        self.num_clients = num_clients

    def sample_round(self) -> Dict[int, Dict[str, float]]:
        # Rayleigh block fading around a base SNR
        h = np.random.rayleigh(scale=self.intensity, size=self.num_clients)
        snr_db = self.base_snr_db + 10*np.log10(h + 1e-6)
        snr_lin = 10 ** (snr_db / 10.0)
        # Map SNR to packet error rate (PER): per = exp(-k * snr_lin)
        per = np.exp(-self.per_k * snr_lin)
        stats = {}
        for i in range(self.num_clients):
            stats[i] = {
                'snr_db': float(snr_db[i]),
                'snr_lin': float(snr_lin[i]),
                'per': float(np.clip(per[i], 0.0, 1.0)),
            }
        return stats
