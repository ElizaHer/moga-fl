import numpy as np
from typing import Dict, Any
from .channel_models import get_channel_model_params


class ChannelSimulator:
    def __init__(self, cfg: Dict[str, Any], num_clients: int):
        wcfg = cfg.get('wireless', {})
        model_name = wcfg.get('channel_model', 'rayleigh_siso')
        params = get_channel_model_params(model_name, wcfg)
        # 兼容旧配置：若用户在 YAML 中直接写了数值，则在近似模型基础上做覆盖
        self.intensity = wcfg.get('block_fading_intensity', params['block_fading_intensity'])
        self.base_snr_db = wcfg.get('base_snr_db', params['base_snr_db'])
        self.per_k = wcfg.get('per_k', params['per_k'])
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
