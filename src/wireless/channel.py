import numpy as np
from typing import Dict, Any
from .channel_models import get_channel_model_params


class ChannelSimulator:
    """简化无线信道仿真器：路径损耗 + log-normal 阴影 + Rayleigh 小尺度衰落。

    对应论文中的“无线链路建模”部分：
    - 基于 3GPP TR 38.901 UMi 形式的路径损耗近似：
        PL[dB] ≈ 32.4 + 21*log10(d[m]) + 20*log10(fc[GHz])；
    - 叠加 log-normal 阴影（几个 dB 的高斯扰动）和 Rayleigh 小尺度衰落；
    - 将得到的 SNR 通过 per = exp(-k * SNR_lin) 映射为丢包率，从而驱动 FL 中
      的半同步/异步聚合与门控切换。"""

    def __init__(self, cfg: Dict[str, Any], num_clients: int):
        model_name = cfg.get('channel_model', 'rayleigh_siso')
        params = get_channel_model_params(model_name, cfg)
        # 兼容旧配置：若用户在 YAML 中直接写了数值，则在近似模型基础上做覆盖
        self.intensity = cfg.get('block_fading_intensity', params['block_fading_intensity'])
        self.base_snr_db = cfg.get('base_snr_db', params['base_snr_db'])
        self.per_k = cfg.get('per_k', params['per_k'])
        self.d_min = float(cfg.get('d_min_m', params.get('d_min_m', 10.0)))
        self.d_max = float(cfg.get('d_max_m', params.get('d_max_m', 250.0)))
        self.shadowing_sigma_db = float(cfg.get('shadowing_sigma_db', params.get('shadowing_sigma_db', 4.0)))
        self.carrier_ghz = float(cfg.get('carrier_ghz', 3.5))

        self.num_clients = num_clients

        # 简单的“用户位置/距离”建模：每个客户端一个随机距离，并可随轮缓慢移动
        self.distances = np.random.uniform(self.d_min, self.d_max, size=self.num_clients)
        mob = cfg.get('mobility', {})
        self.mobility_enabled = bool(mob.get('enabled', False))
        self.max_step_m = float(mob.get('max_step_m_per_round', 5.0))

    def _update_positions(self):
        if not self.mobility_enabled:
            return
        # 简单 1D 移动模型：每轮随机向前/向后移动，限制在 [d_min, d_max]
        steps = np.random.uniform(-self.max_step_m, self.max_step_m, size=self.num_clients)
        self.distances = np.clip(self.distances + steps, self.d_min, self.d_max)

    def _pathloss_db(self, d_m: np.ndarray) -> np.ndarray:
        """3GPP 风格路径损耗近似。

        这里采用 UMi 类似形式：
            PL[dB] ≈ 32.4 + 21*log10(d[m]) + 20*log10(fc[GHz])
        对非 UMi 场景，我们依然使用该形式，只是 d_min/d_max、阴影强度不同。"""
        fc = self.carrier_ghz
        d = np.maximum(d_m, 1.0)
        return 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)

    def sample_round(self) -> Dict[int, Dict[str, float]]:
        # 1) 更新用户位置（若启用 mobility）
        self._update_positions()
        d = self.distances

        # 2) 路径损耗：以某个参考距离 d0 处的 base_snr_db 作为平均 SNR
        d0 = 50.0
        pl0 = self._pathloss_db(np.array([d0]))[0]
        pl = self._pathloss_db(d)
        # 大尺度 SNR：参考 SNR 减去相对路径损耗增量
        large_scale_snr_db = float(self.base_snr_db) - (pl - pl0)

        # 3) log-normal 阴影：高斯分布（dB 域）
        shadowing = np.random.normal(0.0, self.shadowing_sigma_db, size=self.num_clients)

        # 4) Rayleigh 小尺度衰落：幅度 ~ Rayleigh，功率对数映射到 dB
        h = np.random.rayleigh(scale=self.intensity, size=self.num_clients)
        small_scale_db = 10.0 * np.log10(h + 1e-6)

        snr_db = large_scale_snr_db + shadowing + small_scale_db
        snr_lin = 10 ** (snr_db / 10.0)

        # 5) Map SNR to packet error rate (PER): per = exp(-k * snr_lin)
        per = np.exp(-float(self.per_k) * snr_lin)
        stats: Dict[int, Dict[str, float]] = {}
        for i in range(self.num_clients):
            stats[i] = {
                'snr_db': float(snr_db[i]),
                'snr_lin': float(snr_lin[i]),
                'per': float(np.clip(per[i], 0.0, 1.0)),
                'distance_m': float(d[i]),
            }
        return stats
