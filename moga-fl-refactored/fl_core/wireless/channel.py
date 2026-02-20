from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .channel_models import get_channel_model_params


class ChannelSimulator:
    """简化无线信道仿真器。

    该实现与原工程保持一致：路径损耗 + log-normal 阴影 + Rayleigh 小尺度衰落，
    再通过 ``per = exp(-k * SNR_lin)`` 映射到丢包率，用于驱动联邦调度与聚合模式切换。
    """

    def __init__(self, cfg: Dict[str, Any], num_clients: int) -> None:
        wcfg = cfg.get("wireless", {})
        model_name = wcfg.get("channel_model", "rayleigh_siso")
        params = get_channel_model_params(model_name, wcfg)

        # 兼容用户在 YAML 中直接覆盖数值的配置方式
        self.intensity = wcfg.get("block_fading_intensity", params["block_fading_intensity"])
        self.base_snr_db = wcfg.get("base_snr_db", params["base_snr_db"])
        self.per_k = wcfg.get("per_k", params["per_k"])
        self.d_min = float(wcfg.get("d_min_m", params.get("d_min_m", 10.0)))
        self.d_max = float(wcfg.get("d_max_m", params.get("d_max_m", 250.0)))
        self.shadowing_sigma_db = float(
            wcfg.get("shadowing_sigma_db", params.get("shadowing_sigma_db", 4.0))
        )
        self.carrier_ghz = float(wcfg.get("carrier_ghz", 3.5))

        self.num_clients = num_clients

        # 简单“用户位置/距离”模型：每个客户端一个随机距离，可随轮缓慢移动
        self.distances = np.random.uniform(self.d_min, self.d_max, size=self.num_clients)
        mob = wcfg.get("mobility", {})
        self.mobility_enabled = bool(mob.get("enabled", False))
        self.max_step_m = float(mob.get("max_step_m_per_round", 5.0))

    # ---------------- 内部工具 -----------------
    def _update_positions(self) -> None:
        if not self.mobility_enabled:
            return
        steps = np.random.uniform(-self.max_step_m, self.max_step_m, size=self.num_clients)
        self.distances = np.clip(self.distances + steps, self.d_min, self.d_max)

    def _pathloss_db(self, d_m: np.ndarray) -> np.ndarray:
        """3GPP 风格路径损耗近似。

        采用 UMi 形式：
            ``PL[dB] ≈ 32.4 + 21*log10(d[m]) + 20*log10(fc[GHz])``
        """
        fc = self.carrier_ghz
        d = np.maximum(d_m, 1.0)
        return 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)

    # ---------------- 对外接口 -----------------
    def sample_round(self) -> Dict[int, Dict[str, float]]:
        # 1) 更新用户位置（若启用 mobility）
        self._update_positions()
        d = self.distances

        # 2) 路径损耗：以某个参考距离 d0 处的 base_snr_db 作为平均 SNR
        d0 = 50.0
        pl0 = self._pathloss_db(np.array([d0]))[0]
        pl = self._pathloss_db(d)
        large_scale_snr_db = float(self.base_snr_db) - (pl - pl0)

        # 3) log-normal 阴影
        shadowing = np.random.normal(0.0, self.shadowing_sigma_db, size=self.num_clients)

        # 4) Rayleigh 小尺度衰落
        h = np.random.rayleigh(scale=self.intensity, size=self.num_clients)
        small_scale_db = 10.0 * np.log10(h + 1e-6)

        snr_db = large_scale_snr_db + shadowing + small_scale_db
        snr_lin = 10 ** (snr_db / 10.0)

        # 5) 从 SNR 映射到丢包率 PER
        per = np.exp(-float(self.per_k) * snr_lin)
        stats: Dict[int, Dict[str, float]] = {}
        for i in range(self.num_clients):
            stats[i] = {
                "snr_db": float(snr_db[i]),
                "snr_lin": float(snr_lin[i]),
                "per": float(np.clip(per[i], 0.0, 1.0)),
                "distance_m": float(d[i]),
            }
        return stats
