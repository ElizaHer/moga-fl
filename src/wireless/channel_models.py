from typing import Dict, Any


def get_channel_model_params(name: str, wcfg: Dict[str, Any]) -> Dict[str, float]:
    """返回给定场景名下的一组简化信道统计参数。

    说明：
    - 这里不是直接调用 DeepMIMO / NYUSIM / QuaDRiGa / 3GPP TR 38.901 的
      原始仿真代码，而是根据公开文献中的典型数值做一个“工程上可行的近似”。
    - 目标是提供一个统一入口：根据 `wireless.channel_model` 的值，决定
      Rayleigh 块衰落强度、基础 SNR 和 PER 映射参数等，方便后续在真实项目中
      替换为更精细的信道接口。

    参数
    ------
    name: 场景名，例如：
        - "deepmimo_like_urban": 近似 DeepMIMO 中毫米波都市街区场景；
        - "nyusim_like_mmwave": 近似 NYUSIM 毫米波室外场景；
        - "quadriga_like_macro": 近似 QuaDRiGa 宏蜂窝小区；
        - "tr38901_umi": 近似 3GPP TR 38.901 UMi NLOS 场景；
        - 其它值默认为基础 Rayleigh SISO；
    wcfg: 来自配置文件的 wireless 字典，可包含频段、距离等辅助信息。
    """
    carrier = float(wcfg.get('carrier_ghz', 3.5))  # 载频 GHz，仅用于区分场景，不做精细计算

    # 默认参数：相对温和的 Rayleigh SISO 信道
    params = {
        'block_fading_intensity': float(wcfg.get('block_fading_intensity', 1.0)),
        'base_snr_db': float(wcfg.get('base_snr_db', 12.0)),
        'per_k': float(wcfg.get('per_k', 1.0)),
    }

    name = (name or '').lower()
    if name == 'deepmimo_like_urban':
        # 毫米波、街区遮挡较多：SNR 波动较大、PER 对 SNR 更敏感
        params['block_fading_intensity'] = 0.8
        # 若载频较高（例如 28GHz 以上），SNR 略高一些
        params['base_snr_db'] = 10.0 if carrier >= 4.0 else 8.0
        params['per_k'] = 1.2
    elif name == 'nyusim_like_mmwave':
        # 室外毫米波 LoS/NLoS 混合：平均 SNR 较高但易被遮挡
        params['block_fading_intensity'] = 0.9
        params['base_snr_db'] = 14.0 if carrier >= 28.0 else 12.0
        params['per_k'] = 0.9
    elif name == 'quadriga_like_macro':
        # Sub-6GHz 宏小区：阴影衰落较强，但多径更丰富
        params['block_fading_intensity'] = 1.1
        params['base_snr_db'] = 13.0
        params['per_k'] = 0.8
    elif name in ('tr38901_umi', 'tr38901_like'):
        # 3GPP UMi NLOS 简化：中等 SNR、相对较高的 PER 斜率
        params['block_fading_intensity'] = 1.0
        params['base_snr_db'] = 11.0
        params['per_k'] = 1.0
    # 其它值保持默认 Rayleigh SISO

    return params
