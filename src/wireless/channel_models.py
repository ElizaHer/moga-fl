from typing import Dict, Any


def get_channel_model_params(name: str, wcfg: Dict[str, Any]) -> Dict[str, float]:
    """返回给定场景名下的一组简化信道统计参数。

    说明（工程近似，而非完整仿真器）：
    - 我们不直接调用 DeepMIMO / NYUSIM / QuaDRiGa / 3GPP TR 38.901 的原始代码，
      而是提取其公开文档中的典型路径损耗与阴影参数，在此构造“统计近似版”模型；
    - 统一接口：根据 `wireless.channel_model` 的值，给出 Rayleigh 块衰落强度、
      参考 SNR、PER 映射参数 `per_k`，以及用户距离范围和 log-normal 阴影标准差；
    - 具体的路径损耗计算放在 :mod:`channel.py` 中完成，这里只负责给出场景级别的
      典型参数（例如 UMi 场景的阴影标准差取 3~7 dB）。

    参数
    ------
    name: 场景名，例如：
        - "deepmimo_like_urban": 近似 DeepMIMO 中的毫米波都市街区场景；
        - "nyusim_like_mmwave": 近似 NYUSIM 毫米波室外场景；
        - "quadriga_like_macro": 近似 QuaDRiGa 宏蜂窝小区；
        - "tr38901_umi": 近似 3GPP TR 38.901 UMi NLOS 场景；
        - 其它值默认为基础 Rayleigh SISO；
    wcfg: 来自配置文件的 wireless 字典，可包含频段、距离等辅助信息。
    """
    carrier = float(wcfg.get('carrier_ghz', 3.5))  # 载频 GHz

    # 默认参数：相对温和的 Rayleigh SISO 信道
    params: Dict[str, float] = {
        'block_fading_intensity': float(wcfg.get('block_fading_intensity', 1.0)),
        'base_snr_db': float(wcfg.get('base_snr_db', 12.0)),  # 参考距离处的平均 SNR
        'per_k': float(wcfg.get('per_k', 1.0)),               # SNR→PER 的指数衰减系数
        # 额外用于路径损耗与阴影的参数
        'd_min_m': float(wcfg.get('d_min_m', 10.0)),          # 用户与 BS 的最近距离
        'd_max_m': float(wcfg.get('d_max_m', 250.0)),         # 用户与 BS 的最远距离
        'shadowing_sigma_db': float(wcfg.get('shadowing_sigma_db', 4.0)),
    }

    name_l = (name or '').lower()
    if name_l == 'deepmimo_like_urban':
        # DeepMIMO 常用于 28GHz 毫米波城市街区，建筑遮挡多，多径数有限：
        params['block_fading_intensity'] = 0.8
        params['base_snr_db'] = 10.0 if carrier >= 28.0 else 8.0
        params['per_k'] = 1.2
        params['d_min_m'] = 20.0
        params['d_max_m'] = 200.0
        params['shadowing_sigma_db'] = 5.0
    elif name_l == 'nyusim_like_mmwave':
        # NYUSIM 提供 LoS/NLoS 毫米波信道，这里取室外街区中等 SNR 场景：
        params['block_fading_intensity'] = 0.9
        params['base_snr_db'] = 14.0 if carrier >= 28.0 else 12.0
        params['per_k'] = 0.9
        params['d_min_m'] = 10.0
        params['d_max_m'] = 150.0
        params['shadowing_sigma_db'] = 4.0
    elif name_l == 'quadriga_like_macro':
        # QuaDRiGa 宏蜂窝场景：子 6GHz，阴影衰落较强，较大覆盖半径：
        params['block_fading_intensity'] = 1.1
        params['base_snr_db'] = 13.0
        params['per_k'] = 0.8
        params['d_min_m'] = 50.0
        params['d_max_m'] = 500.0
        params['shadowing_sigma_db'] = 6.0
    elif name_l in ('tr38901_umi', 'tr38901_like'):
        # 3GPP TR 38.901 UMi 场景（简化）：
        #   典型路径损耗模型（LoS/NLoS 简化）：
        #     PL[dB] ≈ 32.4 + 21*log10(d[m]) + 20*log10(fc[GHz])
        #   这里不区分 LoS/NLoS，只用一条经验式，并给出较小的阴影标准差。
        params['block_fading_intensity'] = 1.0
        params['base_snr_db'] = 11.0
        params['per_k'] = 1.0
        params['d_min_m'] = 10.0
        params['d_max_m'] = 300.0
        params['shadowing_sigma_db'] = 3.0
    # 其它值保持默认 Rayleigh SISO

    return params
