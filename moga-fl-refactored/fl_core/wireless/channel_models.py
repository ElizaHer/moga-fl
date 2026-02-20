from __future__ import annotations

from typing import Any, Dict


def get_channel_model_params(name: str, wcfg: Dict[str, Any]) -> Dict[str, float]:
    """Return simplified statistical channel parameters for a named scenario.

    该函数与原工程 `src/wireless/channel_models.py` 保持一致，只在注释上
    略作整理，用于为 :mod:`channel` 提供 Rayleigh 块衰落强度、参考 SNR、
    PER 映射系数等参数。
    """

    carrier = float(wcfg.get("carrier_ghz", 3.5))  # GHz

    # 默认：温和 Rayleigh SISO 信道
    params: Dict[str, float] = {
        "block_fading_intensity": float(wcfg.get("block_fading_intensity", 1.0)),
        "base_snr_db": float(wcfg.get("base_snr_db", 12.0)),
        "per_k": float(wcfg.get("per_k", 1.0)),
        "d_min_m": float(wcfg.get("d_min_m", 10.0)),
        "d_max_m": float(wcfg.get("d_max_m", 250.0)),
        "shadowing_sigma_db": float(wcfg.get("shadowing_sigma_db", 4.0)),
    }

    name_l = (name or "").lower()
    if name_l == "deepmimo_like_urban":
        params["block_fading_intensity"] = 0.8
        params["base_snr_db"] = 10.0 if carrier >= 28.0 else 8.0
        params["per_k"] = 1.2
        params["d_min_m"] = 20.0
        params["d_max_m"] = 200.0
        params["shadowing_sigma_db"] = 5.0
    elif name_l == "nyusim_like_mmwave":
        params["block_fading_intensity"] = 0.9
        params["base_snr_db"] = 14.0 if carrier >= 28.0 else 12.0
        params["per_k"] = 0.9
        params["d_min_m"] = 10.0
        params["d_max_m"] = 150.0
        params["shadowing_sigma_db"] = 4.0
    elif name_l == "quadriga_like_macro":
        params["block_fading_intensity"] = 1.1
        params["base_snr_db"] = 13.0
        params["per_k"] = 0.8
        params["d_min_m"] = 50.0
        params["d_max_m"] = 500.0
        params["shadowing_sigma_db"] = 6.0
    elif name_l in ("tr38901_umi", "tr38901_like"):
        params["block_fading_intensity"] = 1.0
        params["base_snr_db"] = 11.0
        params["per_k"] = 1.0
        params["d_min_m"] = 10.0
        params["d_max_m"] = 300.0
        params["shadowing_sigma_db"] = 3.0

    return params
