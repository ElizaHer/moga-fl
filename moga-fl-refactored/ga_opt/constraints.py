from __future__ import annotations

from typing import Any, Dict


def penalty(cfg: Dict[str, Any], metrics: Dict[str, float]) -> float:
    """Simple energy budget penalty.

    若能耗超过一个粗略预算，则返回正的惩罚值；该惩罚会在目标函数中加到
    ``energy`` 上，从而在多目标优化过程中弱化超预算解。
    """

    pen = 0.0
    energy_budget = cfg.get("wireless", {}).get("tx_power_watts", 1.0) * 5.0
    if metrics.get("energy", 0.0) > energy_budget:
        pen += metrics["energy"] - energy_budget
    return pen


def repair(params: Dict[str, Any]) -> Dict[str, Any]:
    """Repair raw continuous parameters into a valid scheduling configuration."""

    p = dict(params)
    p["selection_top_k"] = max(1, int(round(p.get("selection_top_k", 1))))
    p["hysteresis"] = float(min(0.2, max(0.0, p.get("hysteresis", 0.05))))
    return p
