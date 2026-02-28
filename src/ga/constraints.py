from __future__ import annotations

from typing import Any, Dict


def _energy_budget(cfg: Dict[str, Any]) -> float:
    """Infer a per-round energy budget from config.

    Priority:
    1. ``controller.bridge_invariants.energy_budget_round`` if provided.
    2. Otherwise, derive a simple heuristic from wireless TX/compute power.
    """
    controller = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
    bridge_inv = controller.get("bridge_invariants", {}) if isinstance(controller.get("bridge_invariants", {}), dict) else {}
    if "energy_budget_round" in bridge_inv:
        try:
            return float(bridge_inv["energy_budget_round"])
        except (TypeError, ValueError):
            pass

    wireless = cfg.get("wireless", {}) if isinstance(cfg.get("wireless", {}), dict) else {}
    tx_power = float(wireless.get("tx_power_watts", 1.0))
    compute_power = float(wireless.get("compute_power_watts", 8.0))
    # Rough heuristic: assume an effective 10-second window of combined power.
    return float((tx_power + compute_power) * 10.0)


def penalty(cfg: Dict[str, Any], metrics: Dict[str, float]) -> float:
    """Simple energy-budget penalty.

    If the averaged per-round energy exceeds the inferred budget, we return a
    positive penalty term (added onto the ``energy`` objective). Otherwise,
    penalty is zero.
    """
    energy = float(metrics.get("energy", 0.0))
    budget = _energy_budget(cfg)
    if budget <= 0.0:
        return 0.0
    if energy <= budget:
        return 0.0
    return float(energy - budget)


def repair(params: Dict[str, Any]) -> Dict[str, Any]:
    """Repair potentially invalid GA parameters.

    Current rules:
    - ``selection_top_k``: at least 1 (integer)
    - ``staleness_alpha``: clamped to [0.5, 2.0]

    Historic fields like ``hysteresis`` are no longer part of the individual
    and are thus not touched here.
    """
    fixed: Dict[str, Any] = dict(params)

    if "selection_top_k" in fixed:
        try:
            k = int(fixed["selection_top_k"])
        except (TypeError, ValueError):
            k = 1
        fixed["selection_top_k"] = max(1, k)

    if "staleness_alpha" in fixed:
        try:
            alpha = float(fixed["staleness_alpha"])
        except (TypeError, ValueError):
            alpha = 1.0
        alpha = min(2.0, max(0.5, alpha))
        fixed["staleness_alpha"] = float(alpha)

    return fixed
