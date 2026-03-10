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


def _upload_time_budget(cfg: Dict[str, Any]) -> float:
    controller = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
    bridge_inv = controller.get("bridge_invariants", {}) if isinstance(controller.get("bridge_invariants", {}), dict) else {}
    if "upload_time_budget_round" in bridge_inv:
        try:
            return float(bridge_inv["upload_time_budget_round"])
        except (TypeError, ValueError):
            pass
    # Conservative fallback budget for one round.
    return 3.0


def _comm_budget(cfg: Dict[str, Any]) -> float:
    wireless = cfg.get("wireless", {}) if isinstance(cfg.get("wireless", {}), dict) else {}
    fl = cfg.get("fl", {}) if isinstance(cfg.get("fl", {}), dict) else {}
    fraction_fit = float(fl.get("fraction_fit", 0.5))
    num_clients = int(fl.get("num_clients", 10))
    top_k_like = max(1.0, float(num_clients) * max(0.1, fraction_fit))
    # Proxy communication budget proportional to payload and selected clients.
    payload = float(wireless.get("payload_mb", 1.0))
    return top_k_like * payload


def penalty(cfg: Dict[str, Any], metrics: Dict[str, float]) -> float:
    """Budget penalty over energy/time/communication objectives.

    Penalty is additive and non-negative. It is used by optimizers as a soft
    constraint term and should be applied to minimization objectives.
    """
    energy = float(metrics.get("energy", 0.0))
    time = float(metrics.get("time", 0.0))
    comm = float(metrics.get("comm_cost", 0.0))
    budget = _energy_budget(cfg)
    time_budget = _upload_time_budget(cfg)
    comm_budget = _comm_budget(cfg)

    p_energy = max(0.0, energy - budget) if budget > 0.0 else 0.0
    p_time = max(0.0, time - time_budget) if time_budget > 0.0 else 0.0
    p_comm = max(0.0, comm - comm_budget) if comm_budget > 0.0 else 0.0
    return float(p_energy + 0.5 * p_time + 0.25 * p_comm)


def repair(params: Dict[str, Any]) -> Dict[str, Any]:
    """Repair potentially invalid GA parameters.

    Current rules:
    - ``selection_top_k``: at least 1 (integer)
    - ``staleness_alpha``: clamped to [0.5, 2.5]
    - ``bandwidth_alloc_factor``: clamped to [0.6, 1.4]
    - ``bridge_to_async`` in [0.50, 0.90]
    - ``bridge_to_semi_sync`` in [0.10, 0.70] and at least 0.03 below async threshold
    - ``compression_ratio``: clamped to [0.40, 1.00]
    - scheduler weights: non-negative and normalized to sum 1
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
        alpha = min(2.5, max(0.5, alpha))
        fixed["staleness_alpha"] = float(alpha)

    if "bandwidth_alloc_factor" in fixed:
        try:
            b = float(fixed["bandwidth_alloc_factor"])
        except (TypeError, ValueError):
            b = 1.0
        fixed["bandwidth_alloc_factor"] = float(min(1.4, max(0.6, b)))

    if "compression_ratio" in fixed:
        try:
            c = float(fixed["compression_ratio"])
        except (TypeError, ValueError):
            c = 1.0
        fixed["compression_ratio"] = float(min(1.0, max(0.4, c)))

    if "bridge_to_async" in fixed:
        try:
            a = float(fixed["bridge_to_async"])
        except (TypeError, ValueError):
            a = 0.58
        fixed["bridge_to_async"] = float(min(0.9, max(0.5, a)))

    if "bridge_to_semi_sync" in fixed:
        try:
            s = float(fixed["bridge_to_semi_sync"])
        except (TypeError, ValueError):
            s = 0.42
        s = min(0.7, max(0.1, s))
        async_thr = float(fixed.get("bridge_to_async", 0.58))
        s = min(s, async_thr - 0.03)
        fixed["bridge_to_semi_sync"] = float(max(0.1, s))

    weight_keys = ["energy_w", "channel_w", "data_w", "fair_w"]
    if any(k in fixed for k in weight_keys):
        vals = []
        for k in weight_keys:
            try:
                vals.append(max(1e-6, float(fixed.get(k, 0.25))))
            except (TypeError, ValueError):
                vals.append(0.25)
        s = sum(vals)
        if s <= 0:
            vals = [0.25, 0.25, 0.25, 0.25]
            s = 1.0
        for k, v in zip(weight_keys, vals):
            fixed[k] = float(v / s)

    return fixed
