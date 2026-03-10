from __future__ import annotations

from typing import Any, Callable, Dict, Union

import copy
import os

import numpy as np
import pandas as pd

from src.configs.strategy_runtime import load_strategy_yaml
from src.flower.hybrid_opt_demo import run_hybrid_flower_cifar


CfgLike = Union[str, Dict[str, Any]]


def _load_base_cfg(cfg_path_or_dict: CfgLike) -> Dict[str, Any]:
    """Load and normalize strategy config.

    - If given a path, use ``load_strategy_yaml`` (which applies defaults).
    - If given a dict, deep-copy it to avoid in-place mutations across runs.
    """
    if isinstance(cfg_path_or_dict, str):
        return load_strategy_yaml(cfg_path_or_dict)
    if isinstance(cfg_path_or_dict, dict):
        return copy.deepcopy(cfg_path_or_dict)
    raise TypeError(f"Unsupported cfg type: {type(cfg_path_or_dict)!r}")


def _aggregate_metrics_csv(csv_path: str) -> Dict[str, float]:
    """Read HybridWirelessStrategy metrics CSV and aggregate GA objectives.

    The CSV is produced by ``HybridWirelessStrategy._log_metrics`` and contains
    per-round columns such as ``accuracy``, ``jain``, ``energy``, and
    ``est_upload_time``. We take simple means as low-dimensional GA objectives.
    """
    if not os.path.exists(csv_path):
        # Defensive: return neutral metrics rather than crashing the optimizer
        return {"acc": 0.0, "time": 0.0, "energy": 0.0, "comm_cost": 0.0, "fairness": 0.0}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"acc": 0.0, "time": 0.0, "energy": 0.0, "comm_cost": 0.0, "fairness": 0.0}

    # Column names come from HybridWirelessStrategy._init_metrics_file
    acc = float(df["accuracy"].mean()) if "accuracy" in df.columns else 0.0
    time = (
        float(df["est_upload_time"].mean())
        if "est_upload_time" in df.columns
        else 0.0
    )
    energy = float(df["energy"].mean()) if "energy" in df.columns else 0.0
    fairness = float(df["jain"].mean()) if "jain" in df.columns else 0.0
    # Communication-cost proxy: average active clients * effective payload.
    topk_mean = float(df["topk"].mean()) if "topk" in df.columns else 0.0
    payload_mb = float(df.get("payload_mb", pd.Series([1.0])).mean()) if "payload_mb" in df.columns else 1.0
    comm_cost = topk_mean * payload_mb
    return {"acc": acc, "time": time, "energy": energy, "comm_cost": comm_cost, "fairness": fairness}


def make_flower_sim_runner(
    cfg_path_or_dict: CfgLike,
    strategy_name: str = "hybrid_opt",
    round_scale: float = 0.5,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """Construct a Flower-based simulation runner for GA evaluation.

    Parameters
    ----------
    cfg_path_or_dict:
        Either a path to a strategy YAML (compatible with
        ``load_strategy_yaml``) or an in-memory config dict.
    strategy_name:
        Name passed to ``build_strategy`` / ``HybridWirelessStrategy`` factory.
    round_scale:
        Scale factor applied to ``cfg["fl"]["num_rounds"]``.
        - 0.5  → low-fidelity (fewer rounds)
        - 1.0  → high-fidelity (full rounds)

    Returns
    -------
    runner(params) -> metrics
        ``params`` is a GA individual with keys:
        ``energy_w, channel_w, data_w, fair_w, selection_top_k, bandwidth_alloc_factor,``
        ``staleness_alpha, bridge_to_async, bridge_to_semi_sync, compression_ratio``.
        The returned ``metrics`` dict matches the GA objective interface:
        ``{"acc", "time", "energy", "comm_cost", "fairness"}``.
    """

    base_cfg = _load_base_cfg(cfg_path_or_dict)
    # Ensure strategy_name is present for logging directory naming
    base_cfg = dict(base_cfg)
    base_cfg["strategy_name"] = strategy_name

    fl_cfg = base_cfg.get("fl", {}) if isinstance(base_cfg.get("fl", {}), dict) else {}
    base_rounds = int(fl_cfg.get("num_rounds", 8))
    scale = float(round_scale) if round_scale > 0 else 0.5

    def runner(params: Dict[str, Any]) -> Dict[str, float]:
        cfg = copy.deepcopy(base_cfg)
        fl = cfg.get("fl", {}) if isinstance(cfg.get("fl", {}), dict) else {}
        eval_rounds = max(1, int(np.round(base_rounds * scale)))
        fl["num_rounds"] = int(eval_rounds)
        cfg["fl"] = fl

        # Scheduler weights and selection_top_k
        scheduler = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler", {}), dict) else {}
        weights = scheduler.get("weights", {}) if isinstance(scheduler.get("weights", {}), dict) else {}
        weights["energy_w"] = float(params.get("energy_w", weights.get("energy_w", 0.15)))
        weights["channel_w"] = float(params.get("channel_w", weights.get("channel_w", 0.25)))
        weights["data_w"] = float(params.get("data_w", weights.get("data_w", 0.25)))
        weights["fair_w"] = float(params.get("fair_w", weights.get("fair_w", 0.25)))
        scheduler["weights"] = weights
        top_k = int(params.get("selection_top_k", scheduler.get("selection_top_k", 0)))
        scheduler["selection_top_k"] = int(max(1, top_k))
        cfg["scheduler"] = scheduler

        # Wireless knobs
        wireless = cfg.get("wireless", {}) if isinstance(cfg.get("wireless", {}), dict) else {}
        base_bw = float(wireless.get("bandwidth_budget_mb_per_round", 12.0))
        bw_factor = float(params.get("bandwidth_alloc_factor", 1.0))
        wireless["bandwidth_budget_mb_per_round"] = float(max(1e-3, base_bw * bw_factor))
        wireless["payload_compression_ratio"] = float(params.get("compression_ratio", wireless.get("payload_compression_ratio", 1.0)))
        cfg["wireless"] = wireless

        # FedBuff staleness_alpha (top-level for Flower config)
        fedbuff = cfg.get("fedbuff", {}) if isinstance(cfg.get("fedbuff", {}), dict) else {}
        fedbuff["staleness_alpha"] = float(
            params.get("staleness_alpha", fedbuff.get("staleness_alpha", 1.0))
        )
        cfg["fedbuff"] = fedbuff

        # Controller bridge thresholds
        controller = cfg.get("controller", {}) if isinstance(cfg.get("controller", {}), dict) else {}
        gates = controller.get("gate_thresholds", {}) if isinstance(controller.get("gate_thresholds", {}), dict) else {}
        gates["to_async"] = float(params.get("bridge_to_async", gates.get("to_async", 0.58)))
        gates["to_semi_sync"] = float(params.get("bridge_to_semi_sync", gates.get("to_semi_sync", 0.42)))
        controller["gate_thresholds"] = gates
        cfg["controller"] = controller

        # Run Flower simulation and aggregate metrics from the strategy CSV
        result = run_hybrid_flower_cifar(cfg, strategy_name=strategy_name)
        csv_path = str(result.get("metrics_csv", ""))
        return _aggregate_metrics_csv(csv_path)

    return runner
