from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "name": "cifar10",
        "data_dir": "./data",
        "alpha": 0.5,
        "batch_size": 128,
    },
    "fl": {
        "num_clients": 10,
        "num_rounds": 8,
        "local_epochs": 1,
        "lr": 0.001,
        "fraction_fit": 0.5,
        "seed": 42,
        "client_resources": {"num_cpus": 2, "num_gpus": 0.25},
    },
    "wireless": {
        "wireless_model": "simulated",
        "simulated_mode": "good",  # good / bad / jitter
        "jitter_period_rounds": 20,
        "jitter_start_state": "good",
        "wsn_csv_path": "",
        "wsn_snr_col": "snr",
        "wsn_rssi_col": "rssi",
        "wsn_noise_col": "noise_floor",
        "wsn_prr_col": "prr",
        "wsn_per_col": "per",
        "channel_model": "tr38901_umi",
        "block_fading_intensity": 1.0,
        "base_snr_db": 8.0,
        "per_k": 1.0,
        "bandwidth_budget_mb_per_round": 12.0,
        "tx_power_watts": 1.0,
        "compute_power_watts": 8.0,
        "compute_rate_samples_per_sec": 2000,
    },
    "algorithm": {
        "name": "fedavg",
        "fedprox_mu": 0.0,
    },
    "energy": {
        "initial_client_energy": 100.0,
        "client_initial_energies": None,
    },
    "scheduler": {
        "selection_top_k": 0,
        "fair_window_size": 4,
        "weights": {
            "channel_w": 0.25,
            "data_w": 0.25,
            "fair_w": 0.25,
            "energy_w": 0.15,
        },
    },
    "controller": {
        "semi_sync_wait_ratio": 0.7,
        "window_size": 4,
        "gate_thresholds": {"to_async": 0.58, "to_semi_sync": 0.42},
        "hysteresis_margin": 0.03,
        "bridge_rounds": 2,
        "min_rounds_between_switch": 2,
        "gate_weights": {"per": 0.5, "fairness": 0.3, "energy": 0.2},
        "bandwidth_rebalance": {"low_energy_factor": 0.8, "high_energy_factor": 1.0},
        "mode_policy": "hybrid",
    },
    "fedbuff": {
        "buffer_size": 16,
        "min_updates_to_aggregate": 8,
        "staleness_alpha": 1.0,
        "max_staleness": 8,
        "async_agg_interval": 2,
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_strategy_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(DEFAULTS)
    _deep_update(cfg, loaded)
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    if args.num_clients is not None:
        out["fl"]["num_clients"] = int(args.num_clients)
    if args.num_rounds is not None:
        out["fl"]["num_rounds"] = int(args.num_rounds)
    if args.local_epochs is not None:
        out["fl"]["local_epochs"] = int(args.local_epochs)
    if args.batch_size is not None:
        out["dataset"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        out["fl"]["lr"] = float(args.lr)
    if args.alpha is not None:
        out["dataset"]["alpha"] = float(args.alpha)
    if args.fraction_fit is not None:
        out["fl"]["fraction_fit"] = float(args.fraction_fit)
    if args.seed is not None:
        out["fl"]["seed"] = int(args.seed)
    if args.data_dir:
        out["dataset"]["data_dir"] = str(args.data_dir)
    if args.algorithm:
        out["algorithm"]["name"] = str(args.algorithm)
    if args.fedprox_mu is not None:
        out["algorithm"]["fedprox_mu"] = float(args.fedprox_mu)
    if args.wireless_model:
        out["wireless"]["wireless_model"] = str(args.wireless_model)
    if args.simulated_mode:
        out["wireless"]["simulated_mode"] = str(args.simulated_mode)
    if args.jitter_period_rounds is not None:
        out["wireless"]["jitter_period_rounds"] = int(args.jitter_period_rounds)
    if args.jitter_start_state:
        out["wireless"]["jitter_start_state"] = str(args.jitter_start_state)
    if args.wsn_csv_path:
        out["wireless"]["wsn_csv_path"] = str(args.wsn_csv_path)
    if args.selection_top_k is not None:
        out["scheduler"]["selection_top_k"] = int(args.selection_top_k)
    if args.async_agg_interval is not None:
        out["fedbuff"]["async_agg_interval"] = int(args.async_agg_interval)
    if args.fair_window_size is not None:
        out["scheduler"]["fair_window_size"] = int(args.fair_window_size)
    if args.semi_sync_wait_ratio is not None:
        out["controller"]["semi_sync_wait_ratio"] = float(args.semi_sync_wait_ratio)
    if args.initial_client_energy is not None:
        out["energy"]["initial_client_energy"] = float(args.initial_client_energy)
    if args.client_initial_energies:
        energies = [float(x.strip()) for x in args.client_initial_energies.split(",") if x.strip()]
        out["energy"]["client_initial_energies"] = energies
    if args.client_num_cpus is not None:
        out["fl"]["client_resources"]["num_cpus"] = float(args.client_num_cpus)
    if args.client_num_gpus is not None:
        out["fl"]["client_resources"]["num_gpus"] = float(args.client_num_gpus)
    return out


def default_strategy_yaml(strategy_name: str) -> str:
    base = os.path.join("src", "configs", "strategies")
    return os.path.join(base, f"{strategy_name}.yaml")
