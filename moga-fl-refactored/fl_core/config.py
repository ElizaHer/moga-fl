from __future__ import annotations

from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file into a Python dict.

    保持与原工程相同的配置结构，便于迁移已有 YAML 文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg


def merge_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge safe defaults into a raw config dict.

    该函数直接拷贝自原工程的 `src/configs/config.py`，并做轻微整理，
    以保证在字段缺省时不会出现 KeyError，同时为后续 FedLab 集成预留参数位。
    """
    # 数据集相关
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("root", "data")

    # 模型配置：支持 small_cnn / resnet18_cifar / resnet18_emnist
    cfg.setdefault("model", {})
    cfg["model"].setdefault("type", "small_cnn")
    cfg["model"].setdefault("width_factor", 1.0)

    # 客户端与调度
    cfg.setdefault("clients", {})
    cfg["clients"].setdefault("num_clients", 10)
    cfg["clients"].setdefault(
        "selection_top_k", max(1, cfg["clients"]["num_clients"] // 2)
    )

    # 无线与能耗配置
    cfg.setdefault("wireless", {})
    cfg["wireless"].setdefault("channel_model", "rayleigh_siso")
    cfg["wireless"].setdefault("carrier_ghz", 3.5)

    # 训练相关
    cfg.setdefault("training", {})
    # 初始同步模式：sync / semi_sync / async
    if "sync_mode" not in cfg["training"]:
        # 兼容旧版 boolean sync 字段
        if cfg["training"].get("sync", True):
            cfg["training"]["sync_mode"] = "semi_sync"
        else:
            cfg["training"]["sync_mode"] = "async"

    # 训练算法：fedavg / fedprox / scaffold
    cfg["training"].setdefault("algorithm", "fedavg")

    cfg.setdefault("compression", {})

    # 调度与公平债务
    cfg.setdefault("scheduling", {})
    cfg["scheduling"].setdefault(
        "weights",
        {
            "energy": 0.25,
            "channel": 0.25,
            "data_value": 0.25,
            "fairness_debt": 0.2,
            "bandwidth_cost": -0.15,
        },
    )
    cfg["scheduling"].setdefault(
        "fairness_ledger",
        {"debt_increase": 0.05, "repay_rate": 0.1, "max_debt": 1.0},
    )

    # 策略控制器：门控 + 桥接态
    strat = cfg.setdefault("strategy", {})
    strat.setdefault("window_size", 5)
    strat.setdefault(
        "gate_thresholds", {"to_async": 0.6, "to_semi_sync": 0.4}
    )
    strat.setdefault("hysteresis_margin", 0.05)
    strat.setdefault("bridge_rounds", 3)
    strat.setdefault("min_rounds_between_switch", 5)
    strat.setdefault("weights", {"per": 0.5, "fairness": 0.3, "energy": 0.2})
    strat.setdefault(
        "bandwidth_rebalance",
        {"low_energy_factor": 0.8, "high_energy_factor": 1.0},
    )

    # 评估与随机种子
    cfg.setdefault(
        "eval",
        {
            "rounds": 5,
            "test_interval": 1,
            "seed": 42,
            "save_csv": True,
            "save_plots": True,
            "preference": "time",
        },
    )

    return cfg
