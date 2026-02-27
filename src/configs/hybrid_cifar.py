from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class HybridCifarConfig:
    # 基本 FL 配置
    num_clients: int = 10
    num_rounds: int = 8
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 0.001
    alpha: float = 0.5
    fraction_fit: float = 0.5
    seed: int = 42
    data_dir: str = f"{project_root}/data"

    # 半同步与异步
    semi_sync_wait_ratio: float = 0.7
    fedbuff_buffer_size: int = 16
    fedbuff_min_updates_to_aggregate: int = 8
    staleness_alpha: float = 1.0
    max_staleness: int = 8
    async_agg_interval: int = 2

    # 调度与公平性
    selection_top_k: int = 0  # 若<=0，则使用 int(num_clients*fraction_fit)
    fair_window_size: int = 4  # Jain 公平性滑动窗口

    # 门控切换相关
    gate_to_async: float = 0.58
    gate_to_semi_sync: float = 0.42
    hysteresis_margin: float = 0.03
    bridge_rounds: int = 2
    min_rounds_between_switch: int = 2
    w_per: float = 0.5
    w_fair: float = 0.3
    w_energy: float = 0.2
    window_size: int = 4  # 门控历史滑窗长度

    # 调度评分权重
    channel_w: float = 0.25
    data_w: float = 0.25
    fair_w: float = 0.25
    energy_w: float = 0.15
    bwcost_w: float = 0.10

    # FedProx
    fedprox_mu: float = 0.0
    algorithm: str = "fedavg"  # fedavg / fedprox / scaffold

    # 客户端能量模型
    initial_client_energy: float = 100.0
    client_initial_energies: Optional[List[float]] = None

    # 策略嵌套配置（与 docs 一致的层级）
    strategy: Dict[str, object] = field(
        default_factory=lambda: {
            "window_size": 4,
            "gate_thresholds": {"to_async": 0.58, "to_semi_sync": 0.42},
            "hysteresis_margin": 0.03,
            "bridge_rounds": 2,
            "min_rounds_between_switch": 2,
            "gate_weights": {"per": 0.3, "fairness": 0.4, "energy": 0.3},
            "bandwidth_rebalance": {"low_energy_factor": 0.8, "high_energy_factor": 1.0},
            "scheduling": {
                "weights": {
                    "channel_w": 0.25,
                    "data_w": 0.25,
                    "fair_w": 0.25,
                    "energy_w": 0.15,
                    "bwcost_w": 0.10,
                }
            }
        }
    )

    wireless: Dict[str, object] = field(
        default_factory=lambda: {
            "channel_model": "tr38901_umi",
            "block_fading_intensity": 1.0,
            "base_snr_db": 8.0,
            "per_k": 1.0,
            "bandwidth_budget_mb_per_round": 12.0,
            "tx_power_watts": 1.0,
            "compute_power_watts": 8.0,
            "compute_rate_samples_per_sec": 2000,
        }
    )
