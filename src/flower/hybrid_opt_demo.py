from __future__ import annotations

import argparse
import csv
import datetime
import io
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.common import (
    Context,
    NDArrays,
    Parameters,
    Scalar,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from src.training.algorithms.fedprox import fedprox_regularizer
from src.training.algorithms.scaffold import ScaffoldState
from src.training.models.resnet_cifar import build_resnet18_cifar  # type: ignore
from src.wireless.channel import ChannelSimulator  # type: ignore
from src.wireless.bandwidth import BandwidthAllocator  # type: ignore
from src.wireless.energy import EnergyEstimator  # type: ignore


# -----------------------------
# 配置与工具函数
# -----------------------------
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
            },
        }
    )


def _base_wireless_cfg() -> Dict[str, object]:
    return {
        "wireless": {
            "channel_model": "tr38901_umi",
            "block_fading_intensity": 1.0,
            "base_snr_db": 8.0,
            "per_k": 1.0,
            "bandwidth_budget_mb_per_round": 12.0,
            "tx_power_watts": 1.0,
            "compute_power_watts": 8.0,
            "compute_rate_samples_per_sec": 2000,
        }
    }


def get_parameters(model: nn.Module) -> NDArrays:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def _global_state_from_ndarrays(model: nn.Module, parameters: NDArrays, device: torch.device) -> Dict[str, torch.Tensor]:
    keys = list(model.state_dict().keys())
    return {
        k: torch.tensor(v, device=device)
        for k, v in zip(keys, parameters)
    }


def _pack_tensor_dict(tensors: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in tensors.items()}
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def _unpack_tensor_dict(payload: bytes, device: torch.device) -> Dict[str, torch.Tensor]:
    if not payload:
        return {}
    buffer = io.BytesIO(payload)
    obj = torch.load(buffer, map_location=device)
    return {k: v.to(device) for k, v in obj.items()}


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return float(avg_loss), float(accuracy)


def dirichlet_partition_indices(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    num_classes = int(labels.max()) + 1
    for c in range(num_classes):
        class_ids = np.where(labels == c)[0]
        rng.shuffle(class_ids)
        proportions = rng.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_ids)).astype(int)[:-1]
        splits = np.split(class_ids, split_points)
        for cid, idx in enumerate(splits):
            client_indices[cid].extend(idx.tolist())
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def _jain_index(values: List[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or np.allclose(x.sum(), 0.0):
        return 0.0
    return float((x.sum() ** 2) / (len(x) * np.sum(x * x) + 1e-12))


# -----------------------------
# FedBuff 状态（按文档修复 aging 与触发条件）
# -----------------------------


class FedBuffState:
    def __init__(
        self,
        alpha: float,
        max_staleness: int,
        buffer_size: int,
        min_updates: int,
        async_agg_interval: int,
    ) -> None:
        self.alpha = alpha
        self.max_staleness = max_staleness
        self.buffer_size = buffer_size
        self.min_updates = min_updates
        self.async_agg_interval = async_agg_interval

        # entries: (params, staleness, num_examples)
        self.entries: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        self.updates_since_last_agg = 0
        self.last_agg_round = 0
        self.last_age_round = 0

    def age(self, current_round: int) -> None:
        """按轮次差值 old->new 进行 aging，避免重复过度老化。"""
        if current_round <= self.last_age_round:
            return
        delta = current_round - self.last_age_round
        if delta <= 0:
            return
        aged: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        for params, staleness, num_examples in self.entries:
            s = staleness + delta
            if s <= self.max_staleness:
                aged.append((params, s, num_examples))
        self.entries = aged
        self.last_age_round = current_round

    def push(self, params: List[np.ndarray], num_examples: int) -> None:
        self.entries.append((params, 0, num_examples))
        self.updates_since_last_agg += 1

    def should_aggregate(self, server_round: int) -> bool:
        if len(self.entries) >= self.buffer_size:
            return True
        if self.updates_since_last_agg >= self.min_updates:
            return True
        if (
            self.async_agg_interval > 0
            and server_round - self.last_agg_round >= self.async_agg_interval
            and len(self.entries) > 0
        ):
            return True
        return False

    def aggregate(self, fallback_params: List[np.ndarray], server_round: int) -> List[np.ndarray]:
        if not self.entries:
            return fallback_params

        sum_weights = 0.0
        agg = [arr.copy() for arr in fallback_params]
        for i in range(len(agg)):
            if np.issubdtype(agg[i].dtype, np.floating):
                agg[i] = np.zeros_like(agg[i], dtype=np.float32)

        for params, staleness, num_examples in self.entries:
            staleness_weight = 1.0 / ((1.0 + float(staleness)) ** self.alpha)
            w = staleness_weight * float(num_examples)
            sum_weights += w
            for idx, arr in enumerate(params):
                if np.issubdtype(agg[idx].dtype, np.floating):
                    agg[idx] += w * arr.astype(np.float32, copy=False)

        out: List[np.ndarray] = []
        for arr in agg:
            if np.issubdtype(arr.dtype, np.floating):
                out.append(arr / max(sum_weights, 1e-12))
            else:
                out.append(arr)

        self.entries.clear()
        self.updates_since_last_agg = 0
        self.last_agg_round = server_round
        return out


# -----------------------------
# 模式控制器：门控 + 带宽再平衡
# -----------------------------


class ModeController:
    """根据 avg_per/Jain/能耗的滑窗统计，输出 (mode, bridge_weight, bandwidth_factor)。"""

    def __init__(self, cfg: HybridCifarConfig, num_clients: int) -> None:
        self.cfg = cfg
        self.num_clients = num_clients

        strat = cfg.strategy if isinstance(cfg.strategy, dict) else {}
        gates = strat.get("gate_thresholds", {}) if isinstance(strat.get("gate_thresholds", {}), dict) else {}
        gate_weights = strat.get("gate_weights", {}) if isinstance(strat.get("gate_weights", {}), dict) else {}
        bw_cfg = (
            strat.get("bandwidth_rebalance", {})
            if isinstance(strat.get("bandwidth_rebalance", {}), dict)
            else {}
        )

        self.window_size = int(strat.get("window_size", cfg.window_size))
        self.to_async = float(gates.get("to_async", cfg.gate_to_async))
        self.to_semi = float(gates.get("to_semi_sync", cfg.gate_to_semi_sync))
        self.hysteresis_margin = float(strat.get("hysteresis_margin", cfg.hysteresis_margin))
        self.bridge_rounds = int(strat.get("bridge_rounds", cfg.bridge_rounds))
        self.min_rounds_between_switch = int(
            strat.get("min_rounds_between_switch", cfg.min_rounds_between_switch)
        )

        self.w_per = float(gate_weights.get("per", cfg.w_per))
        self.w_fair = float(gate_weights.get("fairness", cfg.w_fair))
        self.w_energy = float(gate_weights.get("energy", cfg.w_energy))

        self.bw_low = float(bw_cfg.get("low_energy_factor", 0.8))
        self.bw_high = float(bw_cfg.get("high_energy_factor", 1.0))

        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.current_mode: str = "semi_sync"
        self.bridge_target: Optional[str] = None
        self.bridge_start: int = -1
        self.last_switch_round: int = -10**9

    def register(self, avg_per: float, jain: float, total_energy: float) -> None:
        self.history.append(
            {
                "avg_per": float(avg_per),
                "jain": float(jain),
                "energy": float(total_energy),
            }
        )

    def _gate_score(self) -> float:
        if not self.history:
            return 0.0
        avg_per = float(np.mean([row["avg_per"] for row in self.history]))
        jain = float(np.mean([row["jain"] for row in self.history]))
        energies = [row["energy"] for row in self.history]
        mean_energy = float(np.mean(energies))
        max_energy = float(max(energies)) if energies else 0.0
        if max_energy > 0.0:
            energy_norm = float(np.clip(mean_energy / max_energy, 0.0, 1.0))
        else:
            energy_norm = 0.0
        fairness_deficit = 1.0 - jain
        score = (
            self.w_per * avg_per
            + self.w_fair * fairness_deficit
            + self.w_energy * energy_norm
        )
        print(f"avg_per: {avg_per:.4f}, fairness: {fairness_deficit:.4f}, energy: {energy_norm:.4f}, score: {score:.4f}")
        return float(score)

    def _bandwidth_factor(self) -> float:
        if not self.history:
            return self.bw_high
        energies = [row["energy"] for row in self.history]
        mean_energy = float(np.mean(energies))
        max_energy = float(max(energies)) if energies else 0.0
        if max_energy > 0.0:
            energy_norm = float(np.clip(mean_energy / max_energy, 0.0, 1.0))
        else:
            energy_norm = 0.0
        factor = self.bw_high - (self.bw_high - self.bw_low) * energy_norm
        return float(np.clip(factor, self.bw_low, self.bw_high))

    def decide(self, server_round: int) -> Tuple[str, float, float]:
        """返回 (mode, bridge_weight, bandwidth_factor)。"""
        bw_factor = self._bandwidth_factor()

        # bridge 态：只负责桥接权重随轮数演化
        if self.current_mode == "bridge":
            t = max(0, server_round - self.bridge_start)
            w = float(np.clip(t / max(1, self.bridge_rounds), 0.0, 1.0))
            if t >= self.bridge_rounds and self.bridge_target is not None:
                self.current_mode = self.bridge_target
                self.bridge_target = None
                self.bridge_start = -1
            return "bridge", w, bw_factor

        # 防抖：两次切换之间强制间隔
        if server_round - self.last_switch_round < self.min_rounds_between_switch:
            return self.current_mode, 0.0, bw_factor

        gate = self._gate_score()
        target = self.current_mode
        if gate >= self.to_async + self.hysteresis_margin:
            target = "async"
        elif gate <= self.to_semi - self.hysteresis_margin:
            target = "semi_sync"

        if target != self.current_mode:
            # 进入桥接态
            self.current_mode = "bridge"
            self.bridge_target = target
            self.bridge_start = server_round
            self.last_switch_round = server_round
            return "bridge", 0.0, bw_factor

        return self.current_mode, 0.0, bw_factor


# -----------------------------
# 客户端实现
# -----------------------------


class CifarClient(fl.client.NumPyClient):  # type: ignore[misc]
    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        local_epochs: int,
        lr: float,
    ) -> None:
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_cifar().to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        # T_max 取一个相对较大的值，近似余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.scaffold_ci: Dict[str, torch.Tensor] | None = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:  # type: ignore[override]
        del config
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:  # type: ignore[override]
        set_parameters(self.model, parameters)
        algo = str(config.get("algorithm", "fedavg")).lower()
        mu = float(config.get("fedprox_mu", 0.0))
        criterion = nn.CrossEntropyLoss()
        global_state = _global_state_from_ndarrays(self.model, parameters, self.device)
        delta_ci_bytes = b""

        # SCAFFOLD: 读取 c_global，若首次出现则初始化本地 c_i
        c_global: Dict[str, torch.Tensor] = {}
        if algo == "scaffold":
            c_global = _unpack_tensor_dict(bytes(config.get("scaffold_c_global", b"")), self.device)
            if self.scaffold_ci is None:
                self.scaffold_ci = {k: torch.zeros_like(v) for k, v in c_global.items()}

        self.model.train()
        num_steps = 0
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                if algo == "fedprox" and mu > 0.0:
                    loss = loss + fedprox_regularizer(self.model, global_state, mu, self.device)

                loss.backward()
                if algo == "scaffold" and self.scaffold_ci is not None:
                    # SCAFFOLD 梯度校正: grad <- grad + (c_i - c_global)
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        ci = self.scaffold_ci.get(name)
                        cg = c_global.get(name)
                        if ci is not None and cg is not None:
                            p.grad.data = p.grad.data + (ci - cg)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                num_steps += 1

        if algo == "scaffold" and self.scaffold_ci is not None:
            local_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            lr = float(self.optimizer.param_groups[0].get("lr", 0.001))
            scale = 1.0 / max(1, int(num_steps)) / max(1e-8, lr)
            delta_ci: Dict[str, torch.Tensor] = {}
            new_ci: Dict[str, torch.Tensor] = {}
            for k in c_global.keys():
                old_ci = self.scaffold_ci[k]
                w_g = global_state[k]
                w_l = local_state[k].to(self.device)
                c_g = c_global[k]
                ci_new = old_ci - c_g + (w_g - w_l) * scale
                delta_ci[k] = ci_new - old_ci
                new_ci[k] = ci_new
            self.scaffold_ci = new_ci
            delta_ci_bytes = _pack_tensor_dict(delta_ci)

        return get_parameters(self.model), len(self.trainloader.dataset), {
            "cid": float(self.cid),
            "delta_ci": delta_ci_bytes,
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:  # type: ignore[override]
        del config
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


# -----------------------------
# 服务器策略：调度 + 半同步/异步/桥接聚合
# -----------------------------


class HybridWirelessStrategy(Strategy):  # type: ignore[misc]
    def __init__(
        self,
        cfg: HybridCifarConfig,
        partition_sizes: List[int],
        testloader: DataLoader,
    ) -> None:
        self.cfg = cfg
        self.partition_sizes = partition_sizes
        self.testloader = testloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_for_eval = build_resnet18_cifar().to(self.device)

        self.controller = ModeController(cfg, cfg.num_clients)
        self.fedbuff = FedBuffState(
            alpha=cfg.staleness_alpha,
            max_staleness=cfg.max_staleness,
            buffer_size=cfg.fedbuff_buffer_size,
            min_updates=cfg.fedbuff_min_updates_to_aggregate,
            async_agg_interval=cfg.async_agg_interval,
        )

        wireless_cfg = _base_wireless_cfg()
        self.channel = ChannelSimulator(wireless_cfg, cfg.num_clients)
        self.bw = BandwidthAllocator(wireless_cfg)
        self.energy = EnergyEstimator(wireless_cfg)

        base_budget = getattr(self.bw, "budget_mb", None)
        self._base_bandwidth_budget: Optional[float] = float(base_budget) if base_budget is not None else None

        self.global_params_cache: Optional[Parameters] = None
        self.scaffold_state: Optional[ScaffoldState] = None

        # 调度和公平性统计
        self.selection_window: Deque[List[int]] = deque(maxlen=self.cfg.fair_window_size)
        self.last_selected_cids: List[int] = []
        self.current_wireless_stats: Optional[Dict[int, Dict[str, float]]] = None

        # 模式与带宽
        self.current_mode: str = "semi_sync"
        self.current_bridge_weight: float = 0.0
        self.current_bw_factor: float = 1.0
        self.last_topk: int = 0

        # 调度评分权重
        strat = cfg.strategy if isinstance(cfg.strategy, dict) else {}
        sched_cfg = strat.get("scheduling", {}) if isinstance(strat.get("scheduling", {}), dict) else {}
        weights = sched_cfg.get("weights", {}) if isinstance(sched_cfg.get("weights", {}), dict) else {}
        self.channel_w = float(weights.get("channel_w", cfg.channel_w))
        self.data_w = float(weights.get("data_w", cfg.data_w))
        self.fair_w = float(weights.get("fair_w", cfg.fair_w))
        self.energy_w = float(weights.get("energy_w", cfg.energy_w))
        self.bwcost_w = float(weights.get("bwcost_w", cfg.bwcost_w))

        # 历史指标
        self.mode_history: List[str] = []
        self.acc_history: List[float] = []
        self.energy_history: List[float] = []
        self.avg_per_history: List[float] = []
        self.jain_history: List[float] = []

        # metrics.csv
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_path = os.path.join("outputs/hybrid_metrics", f"{self.cfg.algorithm}_{current_time}.csv")
        print(f"Result file: {self.metrics_path}")
        self._init_metrics_file()

    # --- 工具方法 ---

    def _init_metrics_file(self) -> None:
        os.makedirs(os.path.dirname(self.metrics_path) or ".", exist_ok=True)
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "round",
                        "mode",
                        "accuracy",
                        "loss",
                        "avg_per",
                        "jain",
                        "energy",
                        "bw_factor",
                        "topk",
                    ]
                )

    def _log_metrics(
        self,
        server_round: int,
        mode: str,
        accuracy: float,
        loss: float,
        avg_per: float,
        jain: float,
        energy: float,
        bw_factor: float,
        topk: int,
    ) -> None:
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    int(server_round),
                    mode,
                    float(accuracy),
                    float(loss),
                    float(avg_per),
                    float(jain),
                    float(energy),
                    float(bw_factor),
                    int(topk),
                ]
            )

    def _compute_selection_counts(self) -> np.ndarray:
        counts = np.zeros(self.cfg.num_clients, dtype=np.float64)
        for cids in self.selection_window:
            for cid in cids:
                if 0 <= cid < self.cfg.num_clients:
                    counts[cid] += 1.0
        return counts

    def _compute_fairness_debt(self) -> np.ndarray:
        counts = self._compute_selection_counts()
        if self.selection_window and counts.size > 0:
            mean = float(counts.mean())
            if mean > 0.0:
                debt = (mean - counts) / mean
                debt = np.clip(debt, 0.0, 1.0)
            else:
                debt = np.ones_like(counts)
        else:
            debt = np.ones(self.cfg.num_clients, dtype=np.float64)
        return debt

    def _update_bandwidth_budget(self) -> None:
        if self._base_bandwidth_budget is not None:
            self.bw.budget_mb = self._base_bandwidth_budget * float(self.current_bw_factor)

    def _weighted_avg(
        self, pairs: List[Tuple[List[np.ndarray], float]], fallback: List[np.ndarray]
    ) -> List[np.ndarray]:
        if not pairs:
            return fallback
        total = float(sum(w for _, w in pairs))
        base = [arr.copy() for arr in fallback]
        for i in range(len(base)):
            if np.issubdtype(base[i].dtype, np.floating):
                base[i] = np.zeros_like(base[i], dtype=np.float32)
        for params, weight in pairs:
            scale = weight / max(total, 1e-12)
            for i, arr in enumerate(params):
                if np.issubdtype(base[i].dtype, np.floating):
                    base[i] += scale * arr.astype(np.float32, copy=False)
        return base

    # --- Strategy 接口实现 ---

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:  # type: ignore[override]
        del client_manager
        model = build_resnet18_cifar()
        ndarrays = get_parameters(model)
        self.global_params_cache = ndarrays_to_parameters(ndarrays)
        if self.cfg.algorithm.lower() == "scaffold":
            global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.scaffold_state = ScaffoldState(global_state)
        return self.global_params_cache

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ):
        # 1) 门控决策 + 带宽因子
        mode, bridge_w, bw_factor = self.controller.decide(server_round)
        self.current_mode = mode
        self.current_bridge_weight = bridge_w
        self.current_bw_factor = bw_factor
        self.mode_history.append(mode)

        # 2) Top-K 数量
        if self.cfg.selection_top_k > 0:
            top_k = min(self.cfg.selection_top_k, self.cfg.num_clients)
        else:
            top_k = max(1, int(round(self.cfg.num_clients * self.cfg.fraction_fit)))
        self.last_topk = top_k

        # 3) 获取当前可用客户端
        if hasattr(client_manager, "all"):
            all_clients = client_manager.all()
            if isinstance(all_clients, dict):
                available_list = list(all_clients.values())
            else:
                available_list = list(all_clients)
        else:
            sampled_clients = client_manager.sample(
                num_clients=self.cfg.num_clients,
                min_num_clients=self.cfg.num_clients,
            )
            available_list = list(sampled_clients)

        cid_to_client: Dict[int, object] = {i: available_list[i] for i in range(self.cfg.num_clients)}

        if not cid_to_client:
            # 回退为随机采样
            print(f"Warning: No valid cid found for {top_k} clients. Falling back to random sampling.")
            clients = client_manager.sample(
                num_clients=max(1, top_k),
                min_num_clients=max(1, top_k),
            )
            fit_cfg: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "fedprox_mu": float(self.cfg.fedprox_mu),
                "algorithm": str(self.cfg.algorithm),
            }
            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                fit_cfg["scaffold_c_global"] = _pack_tensor_dict(self.scaffold_state.c_global)
            fit_ins = FitIns(parameters, fit_cfg)
            return [(client, fit_ins) for client in clients]
        
        print(f"selection window size: {len(self.selection_window)}")

        # 4) 信道抽样 + 带宽预算更新
        self.current_wireless_stats = self.channel.sample_round()
        self._update_bandwidth_budget()

        # 5) 预估各客户端的数据量/能耗/带宽成本
        num_clients = self.cfg.num_clients
        data_sizes = np.asarray(self.partition_sizes, dtype=np.float64)
        if data_sizes.size != num_clients:
            # 防御性处理
            if data_sizes.size < num_clients:
                data_sizes = np.pad(data_sizes, (0, num_clients - data_sizes.size), mode="edge")
            else:
                data_sizes = data_sizes[:num_clients]
        data_norm = data_sizes / max(float(data_sizes.max()), 1e-12)

        fairness_debt = self._compute_fairness_debt()

        # 按所有客户端分配带宽，用于近似预测 bwcost/energy
        bw_map_all = self.bw.allocate_uniform(list(range(num_clients)))

        channel_score = np.zeros(num_clients, dtype=np.float64)
        energy_arr = np.zeros(num_clients, dtype=np.float64)
        bwcost_arr = np.zeros(num_clients, dtype=np.float64)

        for cid in range(num_clients):
            stats = (self.current_wireless_stats or {}).get(cid)
            if stats is None:
                continue
            per = float(stats.get("per", 0.0))
            channel_score[cid] = 1.0 - per
            allocated_mb = float(bw_map_all.get(cid, 0.0))
            tx_time = self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=allocated_mb)
            bwcost_arr[cid] = float(tx_time)
            total_energy = self.energy.comm_energy(tx_time) + self.energy.compute_energy(int(data_sizes[cid]))
            energy_arr[cid] = float(total_energy)

        max_energy = float(energy_arr.max()) if energy_arr.size > 0 else 0.0
        energy_norm = energy_arr / max(max_energy, 1e-12) if max_energy > 0.0 else np.zeros_like(energy_arr)
        max_bwcost = float(bwcost_arr.max()) if bwcost_arr.size > 0 else 0.0
        bwcost_norm = (
            bwcost_arr / max(max_bwcost, 1e-12)
            if max_bwcost > 0.0
            else np.zeros_like(bwcost_arr)
        )

        # 6) 调度评分：信道 + 数据 + 公平债务 + 能耗 + 带宽成本
        scores = np.zeros(num_clients, dtype=np.float64)
        for cid in range(num_clients):
            scores[cid] = (
                self.channel_w * channel_score[cid]
                + self.data_w * data_norm[cid]
                + self.fair_w * fairness_debt[cid]
                + self.energy_w * (1.0 - energy_norm[cid])
                + self.bwcost_w * (1.0 - bwcost_norm[cid])
            )

        candidates = sorted(cid_to_client.keys())
        if not candidates:
            clients = client_manager.sample(
                num_clients=max(1, top_k),
                min_num_clients=max(1, top_k),
            )
            fit_cfg: Dict[str, Scalar] = {
                "server_round": float(server_round),
                "fedprox_mu": float(self.cfg.fedprox_mu),
                "algorithm": str(self.cfg.algorithm),
            }
            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                fit_cfg["scaffold_c_global"] = _pack_tensor_dict(self.scaffold_state.c_global)
            fit_ins = FitIns(parameters, fit_cfg)
            return [(client, fit_ins) for client in clients]

        candidates_sorted = sorted(candidates, key=lambda cid: scores[cid], reverse=True)
        topk = min(len(candidates_sorted), top_k)
        selected_cids = candidates_sorted[:topk]
        self.last_selected_cids = list(selected_cids)

        # 更新滑窗选择历史（供 Jain 和公平债务使用）
        self.selection_window.append(list(selected_cids))

        fit_cfg: Dict[str, Scalar] = {
            "server_round": float(server_round),
            "fedprox_mu": float(self.cfg.fedprox_mu),
            "algorithm": str(self.cfg.algorithm),
        }
        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
            fit_cfg["scaffold_c_global"] = _pack_tensor_dict(self.scaffold_state.c_global)
        fit_ins = FitIns(parameters, fit_cfg)
        print(f"selected cids: {selected_cids}")
        return [(cid_to_client[cid], fit_ins) for cid in selected_cids]

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[override]
        del failures
        if self.global_params_cache is None:
            return None, {}

        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)

        scheduled_cids = list(self.last_selected_cids)
        if not scheduled_cids:
            # 回退：以本轮实际返回的 cid 作为调度集合
            scheduled_cids = [int(res.metrics.get("cid", -1)) for _, res in results]
            scheduled_cids = [cid for cid in scheduled_cids if 0 <= cid < self.cfg.num_clients]

        print(f"selected cids: {scheduled_cids}")

        wireless_stats = self.current_wireless_stats or self.channel.sample_round()

        # 当前轮的 avg_per
        per_values: List[float] = []
        for cid in scheduled_cids:
            stats = wireless_stats.get(cid)
            if stats is None:
                continue
            per_values.append(float(stats.get("per", 0.0)))
        avg_per = float(np.mean(per_values)) if per_values else 0.0

        # aging 现有 FedBuff 条目
        self.fedbuff.age(server_round)

        # 带宽预算 + tx_time 估计
        self._update_bandwidth_budget()
        bw_map = self.bw.allocate_uniform(scheduled_cids)
        tx_times: Dict[int, float] = {
            cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0)))
            for cid in scheduled_cids
        }

        threshold = float("inf")
        if scheduled_cids and self.current_mode in ("semi_sync", "bridge"):
            tx_list = [tx_times[cid] for cid in scheduled_cids]
            threshold = float(
                np.quantile(np.asarray(tx_list, dtype=np.float64), self.cfg.semi_sync_wait_ratio)
            )

        valid_sync_updates: List[Tuple[List[np.ndarray], float]] = []
        scaffold_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
        scaffold_weights: Dict[int, float] = {}
        round_energy = 0.0

        for _, fit_res in results:
            cid = int(fit_res.metrics.get("cid", -1))
            if cid < 0 or cid >= self.cfg.num_clients:
                continue
            num_examples = int(fit_res.num_examples)
            if num_examples <= 0:
                continue

            tx_time = tx_times.get(
                cid,
                self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0))),
            )
            stats = wireless_stats.get(cid, {})
            per = float(stats.get("per", 0.0))

            # 通信 + 计算能耗均计入
            round_energy += self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)

            # PER 丢包：仅影响是否使用该更新
            if np.random.rand() < per:
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight = float(num_examples)

            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                payload = fit_res.metrics.get("delta_ci", b"")
                if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
                    delta_ci = _unpack_tensor_dict(bytes(payload), torch.device("cpu"))
                    if delta_ci:
                        scaffold_deltas[cid] = delta_ci
                        scaffold_weights[cid] = weight

            # 异步路径：所有未丢包更新进入 FedBuff
            self.fedbuff.push(ndarrays, num_examples)

            # 半同步路径：仅聚合 tx_time 不超过阈值的按时更新
            if self.current_mode in ("semi_sync", "bridge") and tx_time > threshold:
                continue
            valid_sync_updates.append((ndarrays, weight))

        # 同步聚合（FedAvg）
        if valid_sync_updates:
            sync_result = self._weighted_avg(valid_sync_updates, global_ndarrays)
        else:
            sync_result = global_ndarrays

        # 异步聚合（FedBuff）
        async_result = global_ndarrays
        if self.fedbuff.should_aggregate(server_round):
            async_result = self.fedbuff.aggregate(global_ndarrays, server_round)

        # 模式选择与桥接混合
        if self.current_mode == "async":
            merged = async_result
        elif self.current_mode == "semi_sync":
            merged = sync_result
        else:  # bridge
            w = float(np.clip(self.current_bridge_weight, 0.0, 1.0))
            merged = []
            for s, a in zip(sync_result, async_result):
                if np.issubdtype(s.dtype, np.floating):
                    merged.append(
                        (1.0 - w) * s.astype(np.float32, copy=False)
                        + w * a.astype(np.float32, copy=False)
                    )
                else:
                    merged.append(s)

        self.global_params_cache = ndarrays_to_parameters(merged)

        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None and scaffold_deltas:
            self.scaffold_state.update_global(scaffold_deltas, scaffold_weights)

        # 评估全局模型
        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))

        # 基于滑窗选择历史计算 Jain 公平指数
        counts = self._compute_selection_counts()
        jain = _jain_index(counts)
        self.avg_per_history.append(avg_per)
        self.jain_history.append(jain)

        # 将本轮指标喂给门控控制器（供下一轮决策）
        self.controller.register(avg_per=avg_per, jain=jain, total_energy=round_energy)

        # 写入 metrics.csv
        self._log_metrics(
            server_round=server_round,
            mode=self.current_mode,
            accuracy=float(acc),
            loss=float(loss),
            avg_per=avg_per,
            jain=jain,
            energy=float(round_energy),
            bw_factor=float(self.current_bw_factor),
            topk=len(scheduled_cids),
        )

        metrics: Dict[str, Scalar] = {
            "accuracy": float(acc),
            "loss": float(loss),
            "avg_per": float(avg_per),
            "jain": float(jain),
            "mode": self.current_mode,
            "energy": float(round_energy),
            "bw_factor": float(self.current_bw_factor),
            "topk": float(len(scheduled_cids)),
        }

        return self.global_params_cache, metrics

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):  # type: ignore[override]
        del server_round, parameters, client_manager
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):  # type: ignore[override]
        del server_round, results, failures
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):  # type: ignore[override]
        del server_round
        set_parameters(self.model_for_eval, parameters_to_ndarrays(parameters))
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        return float(loss), {"accuracy": float(acc)}


# -----------------------------
# 仿真入口：数据准备 + Flower simulation
# -----------------------------


def run_hybrid_flower_cifar(cfg: HybridCifarConfig) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_transform = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
    raw_trainset = CIFAR10(root=cfg.data_dir, train=True, download=False, transform=ToTensor())
    testset = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)

    labels = np.array(raw_trainset.targets)
    partitions = dirichlet_partition_indices(labels, cfg.num_clients, cfg.alpha, cfg.seed)
    partition_sizes = [len(idx) for idx in partitions]

    trainloaders = [
        DataLoader(
            Subset(trainset, indices),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        for indices in partitions
    ]
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_config.get("partition-id", context.node_id))
        return CifarClient(cid, trainloaders[cid], testloader, cfg.local_epochs, cfg.lr).to_client()

    strategy = HybridWirelessStrategy(cfg, partition_sizes, testloader)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
        ray_init_args={"include_dashboard": False},
    )

    final_acc = float(strategy.acc_history[-1]) if strategy.acc_history else 0.0
    avg_energy = float(np.mean(strategy.energy_history)) if strategy.energy_history else 0.0

    return {
        "final_accuracy": final_acc,
        "mode_history": strategy.mode_history,
        "avg_energy": avg_energy,
    }


# -----------------------------
# CLI 与 main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower hybrid FL demo with non-IID CIFAR partition and simplified wireless modeling",
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fraction-fit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=f"{project_root}/data")

    # 新增超参
    parser.add_argument("--selection-top-k", type=int, default=0,
                        help="Top-K 客户端数，<=0 时使用 num_clients*fraction_fit")
    parser.add_argument("--async-agg-interval", type=int, default=2,
                        help="FedBuff 异步聚合的最小轮间隔")
    parser.add_argument("--fair-window-size", type=int, default=4,
                        help="Jain 公平性滑动窗口大小")
    parser.add_argument("--fedprox-mu", type=float, default=0.0,
                        help="FedProx 近端正则系数，0 则关闭")
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "scaffold"],
                        help="本地训练算法")
    parser.add_argument("--semi-sync-wait-ratio", type=float, default=0.7,
                        help="半同步等待的发送时间分位数比例")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HybridCifarConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        fraction_fit=args.fraction_fit,
        seed=args.seed,
        data_dir=args.data_dir,
        selection_top_k=args.selection_top_k,
        async_agg_interval=args.async_agg_interval,
        fair_window_size=args.fair_window_size,
        fedprox_mu=args.fedprox_mu,
        algorithm=args.algorithm,
        semi_sync_wait_ratio=args.semi_sync_wait_ratio,
    )

    result = run_hybrid_flower_cifar(cfg)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Average energy: {result['avg_energy']:.4f}")
    print("Mode history:", ", ".join(result["mode_history"]))


if __name__ == "__main__":
    main()
