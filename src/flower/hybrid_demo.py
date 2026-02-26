from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Context, NDArrays, Parameters, Scalar
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from src.training.models.resnet_cifar import build_resnet18_cifar
from src.wireless.bandwidth import BandwidthAllocator
from src.wireless.channel import ChannelSimulator
from src.wireless.energy import EnergyEstimator


@dataclass
class HybridCifarConfig:
    num_clients: int = 10
    num_rounds: int = 8
    local_epochs: int = 1
    batch_size: int = 128
    lr: float = 0.001
    alpha: float = 0.5
    fraction_fit: float = 0.5
    seed: int = 42
    data_dir: str = "./data"
    semi_sync_wait_ratio: float = 0.7
    fedbuff_buffer_size: int = 16
    fedbuff_min_updates_to_aggregate: int = 8
    staleness_alpha: float = 1.0
    max_staleness: int = 8
    # 兼容旧平铺字段（建议改用 strategy 嵌套结构）
    gate_to_async: float = 0.58
    gate_to_semi_sync: float = 0.42
    hysteresis_margin: float = 0.03
    bridge_rounds: int = 2
    min_rounds_between_switch: int = 2
    w_per: float = 0.5
    w_fair: float = 0.3
    w_energy: float = 0.2
    window_size: int = 4
    strategy: Dict[str, object] = field(
        default_factory=lambda: {
            "window_size": 4,
            "gate_thresholds": {"to_async": 0.58, "to_semi_sync": 0.42},
            "hysteresis_margin": 0.03,
            "bridge_rounds": 2,
            "min_rounds_between_switch": 2,
            "weights": {"per": 0.5, "fairness": 0.3, "energy": 0.2},
            "bandwidth_rebalance": {"low_energy_factor": 0.8, "high_energy_factor": 1.0},
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
    return total_loss / max(1, len(dataloader)), correct / max(1, total)


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
        split = np.split(class_ids, split_points)
        for client_id, idx in enumerate(split):
            client_indices[client_id].extend(idx.tolist())
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def _jain_index(values: List[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    return float((x.sum() ** 2) / (len(x) * np.sum(x * x) + 1e-12))


class FedBuffState:
    def __init__(self, alpha: float, max_staleness: int, buffer_size: int, min_updates: int) -> None:
        self.alpha = alpha
        self.max_staleness = max_staleness
        self.buffer_size = buffer_size
        self.min_updates = min_updates
        self.entries: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        self.updates_since_last_agg = 0
        self.last_agg_round = 0

    def age(self, delta: int) -> None:
        if delta <= 0:
            return
        aged: Deque[Tuple[List[np.ndarray], int, int]] = deque()
        for params, staleness, num_examples in self.entries:
            s = staleness + delta
            if s <= self.max_staleness:
                aged.append((params, s, num_examples))
        self.entries = aged

    def push(self, params: List[np.ndarray], num_examples: int) -> None:
        self.entries.append((params, 0, num_examples))
        self.updates_since_last_agg += 1

    def should_aggregate(self, server_round: int) -> bool:
        if len(self.entries) >= self.buffer_size:
            return True
        if self.updates_since_last_agg >= self.min_updates:
            return True
        if server_round - self.last_agg_round >= 2 and len(self.entries) > 0:
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
            staleness_weight = 1.0 / ((1.0 + staleness) ** self.alpha)
            w = staleness_weight * float(num_examples)
            sum_weights += w
            for idx, arr in enumerate(params):
                if np.issubdtype(agg[idx].dtype, np.floating):
                    agg[idx] += w * arr.astype(np.float32, copy=False)
        out = []
        for arr in agg:
            if np.issubdtype(arr.dtype, np.floating):
                out.append(arr / max(1e-12, sum_weights))
            else:
                out.append(arr)
        self.entries.clear()
        self.updates_since_last_agg = 0
        self.last_agg_round = server_round
        return out


class ModeController:
    def __init__(self, cfg: HybridCifarConfig, num_clients: int) -> None:
        self.cfg = cfg
        self.num_clients = num_clients
        strat = cfg.strategy if isinstance(cfg.strategy, dict) else {}
        gates = strat.get("gate_thresholds", {}) if isinstance(strat.get("gate_thresholds", {}), dict) else {}
        weights = strat.get("weights", {}) if isinstance(strat.get("weights", {}), dict) else {}
        self.window_size = int(strat.get("window_size", cfg.window_size))
        self.to_async = float(gates.get("to_async", cfg.gate_to_async))
        self.to_semi = float(gates.get("to_semi_sync", cfg.gate_to_semi_sync))
        self.hysteresis_margin = float(strat.get("hysteresis_margin", cfg.hysteresis_margin))
        self.bridge_rounds = int(strat.get("bridge_rounds", cfg.bridge_rounds))
        self.min_rounds_between_switch = int(
            strat.get("min_rounds_between_switch", cfg.min_rounds_between_switch)
        )
        self.w_per = float(weights.get("per", cfg.w_per))
        self.w_fair = float(weights.get("fairness", cfg.w_fair))
        self.w_energy = float(weights.get("energy", cfg.w_energy))
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.current_mode = "semi_sync"
        self.bridge_target: str | None = None
        self.bridge_start = -1
        self.last_switch_round = -10**9

    def register(self, avg_per: float, jain: float, total_energy: float) -> None:
        self.history.append(
            {"avg_per": float(avg_per), "jain": float(jain), "energy": float(total_energy)}
        )

    def _gate_score(self) -> float:
        if not self.history:
            return 0.0
        avg_per = float(np.mean([row["avg_per"] for row in self.history]))
        jain = float(np.mean([row["jain"] for row in self.history]))
        energy_norm = float(np.clip(np.mean([row["energy"] for row in self.history]) / 10.0, 0.0, 1.0))
        fairness_deficit = 1.0 - jain
        return float(
            self.w_per * avg_per
            + self.w_fair * fairness_deficit
            + self.w_energy * energy_norm
        )

    def decide(self, server_round: int) -> Tuple[str, float]:
        if self.current_mode == "bridge":
            t = max(0, server_round - self.bridge_start)
            w = float(np.clip(t / max(1, self.bridge_rounds), 0.0, 1.0))
            if t >= self.bridge_rounds and self.bridge_target is not None:
                self.current_mode = self.bridge_target
                self.bridge_target = None
                self.bridge_start = -1
            return "bridge", w
        if server_round - self.last_switch_round < self.min_rounds_between_switch:
            return self.current_mode, 0.0
        gate = self._gate_score()
        target = self.current_mode
        if gate >= self.to_async + self.hysteresis_margin:
            target = "async"
        elif gate <= self.to_semi - self.hysteresis_margin:
            target = "semi_sync"
        if target != self.current_mode:
            self.current_mode = "bridge"
            self.bridge_target = target
            self.bridge_start = server_round
            self.last_switch_round = server_round
            return "bridge", 0.0
        return self.current_mode, 0.0


class CifarClient(fl.client.NumPyClient):
    def __init__(self, cid: int, trainloader: DataLoader, testloader: DataLoader, local_epochs: int, lr: float) -> None:
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_cifar().to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.model.train()
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        self.scheduler.step()
        return get_parameters(self.model), len(self.trainloader.dataset), {"cid": float(self.cid)}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


class HybridWirelessStrategy(Strategy):
    def __init__(self, cfg: HybridCifarConfig, testloader: DataLoader):
        self.cfg = cfg
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_for_eval = build_resnet18_cifar().to(self.device)
        self.controller = ModeController(cfg, cfg.num_clients)
        self.fedbuff = FedBuffState(
            alpha=cfg.staleness_alpha,
            max_staleness=cfg.max_staleness,
            buffer_size=cfg.fedbuff_buffer_size,
            min_updates=cfg.fedbuff_min_updates_to_aggregate,
        )

        wireless_cfg = _base_wireless_cfg()
        self.channel = ChannelSimulator(wireless_cfg, cfg.num_clients)
        self.bw = BandwidthAllocator(wireless_cfg)
        self.energy = EnergyEstimator(wireless_cfg)

        self.global_params_cache: Parameters | None = None
        self.selection_count = [0.0 for _ in range(cfg.num_clients)]
        self.mode_history: List[str] = []
        self.acc_history: List[float] = []
        self.energy_history: List[float] = []

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        del client_manager
        ndarrays = get_parameters(build_resnet18_cifar())
        self.global_params_cache = ndarrays_to_parameters(ndarrays)
        return self.global_params_cache

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        num_fit = max(1, int(self.cfg.num_clients * self.cfg.fraction_fit))
        clients = client_manager.sample(num_clients=num_fit, min_num_clients=num_fit)
        fit_ins = fl.common.FitIns(parameters, {"server_round": float(server_round)})
        return [(client, fit_ins) for client in clients]

    def _weighted_avg(self, pairs: List[Tuple[List[np.ndarray], float]]) -> List[np.ndarray]:
        total = float(sum(w for _, w in pairs))
        base = parameters_to_ndarrays(self.global_params_cache) if self.global_params_cache is not None else pairs[0][0]
        out = [arr.copy() for arr in base]
        for i in range(len(out)):
            if np.issubdtype(out[i].dtype, np.floating):
                out[i] = np.zeros_like(out[i], dtype=np.float32)
        for params, w in pairs:
            scale = w / max(1e-12, total)
            for i, arr in enumerate(params):
                if np.issubdtype(out[i].dtype, np.floating):
                    out[i] += scale * arr.astype(np.float32, copy=False)
        return out

    def aggregate_fit(self, server_round: int, results, failures):
        del failures
        if self.global_params_cache is None:
            return None, {}
        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)
        mode, bridge_w = self.controller.decide(server_round)
        self.mode_history.append(mode)

        wireless_stats = self.channel.sample_round()
        selected_cids = [int(fit_res.metrics.get("cid", -1)) for _, fit_res in results]
        selected_cids = [cid for cid in selected_cids if 0 <= cid < self.cfg.num_clients]
        bw_map = self.bw.allocate_uniform(selected_cids)
        tx_times = {
            cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=bw_map.get(cid, 0.0))
            for cid in selected_cids
        }

        threshold = float("inf")
        if selected_cids and mode in ("semi_sync", "bridge"):
            threshold = float(
                np.quantile(np.asarray([tx_times[cid] for cid in selected_cids]), self.cfg.semi_sync_wait_ratio)
            )

        valid_sync_updates: List[Tuple[List[np.ndarray], float]] = []
        per_values: List[float] = []
        round_energy = 0.0
        for _, fit_res in results:
            cid = int(fit_res.metrics.get("cid", -1))
            if cid < 0 or cid >= self.cfg.num_clients:
                continue
            per = float(wireless_stats[cid]["per"])
            per_values.append(per)
            self.selection_count[cid] += 1.0
            tx_time = tx_times.get(cid, 1e9)
            if np.random.rand() < per:
                round_energy += self.energy.comm_energy(tx_time)
                continue
            if mode in ("semi_sync", "bridge") and tx_time > threshold:
                continue
            nd = parameters_to_ndarrays(fit_res.parameters)
            num_examples = int(fit_res.num_examples)
            if num_examples <= 0:
                continue
            valid_sync_updates.append((nd, float(num_examples)))
            self.fedbuff.push(nd, num_examples)
            round_energy += self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)

        self.fedbuff.age(max(0, server_round - self.fedbuff.last_agg_round))
        sync_result = global_ndarrays if not valid_sync_updates else self._weighted_avg(valid_sync_updates)
        async_result = global_ndarrays
        if self.fedbuff.should_aggregate(server_round):
            async_result = self.fedbuff.aggregate(global_ndarrays, server_round)

        if mode == "async":
            merged = async_result
        elif mode == "semi_sync":
            merged = sync_result
        else:
            w = float(np.clip(bridge_w, 0.0, 1.0))
            merged = []
            for s, a in zip(sync_result, async_result):
                if np.issubdtype(s.dtype, np.floating):
                    merged.append((1.0 - w) * s.astype(np.float32, copy=False) + w * a.astype(np.float32, copy=False))
                else:
                    merged.append(s)

        self.global_params_cache = ndarrays_to_parameters(merged)
        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)

        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))
        avg_per = float(np.mean(per_values)) if per_values else 0.0
        jain = _jain_index(self.selection_count)
        self.controller.register(avg_per=avg_per, jain=jain, total_energy=round_energy)
        return self.global_params_cache, {
            "accuracy": float(acc),
            "loss": float(loss),
            "avg_per": avg_per,
            "jain": jain,
            "mode": mode,
            "energy": float(round_energy),
        }

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        del server_round, parameters, client_manager
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        del server_round, results, failures
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        del server_round
        set_parameters(self.model_for_eval, parameters_to_ndarrays(parameters))
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        return float(loss), {"accuracy": float(acc)}


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
    test_transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
    raw_trainset = CIFAR10(root=cfg.data_dir, train=True, download=False, transform=ToTensor())
    testset = CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)
    labels = np.array(raw_trainset.targets)
    partitions = dirichlet_partition_indices(labels, cfg.num_clients, cfg.alpha, cfg.seed)
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

    strategy = HybridWirelessStrategy(cfg, testloader)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
        ray_init_args={"include_dashboard": False},
    )
    return {
        "final_accuracy": float(strategy.acc_history[-1]) if strategy.acc_history else 0.0,
        "mode_history": strategy.mode_history,
        "avg_energy": float(np.mean(strategy.energy_history)) if strategy.energy_history else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower hybrid FL demo with non-IID CIFAR partition and wireless modeling"
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fraction-fit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
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
    )
    result = run_hybrid_flower_cifar(cfg)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Average energy: {result['avg_energy']:.4f}")
    print("Mode history:", ", ".join(result["mode_history"]))


if __name__ == "__main__":
    main()
