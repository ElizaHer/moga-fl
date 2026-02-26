from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Context
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from src.flower.hybrid_opt_demo import (
    CifarClient,
    HybridCifarConfig,
    HybridWirelessStrategy,
    _jain_index,
    _unpack_tensor_dict,
    dirichlet_partition_indices,
    evaluate_model,
    set_parameters,
)


class FedAvgOnlyHybridStrategy(HybridWirelessStrategy):
    """Keep scheduling/wireless/mode control unchanged, but always aggregate with FedAvg."""

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[override]
        del failures
        if self.global_params_cache is None:
            return None, {}

        global_ndarrays = parameters_to_ndarrays(self.global_params_cache)
        scheduled_cids = list(self.last_selected_cids)
        if not scheduled_cids:
            scheduled_cids = [int(res.metrics.get("cid", -1)) for _, res in results]
            scheduled_cids = [cid for cid in scheduled_cids if 0 <= cid < self.cfg.num_clients]

        wireless_stats = self.current_wireless_stats or self.channel.sample_round()
        per_values = [
            float((wireless_stats.get(cid) or {}).get("per", 0.0))
            for cid in scheduled_cids
        ]
        avg_per = float(np.mean(per_values)) if per_values else 0.0

        self._update_bandwidth_budget()
        bw_map = self.bw.allocate_uniform(scheduled_cids)
        tx_times: Dict[int, float] = {
            cid: self.bw.estimate_tx_time(payload_mb=1.0, allocated_mb=float(bw_map.get(cid, 0.0)))
            for cid in scheduled_cids
        }
        threshold = float("inf")
        if scheduled_cids and self.current_mode in ("semi_sync", "bridge"):
            threshold = float(
                np.quantile(
                    np.asarray([tx_times[cid] for cid in scheduled_cids], dtype=np.float64),
                    self.cfg.semi_sync_wait_ratio,
                )
            )

        valid_updates: List[Tuple[List[np.ndarray], float]] = []
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
            per = float((wireless_stats.get(cid, {})).get("per", 0.0))
            round_energy += self.energy.comm_energy(tx_time) + self.energy.compute_energy(num_examples)

            if np.random.rand() < per:
                continue
            if self.current_mode in ("semi_sync", "bridge") and tx_time > threshold:
                continue

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weight = float(num_examples)
            valid_updates.append((ndarrays, weight))

            if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None:
                payload = fit_res.metrics.get("delta_ci", b"")
                if isinstance(payload, (bytes, bytearray)) and len(payload) > 0:
                    delta_ci = _unpack_tensor_dict(bytes(payload), torch.device("cpu"))
                    if delta_ci:
                        scaffold_deltas[cid] = delta_ci
                        scaffold_weights[cid] = weight

        merged = self._weighted_avg(valid_updates, global_ndarrays) if valid_updates else global_ndarrays
        self.global_params_cache = ndarrays_to_parameters(merged)

        if self.cfg.algorithm.lower() == "scaffold" and self.scaffold_state is not None and scaffold_deltas:
            self.scaffold_state.update_global(scaffold_deltas, scaffold_weights)

        set_parameters(self.model_for_eval, merged)
        loss, acc = evaluate_model(self.model_for_eval, self.testloader, self.device)
        self.acc_history.append(float(acc))
        self.energy_history.append(float(round_energy))

        counts = self._compute_selection_counts()
        jain = _jain_index(counts)
        self.avg_per_history.append(avg_per)
        self.jain_history.append(jain)
        self.controller.register(avg_per=avg_per, jain=jain, total_energy=round_energy)

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
        return self.global_params_cache, {
            "accuracy": float(acc),
            "loss": float(loss),
            "avg_per": float(avg_per),
            "jain": float(jain),
            "mode": self.current_mode,
            "energy": float(round_energy),
            "bw_factor": float(self.current_bw_factor),
            "topk": float(len(scheduled_cids)),
        }


def run_fedavg_only_flower_cifar(cfg: HybridCifarConfig) -> Dict[str, object]:
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

    strategy = FedAvgOnlyHybridStrategy(cfg, partition_sizes, testloader)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
        ray_init_args={"include_dashboard": False},
    )
    return {
        "final_accuracy": float(strategy.acc_history[-1]) if strategy.acc_history else 0.0,
        "mode_history": strategy.mode_history,
        "avg_energy": float(np.mean(strategy.energy_history)) if strategy.energy_history else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower non-IID CIFAR demo with wireless modeling and FedAvg-only aggregation"
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
    parser.add_argument("--selection-top-k", type=int, default=0)
    parser.add_argument("--fair-window-size", type=int, default=4)
    parser.add_argument("--fedprox-mu", type=float, default=0.0)
    parser.add_argument("--semi-sync-wait-ratio", type=float, default=0.7)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "scaffold"],
    )
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
        fair_window_size=args.fair_window_size,
        fedprox_mu=args.fedprox_mu,
        semi_sync_wait_ratio=args.semi_sync_wait_ratio,
        algorithm=args.algorithm,
    )
    result = run_fedavg_only_flower_cifar(cfg)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Average energy: {result['avg_energy']:.4f}")
    print("Mode history:", ", ".join(result["mode_history"]))


if __name__ == "__main__":
    main()
