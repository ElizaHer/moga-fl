from __future__ import annotations

import argparse
import logging
from typing import Dict

import flwr as fl
import numpy as np
from flwr.common import Context

from src.configs.strategy_runtime import apply_cli_overrides, default_strategy_yaml, load_strategy_yaml
from src.data.dataset_loader import build_wsn_wireless_sampler, cifar_loader
from src.training.client import CifarClient
from src.training.strategy.factory import build_strategy


def run_hybrid_flower_cifar(cfg: Dict, strategy_name: str) -> Dict[str, object]:
    train_loaders, test_loader, partition_sizes = cifar_loader(cfg)
    wireless_model = str(cfg.get("wireless", {}).get("wireless_model", "simulated")).lower()
    wsn_sampler = build_wsn_wireless_sampler(cfg) if wireless_model == "wsn" else None

    local_epochs = int(cfg["fl"]["local_epochs"])
    lr = float(cfg["fl"]["lr"])
    num_clients = int(cfg["fl"]["num_clients"])
    num_rounds = int(cfg["fl"]["num_rounds"])
    resources = cfg["fl"].get("client_resources", {"num_cpus": 2, "num_gpus": 0.25})

    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_config.get("partition-id", context.node_id))
        return CifarClient(cid, train_loaders[cid], test_loader, local_epochs, lr).to_client()

    strategy = build_strategy(strategy_name, cfg, partition_sizes, test_loader, wsn_wireless_sampler=wsn_sampler)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": float(resources.get("num_cpus", 2)), "num_gpus": float(resources.get("num_gpus", 0.25))},
        ray_init_args={"include_dashboard": False},
    )

    final_acc = float(strategy.acc_history[-1]) if strategy.acc_history else 0.0
    avg_energy = float(np.mean(strategy.energy_history)) if strategy.energy_history else 0.0
    return {
        "final_accuracy": final_acc,
        "avg_energy": avg_energy,
        "mode_history": strategy.mode_history,
        "metrics_csv": strategy.metrics_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower hybrid FL demo with strategy YAML + CLI overrides")
    parser.add_argument("--strategy", type=str, default="hybrid_opt", choices=["hybrid_opt", "sync", "async", "bridge_free", "bandwidth_first", "energy_first"])
    parser.add_argument("--config", type=str, default="", help="Optional explicit YAML path, default is src/configs/strategies/<strategy>.yaml")

    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--fraction-fit", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--algorithm", type=str, default="", choices=["", "fedavg", "fedprox", "scaffold"])
    parser.add_argument("--fedprox-mu", type=float, default=None)

    parser.add_argument("--wireless-model", type=str, default="", choices=["", "simulated", "wsn"])
    parser.add_argument("--simulated-mode", type=str, default="", choices=["", "good", "bad", "jitter"])
    parser.add_argument("--jitter-period-rounds", type=int, default=None)
    parser.add_argument("--jitter-start-state", type=str, default="", choices=["", "good", "bad"])
    parser.add_argument("--wsn-csv-path", type=str, default="")
    parser.add_argument("--selection-top-k", type=int, default=None)
    parser.add_argument("--async-agg-interval", type=int, default=None)
    parser.add_argument("--fair-window-size", type=int, default=None)
    parser.add_argument("--semi-sync-wait-ratio", type=float, default=None)
    parser.add_argument("--initial-client-energy", type=float, default=None)
    parser.add_argument("--client-initial-energies", type=str, default="")
    parser.add_argument("--client-num-cpus", type=float, default=None)
    parser.add_argument("--client-num-gpus", type=float, default=None)
    return parser.parse_args()


def _setup_logging() -> None:
    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False


def main() -> None:
    _setup_logging()
    args = parse_args()
    config_path = args.config.strip() or default_strategy_yaml(args.strategy)
    cfg = load_strategy_yaml(config_path)
    cfg = apply_cli_overrides(cfg, args)

    result = run_hybrid_flower_cifar(cfg, strategy_name=args.strategy)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Average energy: {result['avg_energy']:.4f}")
    print("Mode history:", ", ".join(result["mode_history"]))
    print(f"Metrics CSV: {result['metrics_csv']}")


if __name__ == "__main__":
    main()
