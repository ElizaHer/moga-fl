from __future__ import annotations

import flwr as fl
from flwr.common import Context
from flwr.server.client_manager import ClientManager

from src.configs.hybrid_cifar import HybridCifarConfig
from src.data.dataset_loader import cifar_loader
from src.training.client import CifarClient
from src.training.strategy.hybrid_wireless import HybridWirelessStrategy
from src.utils.train import *


def run_hybrid_flower_cifar(cfg: HybridCifarConfig) -> Dict[str, object]:
    (train_loaders, test_loader, partition_sizes) = cifar_loader(cfg)

    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_config.get("partition-id", context.node_id))
        return CifarClient(cid, train_loaders[cid], test_loader, cfg.local_epochs, cfg.lr).to_client()

    strategy = HybridWirelessStrategy(cfg, partition_sizes, test_loader)

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
    parser.add_argument("--data-dir", type=str, default=f"./data")

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
    parser.add_argument("--initial-client-energy", type=float, default=100.0,
                        help="每个客户端初始能量（默认所有客户端相同）")
    parser.add_argument(
        "--client-initial-energies",
        type=str,
        default="",
        help="按客户端顺序指定初始能量，逗号分隔，例如: 120,100,80,60",
    )

    return parser.parse_args()


def _setup_logging() -> None:
    import logging

    flwr_logger = logging.getLogger("flwr")
    flwr_logger.propagate = False  # 防止再冒泡到 root 导致重复打印


def main() -> None:
    _setup_logging()
    args = parse_args()
    custom_energies = None
    if args.client_initial_energies.strip():
        custom_energies = [
            float(x.strip()) for x in args.client_initial_energies.split(",") if x.strip()
        ]
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
        initial_client_energy=args.initial_client_energy,
        client_initial_energies=custom_energies,
    )

    result = run_hybrid_flower_cifar(cfg)
    print(f"Final accuracy: {result['final_accuracy']:.4f}")
    print(f"Average energy: {result['avg_energy']:.4f}")
    print("Mode history:", ", ".join(result["mode_history"]))


if __name__ == "__main__":
    main()
