import flwr as fl
from flwr.common import Context

from src.configs.hybrid_cifar import HybridCifarConfig
from src.data.dataset_loader import cifar_loader
from src.training.client import CifarClient
from src.training.strategy.fedavg_only import FedAvgOnlyHybridStrategy
from src.utils.train import *


def run_fedavg_only_flower_cifar(cfg: HybridCifarConfig) -> Dict[str, object]:
    (train_loaders, test_loader, partition_sizes) = cifar_loader(cfg)

    def client_fn(context: Context) -> fl.client.Client:
        cid = int(context.node_config.get("partition-id", context.node_id))
        return CifarClient(cid, train_loaders[cid], test_loader, cfg.local_epochs, cfg.lr).to_client()

    strategy = FedAvgOnlyHybridStrategy(cfg, partition_sizes, test_loader)
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
