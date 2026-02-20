import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import torch

from fl_core import load_config, merge_defaults
from fl_core.data.dataset_loader import DatasetManager
from fl_core.data.partition import (
    apply_quick_limit,
    dirichlet_partition,
    label_bias_partition,
    quantity_bias_partition,
)
from fl_core.training.fedlab_engine import MogaFLServer
from fl_core.training.models import (
    SimpleCIFARCNN,
    SimpleEMNISTCNN,
    build_resnet18_cifar,
    build_resnet18_emnist,
)
from fl_core.utils.metrics import MetricsRecorder
from fl_core.utils.plotting import plot_curves
from fl_core.utils.seed import set_seed
from ga_opt import MogaFLOptimizer


def build_model_fn(cfg: Dict[str, Any], num_classes: int):
    """根据配置选择模型结构。"""
    model_cfg = cfg.get("model", {})
    mtype = model_cfg.get("type", "small_cnn")
    width_factor = float(model_cfg.get("width_factor", 1.0))
    dataset_name = cfg["dataset"]["name"]

    if mtype == "resnet18_cifar" and dataset_name == "cifar10":
        return lambda: build_resnet18_cifar(num_classes=num_classes, width_factor=width_factor)
    if mtype == "resnet18_emnist" and dataset_name == "emnist":
        return lambda: build_resnet18_emnist(num_classes=num_classes, width_factor=width_factor)

    # 回退到轻量 CNN
    if dataset_name == "cifar10":
        return SimpleCIFARCNN
    else:
        return lambda: SimpleEMNISTCNN(num_classes)


def build_partitions(train_dataset, cfg: Dict[str, Any], num_classes: int):
    labels = np.array([y for _, y in train_dataset])
    num_clients = cfg["clients"]["num_clients"]
    niid = cfg["dataset"]["noniid"]
    if niid["type"] == "dirichlet":
        client_indices = dirichlet_partition(labels, num_clients, num_classes, niid.get("alpha", 0.5))
    elif niid["type"] == "label_bias":
        cpc = niid.get("classes_per_client", niid.get("label_bias_classes_per_client", 2))
        client_indices = label_bias_partition(labels, num_clients, num_classes, cpc)
    else:
        client_indices = quantity_bias_partition(labels, num_clients, niid.get("quantity_bias_sigma", 0.6))
    spc = cfg["clients"].get("samples_per_client")
    if spc:
        client_indices = apply_quick_limit(client_indices, spc)
    return client_indices


def make_sim_runner(
    cfg: Dict[str, Any],
    train,
    test,
    num_classes: int,
    partitions,
    device: torch.device,
    round_scale: float = 0.6,
):
    """构造用于 GA 评估的联邦训练短跑器。

    round_scale 控制评估轮数比例：
    - 低保真：<1.0，只跑少量轮数；
    - 高保真：1.0，使用完整轮数。
    """

    def runner(params: Dict[str, Any]) -> Dict[str, float]:
        cfg2 = deepcopy(cfg)
        # scheduling 权重
        cfg2.setdefault("scheduling", cfg.get("scheduling", {})).setdefault("weights", {})
        cfg2["scheduling"]["weights"] = {
            "energy": params["energy_w"],
            "channel": params["channel_w"],
            "data_value": params["data_w"],
            "fairness_debt": params["fair_w"],
            "bandwidth_cost": params["bwcost_w"],
        }
        # 客户端选择参数
        cfg2.setdefault("clients", cfg.get("clients", {}))
        cfg2["clients"]["selection_top_k"] = int(params["selection_top_k"])
        cfg2["clients"]["hysteresis"] = float(params["hysteresis"])
        # FedBuff 陈旧度加权
        cfg2.setdefault("training", cfg.get("training", {}))
        fb_cfg = cfg2["training"].get("fedbuff", cfg["training"].get("fedbuff", {}).copy())
        fb_cfg["staleness_alpha"] = float(params["staleness_alpha"])
        cfg2["training"]["fedbuff"] = fb_cfg
        # 评估轮数
        cfg2.setdefault("eval", cfg.get("eval", {}))
        cfg2["eval"]["rounds"] = max(3, int(cfg["eval"]["rounds"] * round_scale))

        model_fn = build_model_fn(cfg2, num_classes)
        server = MogaFLServer(model_fn, train, test, num_classes, cfg2, device)

        accs: list[float] = []
        times: list[float] = []
        fairs: list[float] = []
        energies: list[float] = []
        for r in range(cfg2["eval"]["rounds"]):
            row = server.round(r, partitions)
            accs.append(row["accuracy"])
            times.append(row["comm_time"])
            fairs.append(row["jain_index"])
            energies.append(row["comm_energy"] + row["comp_energy"])

        return {
            "acc": float(np.mean(accs)),
            "time": float(np.mean(times)),
            "fairness": float(np.mean(fairs)),
            "energy": float(np.mean(energies)),
        }

    return runner


def run_baseline(cfg_path: str, run_ga: bool = False, ga_generations: int = 4, ga_pop: int = 12) -> None:
    cfg = merge_defaults(load_config(cfg_path))
    set_seed(cfg["eval"].get("seed"))

    dm = DatasetManager(cfg)
    train, test, num_classes = dm.load()
    partitions = build_partitions(train, cfg, num_classes)

    device = torch.device("cpu")  # 默认使用 CPU，便于在无 GPU 环境运行
    model_fn = build_model_fn(cfg, num_classes)
    server = MogaFLServer(model_fn, train, test, num_classes, cfg, device)

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    rec = MetricsRecorder()

    rounds = cfg["eval"]["rounds"]
    for r in range(rounds):
        row = server.round(r, partitions)
        rec.add(row)
        print(
            f"Round {r}: acc={row['accuracy']:.4f}, fairness={row['jain_index']:.3f}, "
            f"comm_energy={row['comm_energy']:.3f}"
        )

    df = rec.to_csv("outputs/results/metrics.csv")
    if cfg["eval"].get("save_plots", True):
        plot_curves(df, "outputs/plots/run")

    if not run_ga:
        return

    # GA 优化：构造低/高保真评估器并运行 pymoo-NSGA2
    low_sim = make_sim_runner(cfg, train, test, num_classes, partitions, device, round_scale=0.5)
    high_sim = make_sim_runner(cfg, train, test, num_classes, partitions, device, round_scale=1.0)

    opt = MogaFLOptimizer(cfg, low_eval=low_sim, high_eval=high_sim, pop_size=ga_pop)
    pop, metrics = opt.run(generations=ga_generations)

    # 保存 Pareto 候选
    import pandas as pd

    rows = []
    for p, m in zip(pop, metrics):
        row = {**p, **m}
        rows.append(row)
    df_pareto = pd.DataFrame(rows)
    df_pareto.to_csv("outputs/results/pareto_candidates.csv", index=False)
    print("Saved Pareto candidates to outputs/results/pareto_candidates.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/quick_cifar10.yaml")
    parser.add_argument("--run_ga", action="store_true", help="同时运行基于 pymoo 的 GA 优化")
    parser.add_argument("--ga_generations", type=int, default=4)
    parser.add_argument("--ga_pop", type=int, default=12)
    args = parser.parse_args()

    run_baseline(args.config, run_ga=args.run_ga, ga_generations=args.ga_generations, ga_pop=args.ga_pop)


if __name__ == "__main__":
    main()
