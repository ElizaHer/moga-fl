import argparse
import os
import sys

import yaml
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl_core.data import SyntheticConfig, make_synthetic_dataset, dirichlet_partition, build_federated_dataloaders
from fl_core.trainer import FLConfig, FLSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description="最小联邦学习示例（基于 FedLab 1.3.0 API 封装）")
    parser.add_argument("--config", type=str, default="configs/demo_fl.yaml", help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    fl_cfg_raw = cfg.get("fl", {})

    syn_cfg = SyntheticConfig(
        num_samples=int(data_cfg.get("num_samples", 2000)),
        num_clients=int(fl_cfg_raw.get("num_clients", 10)),
        num_classes=int(data_cfg.get("num_classes", 10)),
        image_size=tuple(data_cfg.get("image_size", [28, 28])),
    )

    train_ds, test_ds, num_classes = make_synthetic_dataset(syn_cfg)
    labels = np.array([y for _, y in train_ds])
    parts = dirichlet_partition(labels, num_clients=syn_cfg.num_clients, alpha=0.5)
    train_loaders = build_federated_dataloaders(train_ds, parts, batch_size=int(fl_cfg_raw.get("batch_size", 32)))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(fl_cfg_raw.get("test_batch_size", 128)), shuffle=False)

    fl_cfg = FLConfig(
        num_clients=syn_cfg.num_clients,
        sample_ratio=float(fl_cfg_raw.get("sample_ratio", 0.5)),
        local_epochs=int(fl_cfg_raw.get("local_epochs", 1)),
        batch_size=int(fl_cfg_raw.get("batch_size", 32)),
        lr=float(fl_cfg_raw.get("lr", 0.1)),
        rounds=int(fl_cfg_raw.get("rounds", 5)),
        test_batch_size=int(fl_cfg_raw.get("test_batch_size", 128)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = FLSimulation(train_loaders, test_loader, fl_cfg, device=device, num_classes=num_classes)
    res = sim.run()

    print("=== 联邦学习最小示例结果 ===")
    print(f"平均准确率: {res['accuracy_mean']:.4f}")
    print(f"平均时间 proxy: {res['time_mean']:.4f}")
    print(f"平均 Jain 公平指数: {res['fairness_mean']:.4f}")


if __name__ == "__main__":
    main()
