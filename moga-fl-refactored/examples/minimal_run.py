"""最小可运行示例：直接在 Python 中调用 FLSimulation。

运行方式：

    python examples/minimal_run.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fl_core.data import SyntheticConfig, make_synthetic_dataset, dirichlet_partition, build_federated_dataloaders
from fl_core.trainer import FLConfig, FLSimulation


def main() -> None:
    syn_cfg = SyntheticConfig(num_samples=1000, num_clients=5, num_classes=5)
    train_ds, test_ds, _ = make_synthetic_dataset(syn_cfg)
    labels = np.array([y for _, y in train_ds])
    parts = dirichlet_partition(labels, num_clients=syn_cfg.num_clients, alpha=0.5)
    train_loaders = build_federated_dataloaders(train_ds, parts, batch_size=16)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    fl_cfg = FLConfig(num_clients=syn_cfg.num_clients, sample_ratio=0.6, local_epochs=1, batch_size=16, lr=0.1, rounds=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim = FLSimulation(train_loaders, test_loader, fl_cfg, device=device, num_classes=syn_cfg.num_classes)
    res = sim.run()
    print("Minimal example results:")
    print(res)


if __name__ == "__main__":
    main()
