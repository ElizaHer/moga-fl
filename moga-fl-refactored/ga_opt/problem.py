from __future__ import annotations

from typing import Dict, Any

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from fl_core.trainer import FLSimulation, FLConfig
from fl_core.data import SyntheticConfig, make_synthetic_dataset, dirichlet_partition, build_federated_dataloaders
from fl_core.models import build_model
import torch


class FLHyperparamProblem(ElementwiseProblem):
    """基于 pymoo 的联邦学习超参数多目标优化问题。

    决策变量 x = [w_energy, w_channel, w_data, w_fair, w_bwcost, top_k_ratio, local_epochs].
    为简单起见：
    - 目标 1: 最小化 -accuracy（即最大化准确率）；
    - 目标 2: 最小化 时间 proxy（客户端数×本地 epoch）；
    - 目标 3: 最小化 -公平性（即最大化 Jain 指数）。
    """

    def __init__(self, num_clients: int = 10, num_classes: int = 10):
        self.num_clients = num_clients
        self.num_classes = num_classes

        # 在问题内构建一次数据与仿真环境，后续评估复用
        syn_cfg = SyntheticConfig(num_samples=2000, num_clients=num_clients, num_classes=num_classes)
        train_ds, test_ds, _ = make_synthetic_dataset(syn_cfg)
        labels = np.array([y for _, y in train_ds])
        parts = dirichlet_partition(labels, num_clients=num_clients, alpha=0.5)
        train_loaders = build_federated_dataloaders(train_ds, parts, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fl_cfg = FLConfig(num_clients=num_clients, sample_ratio=0.5, local_epochs=1, batch_size=32, lr=0.1, rounds=5)
        self.fl_sim = FLSimulation(train_loaders, test_loader, fl_cfg, device=device, num_classes=num_classes)

        super().__init__(
            n_var=7,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.1, 1]),
            xu=np.array([1.0, 1.0, 1.0, 1.0, -0.01, 1.0, 5]),
        )

    def _evaluate(self, x: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:  # type: ignore[override]
        # 解析超参数（目前主要影响 sample_ratio 与 local_epochs）
        top_k_ratio = float(x[5])
        local_epochs = int(round(float(x[6])))
        local_epochs = max(1, min(5, local_epochs))

        # 更新仿真配置
        self.fl_sim.cfg.sample_ratio = max(0.1, min(1.0, top_k_ratio))
        self.fl_sim.cfg.local_epochs = local_epochs

        res = self.fl_sim.run(rounds=self.fl_sim.cfg.rounds)
        acc = res["accuracy_mean"]
        time_cost = res["time_mean"]
        fairness = res["fairness_mean"]

        # pymoo 统一为“最小化”，因此对需要最大化的目标取负号
        f1 = -float(acc)
        f2 = float(time_cost)
        f3 = -float(fairness)

        out["F"] = np.array([f1, f2, f3], dtype=float)
