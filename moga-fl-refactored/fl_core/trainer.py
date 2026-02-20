from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .fedlab_wrappers import SimpleClientTrainer, GlobalModel
from .models import build_model


@dataclass
class FLConfig:
    num_clients: int = 10
    sample_ratio: float = 0.5
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.1
    rounds: int = 5
    test_batch_size: int = 128


def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return float(correct) / max(1, total)


class FLSimulation:
    """基于 FedLab 抽象封装的极简联邦学习仿真器。

    - 使用 GlobalModel 维护全局模型；
    - 每个客户端使用 SimpleClientTrainer（FedLab ClientTrainer 子类）进行本地训练；
    - 聚合阶段采用标准 FedAvg（按样本数加权平均）；
    - 不依赖 torch.distributed 或 NetworkManager，适合在单机 CPU 上快速运行
      与 GA 评估。
    """

    def __init__(
        self,
        train_loaders: Dict[int, DataLoader],
        test_loader: DataLoader,
        cfg: FLConfig,
        device: torch.device | None = None,
        num_classes: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_clients = cfg.num_clients
        self.test_loader = test_loader

        # 推断类别数
        self.num_classes = num_classes or self._infer_num_classes(train_loaders)

        # 全局模型
        base_model = build_model(num_classes=self.num_classes)
        self.global_model = GlobalModel(base_model, cuda=self.device.type == "cuda")

        # 客户端训练器
        self.clients: Dict[int, SimpleClientTrainer] = {}
        self.client_sample_counts: Dict[int, int] = {}
        for cid in range(self.num_clients):
            loader = train_loaders[cid]
            self.client_sample_counts[cid] = len(loader.dataset)
            cl_model = build_model(num_classes=self.num_classes)
            self.clients[cid] = SimpleClientTrainer(
                model=cl_model,
                train_loader=loader,
                epochs=cfg.local_epochs,
                lr=cfg.lr,
                cuda=self.device.type == "cuda",
                device=str(self.device) if self.device.type == "cuda" else None,
            )

        # 追踪选择频率以计算 Jain 公平性
        self.selection_history: List[List[int]] = []

    @staticmethod
    def _infer_num_classes(train_loaders: Dict[int, DataLoader]) -> int:
        # 简单从所有客户端数据中推断类别数
        max_label = 0
        for loader in train_loaders.values():
            for _, y in loader:
                max_label = max(max_label, int(y.max().item()))
        return int(max_label + 1)

    def _sample_clients(self) -> List[int]:
        m = max(1, int(self.num_clients * self.cfg.sample_ratio))
        return list(np.random.choice(self.num_clients, size=m, replace=False))

    def _fedavg_aggregate(self, client_params: List[torch.Tensor], weights: List[int]) -> torch.Tensor:
        assert client_params, "no client params to aggregate"
        total = float(sum(weights))
        stacked = torch.stack(client_params, dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1)
        return (stacked * (w / total)).sum(dim=0)

    def _jain_index(self) -> float:
        if not self.selection_history:
            return 0.0
        freq = np.zeros(self.num_clients, dtype=float)
        for hist in self.selection_history:
            for cid in hist:
                freq[cid] += 1
        if freq.sum() == 0:
            return 0.0
        num = freq.sum() ** 2
        den = self.num_clients * (freq ** 2).sum() + 1e-9
        return float(num / den)

    def run(self, rounds: int | None = None) -> Dict[str, float]:
        rounds = rounds or self.cfg.rounds
        acc_list: List[float] = []
        time_list: List[float] = []
        fairness_list: List[float] = []

        for r in range(rounds):
            selected = self._sample_clients()
            self.selection_history.append(selected)

            downlink = [self.global_model.model_parameters]
            client_params: List[torch.Tensor] = []
            weights: List[int] = []

            for cid in selected:
                client = self.clients[cid]
                client.local_process(downlink)
                client_params.append(client.model_parameters)
                weights.append(self.client_sample_counts[cid])

            agg_params = self._fedavg_aggregate(client_params, weights)
            self.global_model.set_model(agg_params)

            # 评估当前全局模型
            model = build_model(num_classes=self.num_classes)
            model.load_state_dict(self.global_model.model.state_dict())
            model.to(self.device)
            acc = eval_model(model, self.test_loader, self.device)
            acc_list.append(acc)

            # 这里用“参与客户端数 × 本地 epoch”作为简化的时间/能耗 proxy
            time_cost = len(selected) * self.cfg.local_epochs
            time_list.append(float(time_cost))
            fairness_list.append(self._jain_index())

        return {
            "accuracy_mean": float(np.mean(acc_list)),
            "time_mean": float(np.mean(time_list)),
            "fairness_mean": float(np.mean(fairness_list)),
        }
