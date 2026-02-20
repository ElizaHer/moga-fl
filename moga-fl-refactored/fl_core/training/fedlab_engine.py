from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.contrib.algorithm import fedprox as fedprox_mod
from fedlab.contrib.algorithm import scaffold as scaffold_mod
from torch.utils.data import DataLoader

from ..data.partition import apply_quick_limit  # 保留以便示例扩展
from ..scheduling.fairness_ledger import FairnessDebtLedger
from ..scheduling.scorer import ClientScorer
from ..scheduling.selector import TopKSelector
from ..wireless.bandwidth import BandwidthAllocator
from ..wireless.channel import ChannelSimulator
from ..wireless.energy import EnergyEstimator
from .aggregator import MogaAggregator
from .strategy_controller import StrategyController
from .utils import eval_model, make_loader


class LocalTrainerBackend:
    """基于 FedLab SerialClientTrainer 的本地训练后端。

    - FedAvg: 使用 :class:`SGDSerialClientTrainer`；
    - FedProx: 使用 :class:`FedProxSerialClientTrainer`；
    - SCAFFOLD: 使用 :class:`ScaffoldSerialClientTrainer`（目前仅在同步模式下使用）。
    """

    def __init__(
        self,
        model_fn,
        num_clients: int,
        training_cfg: Dict[str, Any],
        device: torch.device,
        use_cuda: bool = False,
    ) -> None:
        self.num_clients = num_clients
        self.training_cfg = training_cfg
        self.use_cuda = bool(use_cuda)
        self.device_str: Optional[str] = str(device) if self.use_cuda else None

        epochs = int(training_cfg.get("local_epochs", 1))
        batch_size = int(training_cfg.get("batch_size", 64))
        lr = float(training_cfg.get("lr", 0.01))
        mu = float(training_cfg.get("fedprox_mu", 0.0))

        # FedAvg backend
        self.fedavg_trainer = SGDSerialClientTrainer(
            model_fn(),
            num_clients=num_clients,
            cuda=self.use_cuda,
            device=self.device_str,
        )
        self.fedavg_trainer.setup_optim(epochs, batch_size, lr)

        # FedProx backend
        FedProxSerialClientTrainer = fedprox_mod.FedProxSerialClientTrainer
        self.fedprox_trainer = FedProxSerialClientTrainer(
            model_fn(),
            num_clients=num_clients,
            cuda=self.use_cuda,
            device=self.device_str,
        )
        # FedProx 需要额外的 mu 参数
        self.fedprox_trainer.setup_optim(epochs, batch_size, lr, mu)
        self.fedprox_mu = mu

        # SCAFFOLD backend（仅在需要时使用）
        ScaffoldSerialClientTrainer = scaffold_mod.ScaffoldSerialClientTrainer
        self.scaffold_trainer = ScaffoldSerialClientTrainer(
            model_fn(),
            num_clients=num_clients,
            cuda=self.use_cuda,
            device=self.device_str,
        )
        self.scaffold_trainer.setup_optim(epochs, batch_size, lr)

    # ---------------- 对外接口 -----------------
    def train_fedavg(
        self,
        global_params: torch.Tensor,
        loader: DataLoader,
    ) -> torch.Tensor:
        pack = self.fedavg_trainer.train(global_params, loader)
        # SGDSerialClientTrainer.train 返回 [model_parameters]
        return pack[0]

    def train_fedprox(
        self,
        global_params: torch.Tensor,
        loader: DataLoader,
    ) -> torch.Tensor:
        pack = self.fedprox_trainer.train(global_params, loader, self.fedprox_mu)
        return pack[0]

    def train_scaffold(
        self,
        cid: int,
        global_params: torch.Tensor,
        global_c: torch.Tensor,
        loader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pack = self.scaffold_trainer.train(cid, global_params, global_c, loader)
        dy, dc = pack[0], pack[1]
        return dy, dc


class MogaFLServer:
    """单进程模拟的 MOGA-FL Server，使用 FedLab 1.3.0 作为训练后端。

    该类在工程结构上替代原工程的 ``Server``，但内部：
    - 全局模型由 :class:`ModelMaintainer` 管理（与 FedLab API 对齐）；
    - 本地训练通过 :class:`LocalTrainerBackend` 调用 FedLab 的 SerialClientTrainer；
    - 聚合逻辑由 :class:`MogaAggregator` 完成，并支持 FedBuff 风格的异步聚合；
    - 调度评分、无线仿真、公平债务、门控切换等模块直接复用原工程实现。
    """

    def __init__(
        self,
        model_fn,
        train_dataset,
        test_dataset,
        num_classes: int,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device

        self.num_clients = int(cfg.get("clients", {}).get("num_clients", 10))
        self.training_cfg = cfg.get("training", {})
        self.algorithm = self.training_cfg.get("algorithm", "fedavg")
        self.semi_sync_wait_ratio = float(self.training_cfg.get("semi_sync_wait_ratio", 0.7))

        # 全局模型：由 ModelMaintainer 管理其序列化参数
        use_cuda = device.type == "cuda"
        self._global_model = self.model_fn()
        self._maintainer = ModelMaintainer(
            self._global_model,
            cuda=use_cuda,
            device=str(device) if use_cuda else None,
        )

        # FedLab 本地训练后端
        self.trainer_backend = LocalTrainerBackend(
            model_fn,
            num_clients=self.num_clients,
            training_cfg=self.training_cfg,
            device=device,
            use_cuda=use_cuda,
        )

        # 无线与能耗
        self.channel = ChannelSimulator(cfg, self.num_clients)
        self.bw_alloc = BandwidthAllocator(cfg)
        self.energy = EnergyEstimator(cfg)

        # 调度与公平性
        self.ledger = FairnessDebtLedger(cfg, self.num_clients)
        self.scorer = ClientScorer(cfg, self.num_clients, self.ledger)
        self.selector = TopKSelector(
            cfg["clients"].get("selection_top_k", max(1, self.num_clients // 2)),
            cfg["clients"].get("sliding_window", 5),
            cfg["clients"].get("hysteresis", 0.05),
        )

        # 策略控制器与聚合器
        self.strategy = StrategyController(cfg, self.num_clients)
        self.aggregator = MogaAggregator(cfg)

        # SCAFFOLD：维护全局控制向量 c_global（序列化形式）
        self.scaffold_global_c: Optional[torch.Tensor] = None
        if self.algorithm == "scaffold":
            self.scaffold_global_c = torch.zeros_like(self._maintainer.model_parameters)

    # ---------------- 辅助属性 -----------------
    @property
    def global_params(self) -> torch.Tensor:
        return self._maintainer.model_parameters

    def set_global_params(self, params: torch.Tensor) -> None:
        self._maintainer.set_model(params)

    # ---------------- API：客户端划分初始化 -----------------
    def init_partitions(self, partitions: Dict[int, List[int]]) -> None:
        """占位函数以兼容原工程的接口。

        在当前实现中，partition 仅在 :meth:`round` 内部按需访问。该方法保留
        主要是为了与原脚本形式保持接近。
        """
        self._partitions = partitions

    # ---------------- 单轮联邦训练 -----------------
    def round(self, r: int, partitions: Dict[int, List[int]]) -> Dict[str, Any]:
        # 信道采样
        stats = self.channel.sample_round()
        avg_per = float(np.mean([stats[cid]["per"] for cid in range(self.num_clients)]))

        # 根据历史指标决定聚合模式与带宽缩放
        mode, bridge_w, bw_factor = self.strategy.decide_mode(r)

        # 构造调度所需特征
        energy_avail = [
            1.0
            - min(1.0, self.energy.compute_energy(len(partitions[cid])) / 10.0)
            for cid in range(self.num_clients)
        ]
        channel_quality = [1.0 - stats[cid]["per"] for cid in range(self.num_clients)]
        data_value = [len(partitions[cid]) for cid in range(self.num_clients)]
        bandwidth_cost = [len(partitions[cid]) / 1000.0 for cid in range(self.num_clients)]

        scores = self.scorer.score(energy_avail, channel_quality, data_value, bandwidth_cost)
        selected = self.selector.select(scores)

        # 带宽预算缩放
        orig_budget = self.bw_alloc.budget_mb
        self.bw_alloc.budget_mb = orig_budget * float(bw_factor)
        bw_map = self.bw_alloc.allocate_uniform(selected)

        # 估算发送时间（用于半同步判定）
        tx_times = {
            cid: self.bw_alloc.estimate_tx_time(payload_mb=1.0, allocated_mb=bw_map.get(cid, 0.0))
            for cid in selected
        }
        on_time_clients = list(selected)
        if mode in ("semi_sync", "bridge") and len(tx_times) > 0:
            t_values = np.array(list(tx_times.values()), dtype=float)
            threshold = float(np.quantile(t_values, self.semi_sync_wait_ratio))
            on_time_clients = [cid for cid in selected if tx_times[cid] <= threshold]
            late_clients = set(selected) - set(on_time_clients)
            if late_clients:
                print(f"[round {r}] late clients: {late_clients}")

        # 本地更新收集
        client_params: List[torch.Tensor] = []
        weights: List[float] = []
        staleness: List[int] = []
        scaffold_dys: List[torch.Tensor] = []
        scaffold_dcs: List[torch.Tensor] = []

        global_params = self.global_params

        for cid in selected:
            per = stats[cid]["per"]
            if np.random.rand() < per:
                continue  # 丢包

            if cid not in on_time_clients and mode in ("semi_sync", "bridge"):
                continue

            indices = partitions[cid]
            loader = make_loader(
                self.train_dataset,
                indices,
                batch_size=self.training_cfg.get("batch_size", 64),
            )

            if self.algorithm == "fedavg":
                local_params = self.trainer_backend.train_fedavg(global_params, loader)
                client_params.append(local_params)
                weights.append(len(indices))
                staleness.append(0)
            elif self.algorithm == "fedprox":
                local_params = self.trainer_backend.train_fedprox(global_params, loader)
                client_params.append(local_params)
                weights.append(len(indices))
                staleness.append(0)
            elif self.algorithm == "scaffold":
                if self.scaffold_global_c is None:
                    self.scaffold_global_c = torch.zeros_like(global_params)
                dy, dc = self.trainer_backend.train_scaffold(
                    cid, global_params, self.scaffold_global_c, loader
                )
                scaffold_dys.append(dy)
                scaffold_dcs.append(dc)
                weights.append(len(indices))
            else:
                raise ValueError(f"Unsupported training.algorithm: {self.algorithm}")

        # 聚合
        if self.algorithm in {"fedavg", "fedprox"}:
            if client_params or mode in ("async", "bridge"):
                new_global = self.aggregator.aggregate(
                    global_params,
                    client_params,
                    weights,
                    staleness_list=staleness,
                    round_idx=r,
                    mode=mode,
                    bridge_weight=bridge_w,
                )
                self.set_global_params(new_global)
        elif self.algorithm == "scaffold":
            if scaffold_dys:
                # 近似实现 ScaffoldServerHandler.global_update
                dx = torch.stack(scaffold_dys, dim=0).mean(dim=0)
                dc = torch.stack(scaffold_dcs, dim=0).mean(dim=0)
                lr = float(self.training_cfg.get("lr", 0.01))
                next_model = global_params + lr * dx
                self.set_global_params(next_model)
                assert self.scaffold_global_c is not None
                self.scaffold_global_c = self.scaffold_global_c + (
                    len(scaffold_dcs) / self.num_clients
                ) * dc

        # 公平债务更新
        self.ledger.on_round_end(selected)

        # 评估当前全局模型
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128)
        acc = eval_model(self._maintainer.model, test_loader, self.device)

        # 粗略通信与能耗估计
        comm_time = sum(
            self.bw_alloc.estimate_tx_time(1.0, bw_map.get(cid, 0.0)) for cid in selected
        )
        comm_energy = sum(
            self.energy.comm_energy(
                self.bw_alloc.estimate_tx_time(1.0, bw_map.get(cid, 0.0))
            )
            for cid in selected
        )
        comp_energy = sum(self.energy.compute_energy(len(partitions[cid])) for cid in selected)

        # 更新策略控制器历史
        self.strategy.register_round_metrics(
            r,
            avg_per=avg_per,
            jain_index=self.jain_index_selection(),
            total_energy=comm_energy + comp_energy,
        )

        # 恢复带宽预算
        self.bw_alloc.budget_mb = orig_budget

        return {
            "round": r,
            "selected": selected,
            "accuracy": acc,
            "comm_time": comm_time,
            "comm_energy": comm_energy,
            "comp_energy": comp_energy,
            "jain_index": self.jain_index_selection(),
            "sync_mode": mode,
            "avg_per": avg_per,
            "bandwidth_factor": float(bw_factor),
        }

    # ---------------- Jain 公平指数 -----------------
    def jain_index_selection(self) -> float:
        freq = np.zeros(self.num_clients)
        for hist in self.selector.history:
            for cid in hist:
                freq[cid] += 1
        if freq.sum() == 0:
            return 0.0
        return float((freq.sum() ** 2) / (self.num_clients * (freq ** 2).sum() + 1e-9))
