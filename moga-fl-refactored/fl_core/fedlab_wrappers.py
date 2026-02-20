from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.model_maintainer import ModelMaintainer


class SimpleClientTrainer(ClientTrainer):
    """基于 FedLab ClientTrainer 封装的简易本地训练器。

    - 继承 FedLab 1.3.0 的 ClientTrainer 基类；
    - 实现 local_process / uplink_package / train 等抽象接口；
    - 在本工程中通过显式循环而非 NetworkManager 进行调用，便于与
      无线调度与 GA 搜索集成。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        cuda: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__(model=model, cuda=cuda, device=device)
        self.train_loader = train_loader
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    # --------- FedLab 抽象接口实现 ---------
    def setup_dataset(self):  # type: ignore[override]
        # 在本工程中直接通过构造函数传入 DataLoader，因此此处无需额外逻辑
        return None

    def setup_optim(self):  # type: ignore[override]
        # 同理，优化器在构造函数中完成初始化
        return None

    @property  # type: ignore[override]
    def uplink_package(self) -> List[torch.Tensor]:
        """客户端向服务器上传的内容：这里简单返回更新后的模型参数。"""

        return [self.model_parameters]

    def local_process(self, payload: List[torch.Tensor]):  # type: ignore[override]
        """本地过程：接收全局模型参数，执行若干轮本地训练。

        payload[0] 约定为序列化后的全局模型参数。
        """

        if payload:
            self.set_model(payload[0])
        self.train()
        return True

    def train(self):  # type: ignore[override]
        self.model.train()
        device = getattr(self, "device", torch.device("cpu"))
        self.model.to(device)
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

    def validate(self):  # type: ignore[override]
        # 示例工程中未使用到，可按需扩展
        return None

    def evaluate(self):  # type: ignore[override]
        # 示例工程中未使用到，可按需扩展
        return None


class GlobalModel(ModelMaintainer):
    """对 FedLab ModelMaintainer 的轻量封装，表示全局模型。"""

    def __init__(self, model: nn.Module, cuda: bool = False, device: str | None = None) -> None:
        super().__init__(model=model, cuda=cuda, device=device)
