import torch
from typing import Dict, Any, List
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .algorithms.fedprox import fedprox_regularizer


class Client:
    def __init__(self, cid: int, model_fn, train_dataset, indices: List[int], cfg: Dict[str, Any], device):
        self.cid = cid
        self.model_fn = model_fn
        self.train_dataset = train_dataset
        self.indices = indices
        self.cfg = cfg
        self.device = device
        self.model = self.model_fn().to(self.device)

    def local_train(self, global_state: Dict[str, Any]):
        """本地训练（FedAvg / FedProx 模式）。

        - 当 training.algorithm == "fedprox" 且 fedprox_mu>0 时，在标准交叉熵损失上
          叠加 μ/2 ||w - w_global||^2 对应 FedProx 论文中的近端正则。
        - 否则退化为普通 FedAvg 本地 SGD。"""
        algo = self.cfg['training'].get('algorithm', 'fedavg')
        self.model.load_state_dict(global_state)
        self.model.train()
        loader = DataLoader(
            Subset(self.train_dataset, self.indices),
            batch_size=self.cfg['training'].get('batch_size', 64),
            shuffle=True,
        )
        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg['training'].get('lr', 0.01),
            momentum=self.cfg['training'].get('momentum', 0.9),
        )
        criterion = nn.CrossEntropyLoss()
        mu = float(self.cfg['training'].get('fedprox_mu', 0.0)) if algo == 'fedprox' else 0.0

        for _ in range(self.cfg['training'].get('local_epochs', 1)):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                # FedProx：在普通 loss 上叠加近端正则项
                if mu > 0.0:
                    loss = loss + fedprox_regularizer(self.model, global_state, mu, self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def local_train_scaffold(self, global_state: Dict[str, torch.Tensor], scaffold_state) -> Dict[str, Any]:
        """SCAFFOLD 模式下的本地训练。

        - 使用控制变元 c_global, c_i 对梯度做校正：grad ← grad + (c_i - c_global)；
        - 训练结束后，调用 scaffold_state.compute_delta_ci 计算 Δc_i，供服务器更新全局 c。"""
        from .algorithms.scaffold import ScaffoldState  # 延迟导入避免循环

        assert isinstance(scaffold_state, ScaffoldState)
        self.model.load_state_dict(global_state)
        self.model.train()
        loader = DataLoader(
            Subset(self.train_dataset, self.indices),
            batch_size=self.cfg['training'].get('batch_size', 64),
            shuffle=True,
        )
        lr = self.cfg['training'].get('lr', 0.01)
        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=self.cfg['training'].get('momentum', 0.9),
        )
        criterion = nn.CrossEntropyLoss()

        scaffold_state.ensure_client(self.cid, global_state)
        c_global, c_local = scaffold_state.get_controls(self.cid)
        # 移到设备上以避免重复拷贝
        c_global_dev = {k: v.to(self.device) for k, v in c_global.items()}
        c_local_dev = {k: v.to(self.device) for k, v in c_local.items()}

        num_steps = 0
        for _ in range(self.cfg['training'].get('local_epochs', 1)):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                # 梯度校正：g ← g + c_i - c
                for name, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    ci = c_local_dev.get(name)
                    cg = c_global_dev.get(name)
                    if ci is not None and cg is not None:
                        p.grad.data = p.grad.data + (ci - cg)
                opt.step()
                num_steps += 1

        local_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        delta_ci = scaffold_state.compute_delta_ci(self.cid, global_state, local_state, lr=lr, num_local_steps=num_steps)
        return {'state': local_state, 'delta_ci': delta_ci}
