import torch
from typing import Dict, Any


def fedprox_regularizer(local_model: torch.nn.Module, global_state: Dict[str, Any], mu: float, device: torch.device) -> torch.Tensor:
    """计算 FedProx 近端正则项 μ/2 ||w - w_global||^2。

    对应 FedProx 论文中每个客户端的本地目标：
        f_k(w) + μ/2 ||w - w_global||^2
    其中 w_global 是当前轮从服务器下发的全局参数。"""
    prox = torch.zeros(1, device=device)
    gstate = {k: v.to(device) for k, v in global_state.items()}
    for name, p in local_model.named_parameters():
        if name in gstate:
            prox = prox + ((p - gstate[name]) ** 2).sum()
    return (mu / 2.0) * prox
