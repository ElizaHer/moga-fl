from __future__ import annotations

from typing import Dict, Any, Tuple
import torch


class ScaffoldState:
    """SCAFFOLD 控制变元状态。

    对应 SCAFFOLD 论文中的全局控制向量 c 与每个客户端的 c_i：
    - 服务器维护 c_global（与模型参数同形状的向量）；
    - 每个客户端维护自己的 c_local[cid]；
    - 本地更新时使用 g + c_i - c_global 作为“校正梯度”，减缓 non-IID 漂移；
    - 本地训练结束后，客户端返回 Δc_i，用于在服务器端更新全局 c_global。"""

    def __init__(self, global_state: Dict[str, torch.Tensor]):
        # 初始化为全 0 控制向量，形状与模型参数一致
        self.c_global: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in global_state.items()
        }
        self.c_local: Dict[int, Dict[str, torch.Tensor]] = {}

    # ---------------------- 客户端控制向量管理 ----------------------
    def ensure_client(self, cid: int, global_state: Dict[str, torch.Tensor]) -> None:
        if cid not in self.c_local:
            self.c_local[cid] = {k: torch.zeros_like(v) for k, v in global_state.items()}

    def get_controls(self, cid: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        assert cid in self.c_local, "call ensure_client before get_controls"
        return self.c_global, self.c_local[cid]

    # ---------------------- Δc_i 计算与更新 ----------------------
    def compute_delta_ci(
        self,
        cid: int,
        global_state: Dict[str, torch.Tensor],
        local_state: Dict[str, torch.Tensor],
        lr: float,
        num_local_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """根据 SCAFFOLD 论文的近似更新公式计算 Δc_i 并更新 c_local[cid]。

        近似公式：
            c_i^{new} = c_i - c + (1 / (η * τ)) (w_global - w_local)
        其中 η 为学习率，τ 为本地 SGD 步数。
        服务器稍后会对 Δc_i = c_i^{new} - c_i 做平均，更新全局 c。"""
        old_ci = self.c_local[cid]
        new_ci: Dict[str, torch.Tensor] = {}
        delta_ci: Dict[str, torch.Tensor] = {}
        scale = 1.0 / max(1, int(num_local_steps)) / max(1e-8, lr)
        for k in global_state.keys():
            w_g = global_state[k]
            w_l = local_state[k]
            c = self.c_global[k]
            ci = old_ci[k]
            # (w_global - w_local) 项
            diff = (w_g - w_l) * scale
            ci_new = ci - c + diff
            new_ci[k] = ci_new
            delta_ci[k] = ci_new - ci
        self.c_local[cid] = new_ci
        return delta_ci

    def update_global(self, delta_list: Dict[int, Dict[str, torch.Tensor]], weights: Dict[int, float]) -> None:
        """根据所有参与客户端的 Δc_i 更新全局控制向量 c_global。

        这里采用加权平均（权重通常取客户端样本数），对应论文中的
        c ← c + (1/|S_t|) Σ_i (c_i^{new} - c_i)。"""
        if not delta_list:
            return
        total_w = sum(float(w) for w in weights.values())
        if total_w <= 0:
            total_w = 1.0
        for k in self.c_global.keys():
            agg = None
            for cid, delta_ci in delta_list.items():
                w = float(weights.get(cid, 1.0)) / total_w
                term = delta_ci[k] * w
                agg = term if agg is None else agg + term
            if agg is not None:
                self.c_global[k] = self.c_global[k] + agg
