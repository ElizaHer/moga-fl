from __future__ import annotations

import flwr as fl
from flwr.common import Scalar

from src.training.algorithms.fedprox import fedprox_regularizer
from src.training.models.resnet import build_resnet18
from src.utils.train import *


class CifarClient(fl.client.NumPyClient):  # type: ignore[misc]
    def __init__(
        self,
        cid: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        local_epochs: int,
        lr: float,
    ) -> None:
        self.cid = cid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18().to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        # T_max 取一个相对较大的值，近似余弦退火
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.scaffold_ci: Dict[str, torch.Tensor] | None = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:  # type: ignore[override]
        del config
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:  # type: ignore[override]
        set_parameters(self.model, parameters)
        algo = str(config.get("algorithm", "fedavg")).lower()
        mu = float(config.get("fedprox_mu", 0.0))
        criterion = nn.CrossEntropyLoss()
        global_state = global_state_from_ndarrays(self.model, parameters, self.device)
        delta_ci_bytes = b""

        # SCAFFOLD: 读取 c_global，若首次出现则初始化本地 c_i
        c_global: Dict[str, torch.Tensor] = {}
        if algo == "scaffold":
            c_global = unpack_tensor_dict(bytes(config.get("scaffold_c_global", b"")), self.device)
            if self.scaffold_ci is None:
                self.scaffold_ci = {k: torch.zeros_like(v) for k, v in c_global.items()}

        self.model.train()
        num_steps = 0
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                if algo == "fedprox" and mu > 0.0:
                    loss = loss + fedprox_regularizer(self.model, global_state, mu, self.device)

                loss.backward()
                if algo == "scaffold" and self.scaffold_ci is not None:
                    # SCAFFOLD 梯度校正: grad <- grad + (c_i - c_global)
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        ci = self.scaffold_ci.get(name)
                        cg = c_global.get(name)
                        if ci is not None and cg is not None:
                            p.grad.data = p.grad.data + (ci - cg)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                num_steps += 1

        if algo == "scaffold" and self.scaffold_ci is not None:
            local_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            lr = float(self.optimizer.param_groups[0].get("lr", 0.001))
            scale = 1.0 / max(1, int(num_steps)) / max(1e-8, lr)
            delta_ci: Dict[str, torch.Tensor] = {}
            new_ci: Dict[str, torch.Tensor] = {}
            for k in c_global.keys():
                old_ci = self.scaffold_ci[k]
                w_g = global_state[k]
                w_l = local_state[k].to(self.device)
                c_g = c_global[k]
                ci_new = old_ci - c_g + (w_g - w_l) * scale
                delta_ci[k] = ci_new - old_ci
                new_ci[k] = ci_new
            self.scaffold_ci = new_ci
            delta_ci_bytes = pack_tensor_dict(delta_ci)

        return get_parameters(self.model), len(self.trainloader.dataset), {
            "cid": float(self.cid),
            "delta_ci": delta_ci_bytes,
        }

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:  # type: ignore[override]
        del config
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}

