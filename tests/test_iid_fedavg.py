import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Context, NDArrays, Scalar
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
from src.training.models.resnet_cifar import build_resnet18_cifar


def get_parameters(model: nn.Module) -> NDArrays:
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in zip(model.state_dict().keys(), parameters)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(1, len(dataloader)), correct / max(1, total)


def iid_partition_indices(
    num_samples: int, num_clients: int, seed: int
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int,
        lr: float,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_resnet18_cifar().to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.model.train()
        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        self.scheduler.step()
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        set_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}


def get_server_evaluate_fn(testloader: DataLoader):
    model = build_resnet18_cifar()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, Dict[str, Scalar]]:
        set_parameters(model, parameters)
        loss, acc = evaluate_model(model, testloader, device)
        print(f"[Server] Round {server_round}: loss={loss:.4f}, acc={acc:.4f}")
        return float(loss), {"accuracy": float(acc)}

    return evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower FedAvg + ResNet18 + CIFAR10 simulation (IID)"
    )
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fraction-fit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_transform = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    trainset = CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    testset = CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    partitions = iid_partition_indices(
        num_samples=len(trainset),
        num_clients=args.num_clients,
        seed=args.seed,
    )

    trainloaders = [
        DataLoader(
            Subset(trainset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        for indices in partitions
    ]

    valloader_per_client = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config.get("partition-id", context.node_id))
        return CifarClient(
            trainloader=trainloaders[client_id],
            valloader=valloader_per_client,
            local_epochs=args.local_epochs,
            lr=args.lr,
        ).to_client()

    min_fit_clients = max(1, int(args.num_clients * args.fraction_fit))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_server_evaluate_fn(testloader),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
        ray_init_args={"include_dashboard": False}
    )


if __name__ == "__main__":
    main()
