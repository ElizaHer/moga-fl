import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Tuple


# ----------------------
# 1. 模型定义（同你的 ResNet18）
# ----------------------
def build_resnet18_cifar(num_classes=10):
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ----------------------
# 2. 数据工具：Dirichlet 分区（Client 侧使用，Server 无感）
# ----------------------
def dirichlet_partition(dataset, num_clients: int, alpha: float = 0.4, seed: int = 42) -> List[DataLoader]:
    """按 Dirichlet 分布划分 CIFAR10，返回每个 Client 的 DataLoader"""
    np.random.seed(seed)
    targets = np.array([target for _, target in dataset])
    num_classes = len(np.unique(targets))

    # 按类别分组
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    # Dirichlet 采样
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], proportions)
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    # 构建每个 Client 的 DataLoader
    loaders = []
    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    for indices in client_indices:
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)
        loaders.append(loader)
    return loaders


# ----------------------
# 3. Client 实现（每个 Client 管自己的数据，Server 不知道 cid）
# ----------------------
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader, epochs: int, lr: float):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.testloader)
        acc = correct / len(self.testloader.dataset)
        return loss, len(self.testloader.dataset), {"accuracy": acc}


# ----------------------
# 4. Server 实现（仅聚合，无数据/cid 依赖，评估用全局测试集）
# ----------------------
def get_evaluate_fn(model: nn.Module, testloader: DataLoader):
    """Server 侧全局评估函数：用全局测试集，无 cid"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, Dict[str, Scalar]]:
        # 加载全局模型
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(testloader)
        acc = correct / len(testloader.dataset)
        print(f"Round {server_round}: Global Test Loss {loss:.4f}, Acc {acc:.4f}")
        return loss, {"accuracy": acc}

    return evaluate


# ----------------------
# 5. 主程序（启动联邦训练）
# ----------------------
if __name__ == "__main__":
    # 超参数
    NUM_CLIENTS = 100
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 5
    LR = 0.01
    FRACTION_FIT = 0.1  # 每轮采样 10% 客户端

    # 数据准备
    transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = CIFAR10(root="../data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="../data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Dirichlet 分区（Client 侧数据，Server 完全不知道细节）
    client_trainloaders = dirichlet_partition(trainset, NUM_CLIENTS, alpha=0.4)
    # 每个 Client 用自己的小测试集（可选）
    client_testloaders = [
        DataLoader(random_split(testset, [len(testset) // NUM_CLIENTS] * NUM_CLIENTS)[i], batch_size=128) for i in
        range(NUM_CLIENTS)]

    # 模型
    model = build_resnet18_cifar()


    # 启动 Flower
    # 1) 定义 Client 函数
    def client_fn(cid: str) -> CifarClient:
        cid_int = int(cid)
        return CifarClient(
            model=build_resnet18_cifar(),  # 每个 Client 独立初始化模型
            trainloader=client_trainloaders[cid_int],
            testloader=client_testloaders[cid_int],
            epochs=LOCAL_EPOCHS,
            lr=LR
        )


    # 2) 启动 Server（含全局评估）
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=0.0,  # 不做 Client 侧评估，只做 Server 全局评估
        min_fit_clients=int(NUM_CLIENTS * FRACTION_FIT),
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(model, testloader),  # Server 全局评估
    )

    # 3) 启动模拟（单机多 Client）
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )