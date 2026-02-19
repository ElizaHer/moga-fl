import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# FedLab imports
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import get_best_gpu
from fedlab.core.standalone.trainer import ParallelTrainer
from fedlab.core.client.handler import ClientTrainer
from fedlab.core.server.handler import SyncServerHandler

# torchvision for dataset and transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, transforms

# 从项目中导入 ResNet 构建函数
from src.training.models.resnet_cifar import build_resnet18_cifar


# ===================================================================
# 1. 自定义客户端训练处理器 (Client Handler)
#    定义每个客户端如何执行本地训练。
# ===================================================================
class FedAvgClientTrainer(ClientTrainer):
    def __init__(self, model, cuda=True, epochs=5, batch_size=128, lr=0.01):
        super().__init__(model, cuda)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def train(self, model_parameters, train_loader):
        self.model.load_state_dict(model_parameters)
        self.model.train()
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.gpu), target.cuda(self.gpu)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()


# ===================================================================
# 2. 准确率测试函数
# ===================================================================
def test_model_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# ===================================================================
# 3. 主执行逻辑
# ===================================================================
if __name__ == "__main__":
    # 超参数设置
    NUM_ROUNDS = 50
    NUM_CLIENTS = 100
    NUM_PER_ROUND = 4  # 每轮并行计算的客户端数量
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 128
    LR = 0.01

    # 检查并设置设备
    device = get_best_gpu() if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 准备模型和数据
    # 直接从项目中导入模型
    model = build_resnet18_cifar(num_classes=10)

    # 数据集和变换
    transform = Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 使用 FedLab 的 Partitioner 来划分数据
    # 这里我们使用 Dirichlet 分布来模拟 Non-IID 数据，更贴近真实场景
    data_dir = '../data'
    partitioner = CIFAR10Partitioner(
        CIFAR10(root=data_dir, train=True, download=True, transform=transform),
        num_clients=NUM_CLIENTS,
        partition="dirichlet",
        dir_alpha=0.4,
        seed=42
    )

    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 2. 设置 FedLab 的 Server 和 Client 处理器
    server_handler = SyncServerHandler(
        model=model,
        global_round=NUM_ROUNDS,
    )

    client_handler = FedAvgClientTrainer(
        model=model,
        epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        cuda=True if device != "cpu" else False
    )
    # 将数据集与客户端处理器关联
    client_handler.setup_dataset(partitioner)

    # 3. 设置并行训练器
    # num_workers 就是并行进程的数量
    trainer = ParallelTrainer(
        server_handler=server_handler,
        client_handler=client_handler,
        num_workers=NUM_PER_ROUND  # 核心：设置并行工作进程数为4
    )

    # 4. 开始训练循环
    print(f"开始 {NUM_ROUNDS} 轮联邦训练，每轮并行 {NUM_PER_ROUND} 个客户端...")
    for round_idx in range(NUM_ROUNDS):
        # 随机选择本轮参与的客户端
        selected_clients = np.random.choice(NUM_CLIENTS, NUM_PER_ROUND, replace=False)

        # 启动单轮训练
        trainer.train(client_ids=list(selected_clients))

        # 在服务器端测试全局模型准确率
        # trainer.model 就是聚合后的全局模型
        accuracy = test_model_accuracy(trainer.model, test_loader, device)

        print(f"第 {round_idx + 1}/{NUM_ROUNDS} 轮 | 全局模型准确率: {accuracy:.2f}%")

    print("\n=== 训练完成 ===")
