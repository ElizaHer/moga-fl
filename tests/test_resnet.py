import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from src.training.models.resnet_cifar import build_resnet18_cifar


def load_cifar10_data():
    """加载CIFAR-10数据集"""
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='../data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def create_simple_partitions(train_dataset, num_clients=10, samples_per_client=500):
    """创建简单的数据分区"""
    total_samples = len(train_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)

    partitions = {}
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min((i + 1) * samples_per_client, total_samples)
        partitions[i] = indices[start_idx:end_idx]

    return partitions


class SimpleClient:
    """简化版客户端，用于FedAvg测试"""

    def __init__(self, cid, model, train_dataset, indices, device):
        self.cid = cid
        self.model = model
        self.train_dataset = train_dataset
        self.indices = indices
        self.device = device

    def local_train(self, global_state, local_epochs=1, batch_size=32, lr=0.01, round_idx=None):
        """本地训练"""
        self.model.load_state_dict(global_state)
        self.model.train()

        # 创建数据加载器
        subset = Subset(self.train_dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # 优化器和损失函数
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # 每10轮降低学习率
        if round_idx is not None and round_idx > 0 and round_idx % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (0.5 ** (round_idx // 10))

        # 训练循环
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            val_loss = criterion(self.model(data), target)
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        accuracy = test_model_accuracy(self.model, loader, self.device)
        print(f"Client {self.cid}, Epoch {epoch}, Accuracy: {accuracy:.2f}%")

        # 返回更新后的模型状态
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


def test_model_accuracy(model, test_loader, device):
    """测试模型准确率"""
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


def main():
    # 设置参数
    num_clients = 1
    samples_per_client = 500
    num_rounds = 1000
    local_epochs = 10
    batch_size = 128
    lr = 0.1

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载CIFAR-10数据集...")
    train_dataset, test_dataset = load_cifar10_data()

    # 创建数据分区
    partitions = create_simple_partitions(train_dataset, num_clients, samples_per_client)

    # 初始化全局模型
    print("初始化ResNet-18模型...")
    global_model = build_resnet18_cifar(num_classes=10, width_factor=1.0).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    # 创建客户端
    clients = []
    for cid in range(num_clients):
        model = build_resnet18_cifar(num_classes=10, width_factor=1.0).to(device)
        client = SimpleClient(cid, model, train_dataset, partitions[cid], device)
        clients.append(client)

    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 训练循环
    print(f"开始{num_rounds}轮联邦训练...")
    accuracy_history = []
    best_accuracy = 0.0
    patience = 20  # 早停耐心值
    no_improvement_count = 0

    for round_idx in range(num_rounds):
        # print(f"\n=== 第 {round_idx + 1} 轮 ===")

        # 随机选择部分客户端
        selected_clients = np.random.choice(num_clients, size=max(1, num_clients // 2), replace=False)

        client_states = []
        weights = []

        # 客户端本地训练
        for cid in selected_clients:
            # print(f"客户端 {cid} 训练中...")
            client_update = clients[cid].local_train(
                global_state, local_epochs, batch_size, lr, round_idx
            )
            client_states.append(client_update)
            weights.append(len(partitions[cid]))

        # 服务器聚合
        if client_states:
            global_state = client_states[0]

        # 更新全局模型并测试准确率
        global_model.load_state_dict(global_state)
        accuracy = test_model_accuracy(global_model, test_loader, device)
        accuracy_history.append(accuracy)

        print(f"第 {round_idx + 1} 轮准确率: {accuracy:.2f}%")

        # 早停机制
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            # 保存最佳模型
            torch.save(global_state, 'outputs/test_results/best_model.pth')
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience and round_idx > 100:
            print(f"早停触发，连续{patience}轮没有改进")
            break

    # 输出最终结果
    print(f"\n=== 训练完成 ===")
    print(f"最终准确率: {accuracy_history[-1]:.2f}%")
    print(f"最佳准确率: {max(accuracy_history):.2f}%")
    print(f"准确率历史: {[f'{acc:.2f}%' for acc in accuracy_history]}")

    # 保存结果
    os.makedirs('outputs/test_results', exist_ok=True)
    with open('outputs/test_results/fedavg_resnet_accuracy.txt', 'w') as f:
        f.write(f"FedAvg + ResNet-18 {num_rounds}轮训练结果\n")
        f.write("=" * 50 + "\n")
        for i, acc in enumerate(accuracy_history):
            f.write(f"第 {i + 1} 轮: {acc:.2f}%\n")
        f.write(f"\n最终准确率: {accuracy_history[-1]:.2f}%\n")
        f.write(f"最佳准确率: {max(accuracy_history):.2f}%\n")


if __name__ == "__main__":
    main()