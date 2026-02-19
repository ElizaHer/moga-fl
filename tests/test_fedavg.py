import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from src.training.models.resnet_cifar import build_resnet18_cifar
from src.training.algorithms.fedavg import aggregate_fedavg
from src.data.partition import cifar_iid_partitions, cifar_dirichlet_partitions, cifar_shards_partitions

import torch.multiprocessing as mp
from multiprocessing import Queue

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ====================== Cutout数据增强 ======================
class Cutout(object):
    def __init__(self, length=8):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img


# ====================== 1. 数据加载 ======================
def load_cifar10_data():
    """加载CIFAR-10数据集"""
    train_transform = Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(length=8)
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='../data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


# ====================== 2. 客户端类 ======================
class SimpleClient:
    def __init__(self, cid, model, train_dataset, indices, device, test_loader=None):
        self.cid = cid
        self.model = model
        self.train_dataset = train_dataset
        self.indices = indices
        self.device = device
        self.test_loader = test_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        self.scheduler = None

    def local_train(self, global_state, local_epochs=1, batch_size=32, lr=0.01, round_idx=None):
        """本地训练（单客户端）"""
        self.model.load_state_dict(global_state)
        self.model.train()

        subset = Subset(self.train_dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )

        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        self.scheduler.step()

        train_accuracy = test_model_accuracy(self.model, loader, self.device)
        print(f"Client {self.cid} Train Accuracy: {train_accuracy:.2f}%")

        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


# ====================== 3. 并行训练函数 ======================
def client_train_worker(client, global_state, local_epochs, batch_size, lr, round_idx, result_queue):
    """客户端训练工作进程（用于多进程并行）"""
    try:
        # 执行本地训练
        client_state = client.local_train(global_state, local_epochs, batch_size, lr, round_idx)
        # 将结果放入队列（客户端ID + 模型状态 + 数据量）
        result_queue.put((client.cid, client_state, len(client.indices)))
        print(f"Client {client.cid} training completed, queue size: {result_queue.qsize()}")
    except Exception as e:
        print(f"Client {client.cid} training failed: {str(e)}")
        try:
            result_queue.put((client.cid, None, 0))
        except:
            pass


def parallel_client_train(clients, selected_clients, global_state, local_epochs, batch_size, lr, round_idx, parallel_k):
    # 初始化结果队列
    result_queue = Queue()
    processes = []
    client_states = []
    weights = []

    # 分批次并行训练（每批最多parallel_k个客户端）
    for i in range(0, len(selected_clients), parallel_k):
        batch_clients = selected_clients[i:i + parallel_k]
        batch_processes = []

        # 为批次内的客户端创建进程
        for cid in batch_clients:
            client = clients[cid]
            # 复制全局模型状态（避免进程间参数共享）
            global_state_copy = {k: v.clone() for k, v in global_state.items()}
            # 创建进程
            p = mp.Process(
                target=client_train_worker,
                args=(client, global_state_copy, local_epochs, batch_size, lr, round_idx, result_queue)
            )
            p.daemon = True
            batch_processes.append(p)
            p.start()
            print(f"Process {p.pid} started for client {cid}")

        # 等待批次内进程完成
        for p in batch_processes:
            print(f"Process {p.pid} waiting")
            p.join(timeout=60)  # 添加超时时间，避免无限等待
            if p.is_alive():
                print(f"Process {p.pid} timed out, terminating")
                p.terminate()
            print(f"Process {p.pid} completed")
            processes.append(p)
            print(f"Process {p.pid} completed for client {cid}")

    # 从队列中收集结果
    while not result_queue.empty():
        print(f"Queue size: {result_queue.qsize()}")
        cid, state, weight = result_queue.get()
        if state is not None:
            print(f"Received result for client {cid}, queue size: {result_queue.qsize()}")
            client_states.append(state)
            weights.append(weight)
        else:
            print(f"Client {cid} returned None state")

    return client_states, weights


# ====================== 4. 准确率测试 ======================
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


# ====================== 5. 主函数（调整参数+放宽早停） ======================
def main():
    # 设置参数
    num_clients = 10
    num_rounds = 100
    local_epochs = 1
    batch_size = 128
    lr = 0.001
    select_ratio = 0.1
    client_parallelism = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"并行训练客户端数: {client_parallelism}")

    print("加载CIFAR-10数据集...")
    train_dataset, test_dataset = load_cifar10_data()

    partitions = cifar_iid_partitions(train_dataset, num_clients)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("初始化ResNet-18模型...")
    global_model = build_resnet18_cifar(num_classes=10).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    # 创建客户端
    clients = []
    for cid in range(num_clients):
        model = build_resnet18_cifar(num_classes=10).to(device)
        client = SimpleClient(cid, model, train_dataset, partitions[cid], device, test_loader)
        clients.append(client)

    # 训练循环
    print(f"开始{num_rounds}轮联邦训练（并行k={client_parallelism}）...")
    accuracy_history = []
    best_accuracy = 0.0
    patience = 20  # 早停耐心值
    no_improvement_count = 0

    for round_idx in range(num_rounds):
        # 随机选择60%的客户端参与本轮训练
        selected_clients = np.random.choice(num_clients, int(num_clients * select_ratio), replace=False)
        print(f"\n第 {round_idx + 1} 轮: 选中客户端 {selected_clients}")

        # 并行执行客户端本地训练
        client_states, weights = parallel_client_train(
            clients, selected_clients, global_state,
            local_epochs, batch_size, lr, round_idx, client_parallelism
        )

        # 服务器聚合
        if client_states:
            global_state = aggregate_fedavg(global_state, client_states, weights)

        # 更新全局模型并测试准确率
        global_model.load_state_dict(global_state)
        accuracy = test_model_accuracy(global_model, test_loader, device)
        accuracy_history.append(accuracy)

        print(f"第 {round_idx + 1} 轮准确率: {accuracy:.2f}%")

        # 早停机制
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            os.makedirs('outputs/test_results', exist_ok=True)
            torch.save(global_state, 'outputs/test_results/best_fedavg_model.pth')
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience and round_idx > 30:
            print(f"早停触发，连续{patience}轮没有改进")
            break

    print(f"\n=== 训练完成 ===")
    print(f"最终准确率: {accuracy_history[-1]:.2f}%")
    print(f"最佳准确率: {max(accuracy_history):.2f}%")
    print(f"准确率历史: {[f'{acc:.2f}%' for acc in accuracy_history]}")

    os.makedirs('outputs/test_results', exist_ok=True)
    with open('outputs/test_results/fedavg_accuracy.txt', 'w') as f:
        f.write(f"FedAvg + ResNet-18 {num_rounds}轮训练结果\n")
        f.write("=" * 18 + "\n")
        for i, acc in enumerate(accuracy_history):
            f.write(f"第 {i + 1} 轮: {acc:.2f}%\n")
        f.write(f"\n最终准确率: {accuracy_history[-1]:.2f}%\n")
        f.write(f"最佳准确率: {max(accuracy_history):.2f}%\n")


if __name__ == "__main__":
    main()
