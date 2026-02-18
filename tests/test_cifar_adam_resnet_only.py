import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from torchvision.models import resnet18


# ====================== Cutout数据增强 ======================
# 随机遮挡图像局部区域，迫使模型学习更鲁棒的特征
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


# ====================== 1. 数据加载（增加Cutout） ======================
def load_cifar10_data():
    """加载CIFAR-10数据集（对齐增强策略）"""
    train_transform = Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(length=8)  # 新增Cutout增强
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='../data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


def create_whole_dataset(train_dataset):
    print(f"数据集大小: {len(train_dataset)}")
    return range(0, len(train_dataset))


# ====================== 2. 适配ResNet-18（替换原有build函数） ======================
def build_resnet18_cifar(num_classes=10):
    """对齐模型结构：移除dropout，适配小尺寸数据集"""
    model = resnet18(pretrained=False)
    # 适配小尺寸：7×7卷积→3×3，移除maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 调整全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # 移除dropout（原代码的dropout=0.5会导致精度下降）
    return model


# ====================== 3. 客户端训练（修复调度器+增加标签平滑+梯度裁剪） ======================
class SimpleClient:
    def __init__(self, model, train_dataset, indices, device, test_loader=None):
        self.model = model
        self.train_dataset = train_dataset
        self.indices = indices
        self.device = device
        self.test_loader = test_loader
        # 初始化优化器和调度器（避免每轮重置）
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        self.scheduler = None

    def local_train(self, global_state, local_epochs=1, batch_size=32, lr=0.01):
        self.model.load_state_dict(global_state)
        self.model.train()

        subset = Subset(self.train_dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # 带标签平滑的损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 初始化调度器（仅第一次调用时）
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100, eta_min=1e-6
            )

        # 训练循环
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

        # 调度器在每轮全局训练后更新（而非local epoch）
        self.scheduler.step()

        train_accuracy = test_model_accuracy(self.model, loader, self.device)
        print(f"Train Accuracy: {train_accuracy:.2f}%")

        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


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
    # 对齐参数
    num_clients = 1
    num_rounds = 100  # 对应100个epoch
    local_epochs = 1  # 保持1（联邦学习单轮1个local epoch）
    batch_size = 128
    lr = 0.001  # 对齐学习率

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("加载CIFAR-10数据集...")
    train_dataset, test_dataset = load_cifar10_data()

    partitions = create_whole_dataset(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("初始化ResNet-18模型...")
    model = build_resnet18_cifar(num_classes=10).to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    client = SimpleClient(model, train_dataset, partitions, device, test_loader)

    print(f"开始{num_rounds}轮训练...")
    accuracy_history = []
    best_accuracy = 0.0
    patience = 50  # 早停耐心值
    no_improvement_count = 0

    for round_idx in range(num_rounds):
        state = client.local_train(
            state, local_epochs, batch_size, lr
        )

        model.load_state_dict(state)
        accuracy = test_model_accuracy(model, test_loader, device)
        accuracy_history.append(accuracy)

        print(f"第 {round_idx + 1} 轮准确率: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            os.makedirs('outputs/test_results', exist_ok=True)
            torch.save(state, 'outputs/test_results/best_model.pth')
        else:
            no_improvement_count += 1

        # 调整早停触发条件：至少训练60轮后再判断
        if no_improvement_count >= patience and round_idx > 60:
            print(f"早停触发，连续{patience}轮没有改进")
            break

    print(f"\n=== 训练完成 ===")
    print(f"最终准确率: {accuracy_history[-1]:.2f}%")
    print(f"最佳准确率: {max(accuracy_history):.2f}%")

    os.makedirs('outputs/test_results', exist_ok=True)
    with open('outputs/test_results/only_resnet_accuracy_adam.txt', 'w') as f:
        f.write(f"ResNet-18 {num_rounds}轮训练结果\n")
        f.write("=" * 18 + "\n")
        for i, acc in enumerate(accuracy_history):
            f.write(f"第 {i + 1} 轮: {acc:.2f}%\n")
        f.write(f"\n最终准确率: {accuracy_history[-1]:.2f}%\n")
        f.write(f"最佳准确率: {max(accuracy_history):.2f}%\n")


if __name__ == "__main__":
    main()