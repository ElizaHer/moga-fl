import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST  # 替换为MNIST数据�?
from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from torchvision.models import resnet18


# ====================== 数据增强======================
# MNIST简单，Cutout增益有限，仅保留必要增强
class MNISTTransforms:
    @staticmethod
    def get_train_transform():
        return Compose([
            transforms.Resize((32, 32)),  # 28×28�?2×32适配ResNet
            transforms.RandomRotation(degrees=10),  # 轻微旋转增强（MNIST专属�?
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),  # MNIST标准归一�?
            transforms.Lambda(lambda x: x.expand(3, -1, -1))  # 单通道→三通道
        ])

    @staticmethod
    def get_test_transform():
        return Compose([
            transforms.Resize((32, 32)),
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.expand(3, -1, -1))
        ])


# ====================== 1. 数据加载======================
def load_mnist_data():
    train_transform = MNISTTransforms.get_train_transform()
    test_transform = MNISTTransforms.get_test_transform()

    train_dataset = MNIST(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = MNIST(root='../data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


def create_whole_dataset(train_dataset):
    print(f"数据集大�? {len(train_dataset)}")
    return range(0, len(train_dataset))


# ====================== 2. 适配ResNet-18======================
def build_resnet18_mnist(num_classes=10):
    model = resnet18(weights=None)
    # 适配小尺寸：7×7卷积�?×3，移除maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 调整全连接层（MNIST�?0类）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ====================== 3. 客户端训练（MNIST调优参数�?======================
class SimpleClient:
    def __init__(self, model, train_dataset, indices, device, test_loader=None):
        self.model = model
        self.train_dataset = train_dataset
        self.indices = indices
        self.device = device
        self.test_loader = test_loader
        # MNIST优化器参数微调：学习率不变，权重衰减略降
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-4  # MNIST权重衰减调低
        )
        self.scheduler = None

    def local_train(self, global_state, local_epochs=1, batch_size=32, lr=0.01):
        self.model.load_state_dict(global_state)
        self.model.train()

        subset = Subset(self.train_dataset, self.indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # MNIST标签平滑略降�?.05），避免过度正则�?
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # 初始化调度器（MNIST训练轮数减少�?0�?
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=60, eta_min=1e-6
            )

        # 训练循环
        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # 梯度裁剪（MNIST梯度更稳定，max_norm=1.0保持�?
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

        self.scheduler.step()

        train_accuracy = test_model_accuracy(self.model, loader, self.device)
        print(f"Train Accuracy: {train_accuracy:.2f}%")

        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}


# ====================== 4. 准确率测试（通用�?======================
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


# ====================== 5. 主函数（MNIST专属参数�?======================
def main():
    # MNIST专属参数
    num_clients = 1
    num_rounds = 60  # MNIST收敛更快，训�?0轮足�?
    local_epochs = 1
    batch_size = 128
    lr = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("加载MNIST数据�?..")
    train_dataset, test_dataset = load_mnist_data()

    partitions = create_whole_dataset(train_dataset)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("初始化ResNet-18模型...")
    model = build_resnet18_mnist(num_classes=10).to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    client = SimpleClient(model, train_dataset, partitions, device, test_loader)

    print(f"开始{num_rounds}轮训�?..")
    accuracy_history = []
    best_accuracy = 0.0
    patience = 20  # MNIST早停耐心值降�?
    no_improvement_count = 0

    for round_idx in range(num_rounds):
        state = client.local_train(
            state, local_epochs, batch_size, lr
        )

        model.load_state_dict(state)
        accuracy = test_model_accuracy(model, test_loader, device)
        accuracy_history.append(accuracy)

        print(f"�?{round_idx + 1} 轮准确率: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            os.makedirs('outputs/test_results', exist_ok=True)
            torch.save(state, 'outputs/test_results/best_mnist_model.pth')
        else:
            no_improvement_count += 1

        # MNIST早停触发条件：至少训�?0轮后判断
        if no_improvement_count >= patience and round_idx > 30:
            print(f"早停触发，连续{patience}轮没有改�?")
            break

    print(f"\n=== 训练完成 ===")
    print(f"最终准确率: {accuracy_history[-1]:.2f}%")
    print(f"最佳准确率: {max(accuracy_history):.2f}%")

    os.makedirs('outputs/test_results', exist_ok=True)
    with open('outputs/test_results/mnist_resnet_accuracy_adam.txt', 'w') as f:
        f.write(f"ResNet-18 {num_rounds}轮MNIST训练结果\n")
        f.write("=" * 18 + "\n")
        for i, acc in enumerate(accuracy_history):
            f.write(f"�?{i + 1} �? {acc:.2f}%\n")
        f.write(f"\n最终准确率: {accuracy_history[-1]:.2f}%\n")
        f.write(f"最佳准确率: {max(accuracy_history):.2f}%\n")


if __name__ == "__main__":
    main()
