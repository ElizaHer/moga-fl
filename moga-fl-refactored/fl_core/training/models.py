from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimpleCIFARCNN(nn.Module):
    """Lightweight CNN for CIFAR-10, used in quick demos.

    与原工程 `SimpleCIFARCNN` 等价。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # type: ignore[override]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleEMNISTCNN(nn.Module):
    """Lightweight CNN for EMNIST (balanced)."""

    def __init__(self, num_classes: int = 47) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # type: ignore[override]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_resnet18_cifar(num_classes: int = 10, width_factor: float = 1.0) -> nn.Module:
    """ResNet-18 variant for CIFAR-10.

    - 使用 3×3 首层卷积并移除 maxpool 以适配 32×32 输入；
    - width_factor 目前作为占位参数，便于后续扩展通道缩放。
    """

    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # width_factor 可以在未来用于缩放各 stage 通道数，这里保持结构简单
    _ = width_factor
    return model


def build_resnet18_emnist(num_classes: int = 47, width_factor: float = 1.0) -> nn.Module:
    """ResNet-18 variant for EMNIST.

    通过在前向预处理阶段将单通道图像扩展为三通道，这里直接复用 ResNet-18
    的 3 通道卷积定义，并同样去除 maxpool 以适配 28×28 输入。
    """

    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    _ = width_factor
    return model
