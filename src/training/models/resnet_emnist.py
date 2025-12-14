import torch
import torch.nn as nn
from typing import Type, Optional, List


class BasicBlock(nn.Module):
    """ResNet BasicBlock，用于 EMNIST 版本。

    与 CIFAR 版本相同结构，只是首层卷积输入通道不同。"""

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetEMNIST(nn.Module):
    """ResNet-18 变体，适配 EMNIST（1x28x28 输入）。

    设计要点：
    - 首层卷积输入通道为 1；
    - 保留 4 个 stage 的 ResNet 结构，但注意 28x28 输入下的下采样不能过深；
    - 仍然使用 AdaptiveAvgPool2d 做全局池化，保证输出维度稳定；
    - 支持 width_factor，用于控制每个 stage 的通道宽度。"""

    def __init__(self, num_classes: int = 47, width_factor: float = 1.0) -> None:
        super().__init__()
        assert width_factor > 0, "width_factor must be positive"
        base_channels = int(64 * width_factor)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.in_planes = c1

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, c1, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, c2, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, c3, blocks=2, stride=2)
        # 最后一层可以选择 stride=2 或 stride=1，根据输入大小与算力需求权衡。
        self.layer4 = self._make_layer(BasicBlock, c4, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4 * BasicBlock.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet18_emnist(num_classes: int = 47, width_factor: float = 1.0) -> ResNetEMNIST:
    """工厂函数：用于在 Server/Client 中根据配置构造 EMNIST ResNet-18。"""
    return ResNetEMNIST(num_classes=num_classes, width_factor=width_factor)
