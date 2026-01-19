import torch
import torch.nn as nn
from typing import Type, Callable, Optional, List


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock, adapted for CIFAR-10 (no 7x7 conv/maxpool).

    对应论文中的 ResNet-18 基础残差块：两个 3x3 卷积 + shortcut。
    这里保持实现尽量简洁，方便在低算力环境下复现。"""

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


class Bottleneck(nn.Module):
    """ResNet Bottleneck block for ResNet-50 and deeper networks.

    对应论文中的 Bottleneck 块：1x1 降维 -> 3x3 卷积 -> 1x1 升维。
    expansion 为 4，输出通道数为 planes * 4。"""

    expansion: int = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # 1x1 降维卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 升维卷积
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet 变体，适配 CIFAR-10（3x32x32 输入），支持 ResNet-18 和 ResNet-50。

    关键差异：
    - 移除 ImageNet 版本中的 7x7 大卷积和 3x3 maxpool；
    - 以 3x3, stride=1 的卷积作为首层，保留 4 个 stage 的残差结构；
    - 支持 width_factor，对所有通道数按比例缩放，便于在 quick 配置下减小模型宽度；
    - 支持不同深度的 ResNet 架构。"""

    def __init__(
            self,
            num_classes: int = 10,
            width_factor: float = 1.0,
            depth: int = 18
    ) -> None:
        super().__init__()
        assert width_factor > 0, "width_factor must be positive"
        assert depth in [18, 50], "Only ResNet-18 and ResNet-50 are supported"

        self.depth = depth
        self.block = BasicBlock if depth == 18 else Bottleneck

        # 根据深度设置每个stage的block数量
        if depth == 18:
            blocks_per_stage = [2, 2, 2, 2]
        else:  # depth == 50
            blocks_per_stage = [3, 4, 6, 3]

        base_channels = int(64 * width_factor)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.in_planes = c1

        # CIFAR 版本首层：3x3, stride=1, padding=1
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)

        # 4 个残差 stage
        self.layer1 = self._make_layer(self.block, c1, blocks=blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(self.block, c2, blocks=blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(self.block, c3, blocks=blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(self.block, c4, blocks=blocks_per_stage[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4 * self.block.expansion, num_classes)

    def _make_layer(
            self,
            block: Type[nn.Module],
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


def build_resnet18_cifar(num_classes: int = 10, width_factor: float = 1.0) -> ResNetCIFAR:
    """工厂函数：用于在 Server/Client 中根据配置构造 CIFAR-10 ResNet-18。

    参数
    ------
    num_classes: 分类类别数（CIFAR-10 为 10）。
    width_factor: 通道宽度缩放因子，<1.0 时得到窄版 ResNet，适合 quick 模式。"""
    return ResNetCIFAR(num_classes=num_classes, width_factor=width_factor, depth=18)


def build_resnet50_cifar(num_classes: int = 10, width_factor: float = 1.0) -> ResNetCIFAR:
    """工厂函数：用于在 Server/Client 中根据配置构造 CIFAR-10 ResNet-50。

    参数
    ------
    num_classes: 分类类别数（CIFAR-10 为 10）。
    width_factor: 通道宽度缩放因子，<1.0 时得到窄版 ResNet，适合 quick 模式。"""
    return ResNetCIFAR(num_classes=num_classes, width_factor=width_factor, depth=50)