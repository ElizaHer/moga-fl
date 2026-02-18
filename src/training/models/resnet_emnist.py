import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_mnist(num_classes=10):
    """适配MNIST的ResNet-18：单通道转三通道后兼容"""
    model = resnet18(pretrained=False)
    # 适配小尺寸：7×7卷积→3×3，移除maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 调整全连接层（MNIST为10类）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model