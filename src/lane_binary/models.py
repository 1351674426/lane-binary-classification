"""
车道线二分类模型定义。

提供三种可选主干网络:
- SmallCNN: 轻量级自定义 CNN，适合快速实验
- MobileNetV2: 轻量化模型，适合车载/边缘部署场景
- ResNet18: 经典残差网络，精度基线参考
"""

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """轻量级自定义 CNN，约 0.3M 参数。

    Architecture:
        Conv(3->16, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Conv(16->32, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Conv(32->64, 3x3) -> BN -> ReLU -> MaxPool(2x2)
        Conv(64->128, 3x3) -> BN -> ReLU -> AdaptiveAvgPool
        FC(128->1)
    """

    def __init__(self, input_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV2Binary(nn.Module):
    """基于 torchvision MobileNetV2 的二分类模型。

    使用 ImageNet 预训练权重，替换最后分类层为单输出 Sigmoid。
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet18Binary(nn.Module):
    """基于 torchvision ResNet18 的二分类模型。

    使用 ImageNet 预训练权重，替换最后 FC 层为单输出 Sigmoid。
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(model_name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """模型工厂函数。

    Args:
        model_name: 'small_cnn' | 'mobilenet_v2' | 'resnet18'
        pretrained: 是否加载预训练权重 (仅 MobileNetV2 / ResNet18)
        **kwargs: 传递给具体模型的额外参数

    Returns:
        nn.Module 实例
    """
    model_map = {
        'small_cnn': SmallCNN,
        'mobilenet_v2': MobileNetV2Binary,
        'resnet18': ResNet18Binary,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(model_map.keys())}")

    cls = model_map[model_name]
    if model_name == 'small_cnn':
        return cls(**kwargs)
    return cls(pretrained=pretrained, **kwargs)
