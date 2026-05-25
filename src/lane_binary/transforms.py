"""
数据增强与预处理变换。

提供训练/验证/测试三阶段的标准变换流程。
"""

from typing import Tuple

import torchvision.transforms as T


def get_train_transforms(
    input_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """训练阶段变换: 随机增强 + 归一化。

    增强策略:
        - 随机水平翻转 (p=0.3)
        - 随机亮度/对比度/饱和度调整
        - 轻微随机旋转 (±5°)
        - Resize + ToTensor + Normalize
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
        T.RandomRotation(degrees=5),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(
    input_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """验证/测试阶段变换: 仅 Resize + 归一化，无增强。"""
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(
    input_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """单图推理变换: Resize + ToTensor + Normalize。"""
    return T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
