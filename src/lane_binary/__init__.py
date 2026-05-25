"""车道线二分类感知模块。

提供模型定义、数据集加载、数据增强和工具函数。
"""

from .dataset import LaneBinaryDataset, get_class_distribution
from .models import SmallCNN, MobileNetV2Binary, ResNet18Binary, build_model
from .transforms import get_train_transforms, get_val_transforms, get_inference_transforms
from .utils import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    find_ego_lane_boundaries,
    is_point_in_ego_lane,
    generate_roi_label,
    ensure_dir,
    get_available_font,
    load_json_lines,
)

__all__ = [
    "LaneBinaryDataset",
    "get_class_distribution",
    "SmallCNN",
    "MobileNetV2Binary",
    "ResNet18Binary",
    "build_model",
    "get_train_transforms",
    "get_val_transforms",
    "get_inference_transforms",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "find_ego_lane_boundaries",
    "is_point_in_ego_lane",
    "generate_roi_label",
    "ensure_dir",
    "get_available_font",
    "load_json_lines",
]