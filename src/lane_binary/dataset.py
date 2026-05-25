"""
PyTorch Dataset 类，用于加载 build_dataset.py 生成的二分类 ROI 数据集。
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class LaneBinaryDataset(Dataset):
    """车道线二分类 ROI 数据集。

    期望目录结构:
        data_root/
            train/
                lane_in/
                lane_out/
            val/
                lane_in/
                lane_out/
            test/
                lane_in/
                lane_out/

    Args:
        data_root: 已构建数据集的根目录
        split: 'train' | 'val' | 'test'
        transform: torchvision transforms 或 albumentations 变换
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[object] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = []
        self.labels = []

        # 收集 lane_in (label=1) 样本
        lane_in_dir = split_dir / "lane_in"
        if lane_in_dir.exists():
            for img_path in sorted(lane_in_dir.glob("*.jpg")):
                self.samples.append(str(img_path))
                self.labels.append(1.0)

        # 收集 lane_out (label=0) 样本
        lane_out_dir = split_dir / "lane_out"
        if lane_out_dir.exists():
            for img_path in sorted(lane_out_dir.glob("*.jpg")):
                self.samples.append(str(img_path))
                self.labels.append(0.0)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir} (lane_in/ or lane_out/)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_class_distribution(dataset: LaneBinaryDataset) -> dict:
    """统计数据集的类别分布。

    Returns:
        {'lane_in': count, 'lane_out': count, 'total': count}
    """
    in_count = sum(1 for l in dataset.labels if l == 1.0)
    out_count = sum(1 for l in dataset.labels if l == 0.0)
    return {
        "lane_in": in_count,
        "lane_out": out_count,
        "total": len(dataset),
        "in_ratio": in_count / len(dataset) if len(dataset) > 0 else 0.0,
    }
