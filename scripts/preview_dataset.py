#!/usr/bin/env python3
"""
二分类 ROI 数据集预览脚本。

从已构建数据集中随机抽取 lane_in / lane_out 样本并生成预览网格图。

Usage:
    python scripts/preview_dataset.py --data data/processed --split train
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lane_binary.utils import ensure_dir, get_available_font


def parse_args():
    parser = argparse.ArgumentParser(description="预览二分类 ROI 数据集")
    parser.add_argument("--data", type=str, required=True, help="已构建数据集的根目录")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"], help="预览的数据集划分")
    parser.add_argument("--samples-per-class", type=int, default=16,
                        help="每类抽取样本数 (默认 16)")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def create_preview_grid(images: list, labels: list, cols: int = 8, img_size: tuple = (128, 128)):
    """创建预览网格图。

    lane_in 行在上，lane_out 行在下。
    """
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        ImageDraw = None

    rows = 2  # lane_in 行 + lane_out 行

    grid_w = cols * img_size[0]
    grid_h = rows * img_size[1] + 40  # 顶部留标题空间

    grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    if ImageDraw:
        draw = ImageDraw.Draw(grid)
        font_path = get_available_font(bold=True)
        try:
            font = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
        except (OSError, IOError):
            font = ImageFont.load_default()

        draw.text((5, 5), "lane_in (label=1)", fill=(0, 128, 0), font=font)
        draw.text((5, img_size[1] + 25), "lane_out (label=0)", fill=(200, 0, 0), font=font)

    in_images = [(img, l) for img, l in zip(images, labels) if l == 1]
    out_images = [(img, l) for img, l in zip(images, labels) if l == 0]

    for i, (img_path, _) in enumerate(in_images[:cols]):
        img = Image.open(img_path).resize(img_size, Image.BILINEAR)
        grid.paste(img, (i * img_size[0], 20))

    for i, (img_path, _) in enumerate(out_images[:cols]):
        img = Image.open(img_path).resize(img_size, Image.BILINEAR)
        grid.paste(img, (i * img_size[0], img_size[1] + 40))

    return grid


def main():
    args = parse_args()
    random.seed(args.seed)

    data_root = Path(args.data)
    split_dir = data_root / args.split

    if not split_dir.exists():
        print(f"Error: 数据集划分目录不存在: {split_dir}")
        sys.exit(1)

    output_dir = ensure_dir(args.output)

    # 收集样本路径
    samples = []
    for label_dir, label_val in [("lane_in", 1), ("lane_out", 0)]:
        ld = split_dir / label_dir
        if ld.exists():
            for img_path in ld.glob("*.jpg"):
                samples.append((str(img_path), label_val))

    if len(samples) == 0:
        print("Error: 未找到任何样本")
        sys.exit(1)

    # 按类别抽样
    in_samples = [s for s in samples if s[1] == 1]
    out_samples = [s for s in samples if s[1] == 0]

    selected = (
        random.sample(in_samples, min(args.samples_per_class, len(in_samples))) +
        random.sample(out_samples, min(args.samples_per_class, len(out_samples)))
    )
    random.shuffle(selected)

    images = [s[0] for s in selected]
    labels = [s[1] for s in selected]

    # 生成网格图
    print(f"生成预览图: {len(selected)} 个样本 "
          f"(lane_in={min(args.samples_per_class, len(in_samples))}, "
          f"lane_out={min(args.samples_per_class, len(out_samples))})")

    cols = min(args.samples_per_class, 8)
    grid = create_preview_grid(images, labels, cols=cols)

    save_path = output_dir / "dataset_preview.png"
    grid.save(str(save_path))
    print(f"预览图已保存至: {save_path}")


if __name__ == "__main__":
    main()
