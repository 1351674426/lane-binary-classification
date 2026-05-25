#!/usr/bin/env python3
"""
将 TuSimple 车道线标注转换为二分类 ROI 数据集。

每张图裁剪一个车辆前方 ROI，按规则判定标签:
  - 图像底部存在左右自车道边界 + 图像中心在两条线之间 → lane_in
  - 否则 → lane_out

Usage:
    python scripts/build_dataset.py --tusimple-root data/tusimple --output data/processed
    python scripts/build_dataset.py --tusimple-root data/tusimple --output data/processed_debug --max-images 200
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.utils import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    load_json_lines,
    generate_roi_label,
    ensure_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="构建车道线二分类 ROI 数据集")
    parser.add_argument(
        "--tusimple-root", type=str, required=True,
        help="TuSimple 数据集根目录 (包含 train_set/, test_set/, test_label.json 等)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--roi-width", type=int, default=640,
        help="ROI 裁剪宽度 (默认 640)",
    )
    parser.add_argument(
        "--roi-height", type=int, default=360,
        help="ROI 裁剪高度 (默认 360)",
    )
    parser.add_argument(
        "--check-y", type=int, default=690,
        help="判定车道边界的参考高度 (默认 690)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="验证集比例 (默认 0.15)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.15,
        help="测试集比例 (默认 0.15)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="最大处理图片数 (用于快速调试，默认 None = 全部)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认 42)",
    )
    return parser.parse_args()


def collect_tusimple_annotations(tusimple_root: Path) -> List[Tuple[str, dict]]:
    """收集 TuSimple 数据集中所有图像-标注对。

    Returns:
        [(image_full_path, annotation_dict), ...]
    """
    samples = []

    # --- 训练集 ---
    train_dir = tusimple_root / "train_set"
    if train_dir.exists():
        # 查找所有 label_data_*.json
        label_files = sorted(train_dir.glob("label_data_*.json"))
        for label_file in label_files:
            annotations = load_json_lines(str(label_file))
            for ann in annotations:
                raw_file = ann.get("raw_file", "")
                img_path = train_dir / raw_file
                if img_path.exists():
                    samples.append((str(img_path), ann))

    # --- 测试集 ---
    test_label = tusimple_root / "test_label.json"
    test_dir = tusimple_root / "test_set"
    if test_label.exists() and test_dir.exists():
        annotations = load_json_lines(str(test_label))
        for ann in annotations:
            raw_file = ann.get("raw_file", "")
            img_path = test_dir / raw_file
            if img_path.exists():
                samples.append((str(img_path), ann))

    # 同时尝试 test_set 下的 test_tasks_*.json (无标注仅含 h_samples，跳过)
    # 只使用有完整 lanes 标注的数据

    return samples


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    tusimple_root = Path(args.tusimple_root)
    output_root = Path(args.output)

    if not tusimple_root.exists():
        print(f"Error: TuSimple root not found: {tusimple_root}")
        sys.exit(1)

    print(f"[1/4] 收集 TuSimple 标注...")
    all_samples = collect_tusimple_annotations(tusimple_root)
    print(f"  找到 {len(all_samples)} 个有效图像-标注对")

    if len(all_samples) == 0:
        print("Error: 未找到有效标注数据，请检查 --tusimple-root 路径")
        sys.exit(1)

    # 可选限制数量
    if args.max_images is not None:
        random.shuffle(all_samples)
        all_samples = all_samples[:args.max_images]
        print(f"  限制为 {len(all_samples)} 张图像 (--max-images={args.max_images})")

    # 按图像级划分 train / val / test
    n_total = len(all_samples)
    n_test = int(n_total * args.test_ratio)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val - n_test

    random.shuffle(all_samples)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"\n  划分结果: train={n_train}, val={n_val}, test={n_test}")

    # 构建数据集
    roi_config = {
        "roi_width": args.roi_width,
        "roi_height": args.roi_height,
        "check_y": args.check_y,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_images": args.max_images,
        "seed": args.seed,
    }

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    metadata_rows = []

    for split_name, split_samples in splits.items():
        print(f"\n[2/4] 生成 {split_name} 集 ROI 样本...")

        lane_in_dir = ensure_dir(output_root / split_name / "lane_in")
        lane_out_dir = ensure_dir(output_root / split_name / "lane_out")

        for img_path, ann in tqdm(split_samples, desc=f"  {split_name}"):
            lanes = ann.get("lanes", [])
            h_samples = ann.get("h_samples", [])

            if not lanes or not h_samples:
                continue

            # 每张图生成一个 ROI + 标签
            roi_img, label = generate_roi_label(
                image_path=img_path,
                lanes=lanes,
                h_samples=h_samples,
                roi_width=args.roi_width,
                roi_height=args.roi_height,
                check_y=args.check_y,
            )

            label_name = "lane_in" if label == 1 else "lane_out"
            target_dir = lane_in_dir if label == 1 else lane_out_dir

            base_name = Path(img_path).stem
            # 用相对路径避免不同 clip 下同名文件覆盖
            rel_path = Path(img_path).relative_to(tusimple_root)
            safe_name = str(rel_path).replace("/", "_").replace("\\", "_")
            save_path = target_dir / safe_name

            roi_img = roi_img.resize((224, 224), Image.BILINEAR)
            roi_img.save(str(save_path), quality=95)

            metadata_rows.append({
                "split": split_name,
                "label": label_name,
                "source_image": img_path,
                "roi_file": str(save_path),
            })

    # 保存 metadata
    print(f"\n[3/4] 保存 metadata...")
    import csv
    metadata_path = output_root / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "source_image", "roi_file"])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  metadata 已保存至: {metadata_path}")

    # 保存构建配置
    config_path = output_root / "build_config.json"
    with open(config_path, "w") as f:
        json.dump(roi_config, f, indent=2, ensure_ascii=False)
    print(f"  构建配置已保存至: {config_path}")

    # 统计
    print(f"\n[4/4] 数据集统计:")
    for split_name in ["train", "val", "test"]:
        in_count = len(list((output_root / split_name / "lane_in").glob("*.jpg")))
        out_count = len(list((output_root / split_name / "lane_out").glob("*.jpg")))
        total = in_count + out_count
        in_ratio = in_count / total * 100 if total > 0 else 0
        print(f"  {split_name:5s}: lane_in={in_count:6d}  lane_out={out_count:6d}  "
              f"total={total:6d}  in_ratio={in_ratio:.1f}%")

    print(f"\n数据集构建完成: {output_root}")


if __name__ == "__main__":
    main()
