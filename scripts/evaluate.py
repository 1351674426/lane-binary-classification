#!/usr/bin/env python3
"""
测试集评估脚本。

计算 Accuracy, Precision, Recall, F1-Score，并生成混淆矩阵图。

Usage:
    python scripts/evaluate.py --data data/processed --checkpoint outputs/best.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.models import build_model
from src.lane_binary.dataset import LaneBinaryDataset, get_class_distribution
from src.lane_binary.transforms import get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="评估车道线二分类模型")
    parser.add_argument("--data", type=str, required=True, help="已构建数据集的根目录")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="评估的数据集划分")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 进程数")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--input-size", type=int, default=224, help="输入图像尺寸")
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, save_path: str, class_names=("lane_out", "lane_in")):
    """绘制并保存混淆矩阵图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=18, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  混淆矩阵图已保存至: {save_path}")
    except ImportError:
        print("  (matplotlib 未安装，跳过混淆矩阵图生成)")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """在给定 DataLoader 上评估模型。

    Returns:
        包含 accuracy, precision, recall, f1, confusion_matrix 的字典
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1_score": float(f1_score(all_labels, all_preds, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(all_labels)),
    }

    # 各类别详细统计
    tn, fp, fn, tp = cm.ravel()
    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    return metrics


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = LaneBinaryDataset(
        data_root=args.data,
        split=args.split,
        transform=get_val_transforms((args.input_size, args.input_size)),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    dist = get_class_distribution(dataset)
    print(f"\n{args.split} 集分布: lane_in={dist['lane_in']}, lane_out={dist['lane_out']}, "
          f"total={dist['total']}")

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("args", {}).get("model", "mobilenet_v2")
    print(f"\n加载模型: {model_name} (from {args.checkpoint})")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint val_acc: {checkpoint.get('val_accuracy', 'unknown')}")

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 评估
    print(f"\n开始评估 {args.split} 集...")
    metrics = evaluate(model, loader, device)

    print(f"\n{'='*50}")
    print(f"评估结果 ({args.split})")
    print(f"{'='*50}")
    print(f"  Total samples:  {metrics['total_samples']}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['true_negative']:5d}  FP={metrics['false_positive']:5d}")
    print(f"    FN={metrics['false_negative']:5d}  TP={metrics['true_positive']:5d}")
    print(f"{'='*50}")

    # 保存结果
    metrics_path = output_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n评估指标已保存至: {metrics_path}")

    # 混淆矩阵图
    cm = np.array(metrics["confusion_matrix"])
    cm_path = output_dir / f"confusion_matrix_{args.split}.png"
    plot_confusion_matrix(cm, str(cm_path))


if __name__ == "__main__":
    main()
