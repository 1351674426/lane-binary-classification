#!/usr/bin/env python3
"""
车道线二分类模型训练脚本 (v2 GPU优化版)。

改进策略:
  - Focal Loss 替代 BCEWithLogitsLoss (聚焦难分样本，缓解类别不平衡)
  - AMP 混合精度训练 (加速GPU训练)
  - WeightedRandomSampler 类别平衡采样
  - CosineAnnealingWarmRestarts (周期重启，跳出局部最优)
  - ResNet18 作为默认模型 (更高精度)

Usage:
    # ResNet18 (默认, 推荐)
    python scripts/train.py --data data/processed --model resnet18 --epochs 30

    # MobileNetV2 (轻量对比)
    python scripts/train.py --data data/processed --model mobilenet_v2 --epochs 30
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.models import build_model
from src.lane_binary.dataset import LaneBinaryDataset, get_class_distribution
from src.lane_binary.transforms import get_train_transforms, get_val_transforms


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with logits input.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    相比 BCEWithLogitsLoss，Focal Loss 自动降低易分样本的权重，
    让模型聚焦于难分样本 (多数是 lane_out)。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # 正样本权重 (lane_in)
        self.gamma = gamma  # 聚焦参数, 越大越聚焦难样本

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, 1] logits, targets: [B, 1] 0/1
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)  # p_t = model's estimated probability for true class
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="训练车道线二分类模型")
    parser.add_argument("--data", type=str, required=True, help="已构建数据集的根目录")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["small_cnn", "mobilenet_v2", "resnet18"],
                        help="模型类型 (默认 resnet18)")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader 进程数")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备: 'auto', 'cuda', 'cpu'")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="禁用预训练权重")
    parser.add_argument("--input-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    parser.add_argument("--no-balanced-sampler", action="store_true",
                        help="禁用 WeightedRandomSampler")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal Loss gamma 参数")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler = None,
    use_amp: bool = True,
) -> dict:
    """训练一个 epoch，支持 AMP 混合精度。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # [B] -> [B, 1]

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total
    accuracy = correct / total

    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)

    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    """验证集评估，返回平均 loss 和 accuracy。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", accuracy, epoch)

    return {"loss": avg_loss, "accuracy": accuracy}


def main():
    args = parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == 'cuda'
    print(f"Device: {device}")
    print(f"AMP: {'ON' if use_amp else 'OFF'}")

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "runs" / f"{args.model}_{timestamp}"))

    # 数据集
    train_dataset = LaneBinaryDataset(
        data_root=args.data,
        split="train",
        transform=get_train_transforms((args.input_size, args.input_size)),
    )
    val_dataset = LaneBinaryDataset(
        data_root=args.data,
        split="val",
        transform=get_val_transforms((args.input_size, args.input_size)),
    )

    # 类别分布
    dist = get_class_distribution(train_dataset)
    print(f"\n训练集分布: lane_in={dist['lane_in']}, lane_out={dist['lane_out']}, "
          f"total={dist['total']}, in_ratio={dist['in_ratio']:.2%}")

    dist_val = get_class_distribution(val_dataset)
    print(f"验证集分布: lane_in={dist_val['lane_in']}, lane_out={dist_val['lane_out']}, "
          f"total={dist_val['total']}, in_ratio={dist_val['in_ratio']:.2%}")

    # WeightedRandomSampler: 每 batch 保持 lane_in 和 lane_out 各 50%
    if not args.no_balanced_sampler:
        labels_array = np.array(train_dataset.labels)
        class_counts = np.bincount(labels_array.astype(int))
        if len(class_counts) < 2:
            print(f"  Warning: 训练集只有单一类别，禁用 WeightedRandomSampler")
            sampler = None
            shuffle = True
        else:
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[labels_array.astype(int)]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            shuffle = False
            print(f"  采样策略: WeightedRandomSampler (类别平衡: ~50:50/batch)")
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=shuffle if sampler is None else None,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # 模型
    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: {args.model} | 可训练参数: {n_params / 1e6:.2f}M")

    # Focal Loss: 自动聚焦难分样本
    criterion = FocalLoss(alpha=0.25, gamma=args.focal_gamma)
    print(f"损失函数: FocalLoss(alpha=0.25, gamma={args.focal_gamma})")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # CosineAnnealingWarmRestarts: 每 10 epochs 重启，逐步衰减
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # AMP scaler
    scaler = GradScaler('cuda') if use_amp else None

    # 训练
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    history = {"train": [], "val": []}

    print(f"\n{'='*50}")
    print(f"开始训练: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"模型: {args.model}, Focal Loss, WeightedSampler, AMP={'ON' if use_amp else 'OFF'}")
    print(f"早停策略: val_acc 连续 {patience} 轮未提升触发")
    print(f"{'='*50}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            scaler=scaler, use_amp=use_amp,
        )
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}  LR: {current_lr:.2e}")

        # 保存最佳模型 (基于 val_acc)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "val_loss": val_metrics["loss"],
                "args": vars(args),
            }, checkpoint_path)
            print(f"  >> 保存最佳模型 (val_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发: val_acc 连续 {patience} 轮未提升 (best={best_val_acc:.4f})")
                break

        scheduler.step()
        writer.add_scalar("LR", current_lr, epoch)

    # 保存训练历史
    history_path = output_dir / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n训练历史已保存至: {history_path}")

    writer.close()
    print(f"训练完成, 最佳 val_acc = {best_val_acc:.4f}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'runs'}")


if __name__ == "__main__":
    main()
