#!/usr/bin/env python3
"""
批量预测可视化脚本 — 在原图上标注 ROI 区域和预测结果。

从测试集中随机选取若干样本，在原图上绘制:
  - 蓝框: ROI 裁剪区域
  - 绿色/红色标签: 预测结果 (lane_in / lane_out)
  - 右侧小图: 实际输入模型的 ROI 裁剪图

Usage:
    python scripts/visualize_predictions.py \
        --data data/processed \
        --tusimple-root data/tusimple \
        --checkpoint outputs/best.pt \
        --num 10
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.models import build_model
from src.lane_binary.transforms import get_inference_transforms
from src.lane_binary.utils import get_available_font

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
ROI_WIDTH_DEFAULT = 640
ROI_HEIGHT_DEFAULT = 360
CHECK_Y = 690


def parse_args():
    parser = argparse.ArgumentParser(description="批量预测可视化")
    parser.add_argument("--data", type=str, required=True, help="已构建数据集目录")
    parser.add_argument("--tusimple-root", type=str, required=True, help="TuSimple 原始数据根目录")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--num", type=int, default=10, help="随机选取样本数")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_metadata(data_dir: str):
    """加载 metadata.csv"""
    metadata = []
    csv_path = Path(data_dir) / "metadata.csv"
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata.append(row)
    return metadata


def get_roi_bbox(roi_width=ROI_WIDTH_DEFAULT, roi_height=ROI_HEIGHT_DEFAULT):
    """计算 ROI 在原图上的裁剪坐标"""
    center_x = IMAGE_WIDTH // 2
    roi_top = max(0, IMAGE_HEIGHT - roi_height)
    left = max(0, center_x - roi_width // 2)
    right = min(IMAGE_WIDTH, center_x + roi_width // 2)
    return left, roi_top, right, roi_top + roi_height


def draw_on_original(original_img, roi_bbox, pred_label, pred_prob, gt_label):
    """在原图上绘制 ROI 框和预测结果。

    Returns: 标注后的原图
    """
    img = original_img.copy()
    draw = ImageDraw.Draw(img)

    # 绘制 ROI 框 (蓝色)
    left, top, right, bottom = roi_bbox
    draw.rectangle([left, top, right, bottom], outline=(0, 100, 255), width=3)

    # 绘制检测线 (y=CHECK_Y)
    draw.line([(0, CHECK_Y), (IMAGE_WIDTH, CHECK_Y)], fill=(255, 255, 0), width=1)

    # 预测结果文字
    label_text = "lane_in" if pred_label == 1 else "lane_out"
    label_color = (0, 200, 0) if pred_label == 1 else (255, 60, 60)

    gt_text = "lane_in" if int(gt_label) == 1 else "lane_out"
    match = "✓" if pred_label == int(gt_label) else "✗"

    # 顶部信息条
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 140))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    font_path = get_available_font(bold=True)
    try:
        font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
        font_s = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_s = ImageFont.load_default()

    draw.text((15, 8), f"Pred: {label_text}  ({pred_prob:.2%})", fill=label_color, font=font)
    draw.text((15, 32), f"GT: {gt_text}  {match}", fill=(255, 255, 255), font=font_s)

    return img


@torch.no_grad()
def predict_roi(model, roi_path, device, input_size=224):
    """对裁剪好的 ROI 进行推理"""
    transform = get_inference_transforms((input_size, input_size))
    image = Image.open(roi_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    pred_label = 1 if prob >= 0.5 else 0
    return prob, pred_label


def create_comparison_image(original_annotated, roi_img, output_size=(1000, 420)):
    """将原图(缩小)和ROI拼成一张对比图"""
    # 原图缩放到合适大小
    orig_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    panel_h = output_size[1]
    panel_w = int(panel_h * orig_ratio)

    orig_small = original_annotated.resize((panel_w, panel_h), Image.LANCZOS)
    roi_small = roi_img.resize((224, 224), Image.LANCZOS)

    # 拼接: [原图 | ROI]
    combined = Image.new("RGB", (panel_w + 250, panel_h), (30, 30, 30))
    combined.paste(orig_small, (0, 0))
    combined.paste(roi_small, (panel_w + 13, (panel_h - 224) // 2))

    # 标签
    draw = ImageDraw.Draw(combined)
    font_path = get_available_font()
    try:
        font = ImageFont.truetype(font_path, 12) if font_path else ImageFont.load_default()
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((panel_w + 13, (panel_h - 224) // 2 - 18), "ROI (224x224)", fill=(200, 200, 200), font=font)

    return combined


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("args", {}).get("model", "mobilenet_v2")
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型: {model_name}")

    # 加载 metadata
    metadata = load_metadata(args.data)
    test_rows = [r for r in metadata if r.get("split") == "test"]
    if len(test_rows) < args.num:
        print(f"测试集只有 {len(test_rows)} 个样本，使用全部")
        selected = test_rows
    else:
        selected = random.sample(test_rows, min(args.num, len(test_rows)))

    print(f"随机选取 {len(selected)} 个测试样本\n")

    comparison_images = []
    correct = 0
    total = 0

    for i, row in enumerate(selected):
        source_img = row["source_image"]
        roi_path = row["roi_file"]
        gt_label = row["label"]
        gt_label_int = 1 if gt_label == "lane_in" else 0

        # 加载原图
        source_path = Path(source_img)
        if not source_path.exists():
            # 尝试通过 tusimple_root 解析
            rel = Path(source_img).name
            print(f"  [{i+1}] 原图不存在: {source_img}, 跳过")
            continue

        original = Image.open(source_path).convert("RGB")

        # 预测
        prob, pred_label = predict_roi(model, roi_path, device)

        total += 1
        if pred_label == gt_label_int:
            correct += 1

        # 标注原图
        roi_bbox = get_roi_bbox()
        annotated = draw_on_original(original, roi_bbox, pred_label, prob, gt_label_int)

        # 加载 ROI 小图
        roi_img = Image.open(roi_path).convert("RGB")

        # 拼图
        comparison = create_comparison_image(annotated, roi_img)
        comparison_images.append(comparison)

        status = "✓" if pred_label == gt_label_int else "✗"
        print(f"  [{i+1}/{len(selected)}] {status} "
              f"GT={gt_label}, Pred={'lane_in' if pred_label==1 else 'lane_out'}, "
              f"Conf={prob:.3f}")

    # 拼接所有对比图成一张大图
    if comparison_images:
        n = len(comparison_images)
        cols = min(2, n)
        rows = (n + cols - 1) // cols

        img_w, img_h = comparison_images[0].size
        gap = 10
        canvas_w = cols * img_w + (cols + 1) * gap
        canvas_h = rows * img_h + (rows + 1) * gap

        canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
        for idx, img in enumerate(comparison_images):
            r = idx // cols
            c = idx % cols
            x = gap + c * (img_w + gap)
            y = gap + r * (img_h + gap)
            canvas.paste(img, (x, y))

        save_path = output_dir / "prediction_visualization.png"
        canvas.save(str(save_path))
        print(f"\n可视化结果已保存: {save_path}")
        print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%" if total > 0 else "")

    # 同时保存每个样本的单独大图
    for i, img in enumerate(comparison_images):
        single_path = output_dir / f"pred_sample_{i+1:02d}.png"
        img.save(str(single_path))


if __name__ == "__main__":
    main()
