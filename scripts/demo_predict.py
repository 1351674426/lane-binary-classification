#!/usr/bin/env python3
"""
单张 ROI 图像预测可视化脚本。

对指定图像运行模型推理，输出预测结果和置信度，并生成标注图。

Usage:
    python scripts/demo_predict.py --image data/processed/test/lane_in/xxx.jpg --checkpoint outputs/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.models import build_model
from src.lane_binary.transforms import get_inference_transforms
from src.lane_binary.utils import get_available_font


def parse_args():
    parser = argparse.ArgumentParser(description="单张图像预测演示")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--input-size", type=int, default=224, help="输入图像尺寸")
    return parser.parse_args()


def draw_prediction(image: Image.Image, prob: float, pred_label: int) -> Image.Image:
    """在图像上绘制预测结果。"""
    draw = ImageDraw.Draw(image)

    label_text = "lane_in" if pred_label == 1 else "lane_out"
    color = (0, 200, 0) if pred_label == 1 else (200, 0, 0)

    # 顶部半透明条
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 120))
    image = image.convert("RGBA")
    image = Image.alpha_composite(image, overlay)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    font_path = get_available_font(bold=True)
    try:
        font_large = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default()
        font_small = ImageFont.truetype(font_path, 16) if font_path else ImageFont.load_default()
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    text_y = 10
    draw.text((15, text_y), f"Prediction: {label_text}", fill=color, font=font_large)
    draw.text((15, text_y + 32), f"Confidence: {prob:.4f}", fill=(255, 255, 255),
              font=font_small)
    draw.text((15, text_y + 54), f"Threshold: 0.5", fill=(200, 200, 200),
              font=font_small)

    return image


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    input_size: int = 224,
) -> tuple:
    """对单张图像执行推理。

    Returns:
        (probability, predicted_label)
    """
    transform = get_inference_transforms((input_size, input_size))

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    pred_label = 1 if prob >= 0.5 else 0

    return prob, pred_label


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: 图像不存在: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("args", {}).get("model", "mobilenet_v2")
    print(f"加载模型: {model_name}")

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 预测
    prob, pred_label = predict(model, str(image_path), device, args.input_size)

    label_text = "lane_in (车道内)" if pred_label == 1 else "lane_out (车道外)"
    print(f"\n{'='*40}")
    print(f"预测结果")
    print(f"{'='*40}")
    print(f"  图像:       {image_path.name}")
    print(f"  置信度:     {prob:.4f}")
    print(f"  预测标签:   {pred_label} -> {label_text}")
    print(f"{'='*40}")

    # 生成标注图
    image = Image.open(image_path).convert("RGB")
    annotated = draw_prediction(image, prob, pred_label)

    save_path = output_dir / "demo_prediction.png"
    annotated.save(str(save_path))
    print(f"\n预测结果图已保存至: {save_path}")


if __name__ == "__main__":
    main()
