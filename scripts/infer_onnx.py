#!/usr/bin/env python3
"""
使用 ONNX Runtime 对单张图像执行推理。

Usage:
    python scripts/infer_onnx.py --onnx outputs/lane_binary.onnx --image data/processed/test/lane_in/xxx.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.transforms import get_inference_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime 推理")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--input-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="性能基准测试次数 (0 = 不测试)")
    return parser.parse_args()


def infer_onnx(onnx_path: str, image_path: str, input_size: int = 224) -> tuple:
    """使用 ONNX Runtime 执行单张图像推理。

    Returns:
        (probability, predicted_label)
    """
    import onnxruntime as ort

    transform = get_inference_transforms((input_size, input_size))

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    prob = 1.0 / (1.0 + np.exp(-outputs[0][0][0]))  # sigmoid
    pred_label = 1 if prob >= 0.5 else 0

    return float(prob), pred_label


def benchmark_onnx(onnx_path: str, image_path: str, input_size: int, num_runs: int):
    """ONNX Runtime 推理性能基准测试。"""
    import time
    import onnxruntime as ort

    transform = get_inference_transforms((input_size, input_size))
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # 预热
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # 计时
    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {input_name: input_tensor})
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    print(f"\n性能基准测试 ({num_runs} 次推理):")
    print(f"  总耗时:   {elapsed:.3f}s")
    print(f"  平均耗时: {avg_ms:.2f}ms")
    print(f"  FPS:      {num_runs / elapsed:.1f}")


def main():
    args = parse_args()

    onnx_path = Path(args.onnx)
    image_path = Path(args.image)

    if not onnx_path.exists():
        print(f"Error: ONNX 模型不存在: {onnx_path}")
        sys.exit(1)
    if not image_path.exists():
        print(f"Error: 图像不存在: {image_path}")
        sys.exit(1)

    print(f"ONNX 模型: {onnx_path}")
    print(f"输入图像: {image_path}")

    # 推理
    prob, pred_label = infer_onnx(str(onnx_path), str(image_path), args.input_size)

    label_text = "lane_in (车道内)" if pred_label == 1 else "lane_out (车道外)"
    print(f"\n{'='*40}")
    print(f"ONNX Runtime 推理结果")
    print(f"{'='*40}")
    print(f"  置信度:     {prob:.4f}")
    print(f"  预测标签:   {pred_label} -> {label_text}")
    print(f"{'='*40}")

    # 基准测试
    if args.benchmark > 0:
        benchmark_onnx(str(onnx_path), str(image_path), args.input_size, args.benchmark)


if __name__ == "__main__":
    main()
