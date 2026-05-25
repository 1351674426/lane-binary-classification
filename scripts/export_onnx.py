#!/usr/bin/env python3
"""
将训练好的 PyTorch 模型导出为 ONNX 格式，并验证导出结果。

Usage:
    python scripts/export_onnx.py --checkpoint outputs/best.pt --output outputs/lane_binary.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_binary.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default="outputs/lane_binary.onnx",
                        help="ONNX 输出路径")
    parser.add_argument("--input-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--device", type=str, default="cpu", help="导出设备 (建议 cpu)")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset 版本")
    return parser.parse_args()


def verify_onnx(onnx_path: str, model: nn.Module, device: torch.device,
                input_size: int = 224):
    """使用 onnxruntime 验证导出结果。"""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("Warning: onnx/onnxruntime 未安装，跳过验证")
        return

    # 检查 ONNX 模型有效性
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX 模型格式验证通过 ✓")

    # 创建随机输入
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # PyTorch 推理
    model.eval()
    with torch.no_grad():
        torch_out = model(dummy_input)

    # ONNX Runtime 推理
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)

    # 比较
    diff = np.abs(torch_out.cpu().numpy() - ort_out[0])
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  PyTorch vs ONNX Runtime 差异: max={max_diff:.6e}, mean={mean_diff:.6e}")

    if max_diff < 1e-4:
        print("  精度验证通过 ✓ (max_diff < 1e-4)")
    else:
        print(f"  Warning: 存在较大差异 (max_diff={max_diff:.6e})")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint.get("args", {}).get("model", "mobilenet_v2")
    print(f"加载模型: {model_name}")

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 创建 dummy 输入
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size).to(device)

    # 导出 ONNX
    print(f"\n导出 ONNX (opset={args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"ONNX 模型已导出至: {args.output}")

    # 验证
    print("\n验证 ONNX 模型...")
    verify_onnx(args.output, model, device, args.input_size)

    print("\n导出完成")


if __name__ == "__main__":
    main()
