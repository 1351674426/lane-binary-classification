# 车道线二分类感知模块 — 提交包

## 环境要求

- **Python**: 3.10+
- **CUDA**: 12.4+ (GPU 训练需要，CPU 推理可选)
- **操作系统**: Linux (Ubuntu 22.04 测试通过)

## 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖版本：

| 依赖 | 版本要求 | 用途 |
|------|---------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 预训练模型 + 图像变换 |
| numpy | >=1.24.0 | 数值计算 |
| Pillow | >=9.0.0 | 图像加载 |
| scikit-learn | >=1.3.0 | 评估指标计算 |
| tqdm | >=4.65.0 | 进度条 |
| tensorboard | >=2.13.0 | 训练日志可视化 |
| matplotlib | >=3.7.0 | 混淆矩阵/可视化图表 |
| onnx | >=1.14.0 | ONNX 模型导出 (可选) |
| onnxruntime | >=1.15.0 | ONNX Runtime 推理 (可选) |

## 目录结构

```
submission/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── src/lane_binary/             # 核心模块
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset (读取 ROI 图像)
│   ├── models.py                # 模型定义 (SmallCNN/MobileNetV2/ResNet18)
│   ├── transforms.py            # 数据增强与预处理
│   └── utils.py                 # 工具函数 (车道边界提取、ROI裁剪等)
├── scripts/                     # 可执行脚本
│   ├── train.py                 # 训练脚本 (v2: Focal Loss + WeightedSampler + AMP)
│   ├── evaluate.py              # 测试集评估 (Accuracy/Precision/Recall/F1/混淆矩阵)
│   ├── export_onnx.py           # PyTorch → ONNX 导出
│   ├── infer_onnx.py            # ONNX Runtime 批量推理
│   ├── demo_predict.py          # 单图/批量 Demo 推理 (PyTorch)
│   ├── visualize_predictions.py # 预测结果可视化
│   ├── build_dataset.py         # 数据集构建 (需要原始 TuSimple 数据)
│   └── preview_dataset.py       # 数据集预览
├── data/processed/              # 已构建数据集 (6408 个 224×224 ROI 样本)
│   ├── build_config.json        # 构建参数
│   ├── metadata.csv             # 元数据 (split/label/source/roi_file)
│   ├── train/lane_in/   (3812 jpg)
│   ├── train/lane_out/  (1316 jpg)
│   ├── val/lane_in/     (473 jpg)
│   ├── val/lane_out/    (167 jpg)
│   ├── test/lane_in/    (477 jpg)
│   └── test/lane_out/   (163 jpg)
├── outputs/                     # 训练产物
│   ├── best.pt                  # ResNet18 最佳模型 (val_acc=91.09%)
│   ├── lane_binary.onnx         # ONNX 导出模型
│   ├── metrics_test.json        # 测试集评估结果
│   └── runs/                    # TensorBoard 训练日志 (加分项)
├── docs/
│   ├── 实验报告.md               # 完整实验报告 (v1→v2 全流程)
│   ├── 设计文档.pdf              # PPT设计文档 (6页, 含架构/实验结果/改进方向)
│   └── 简短实验报告.pdf          # 1-2页浓缩实验报告
```

## 快速开始

### 1. 直接推理 (使用预训练模型)

```bash
# PyTorch 单图推理
python scripts/demo_predict.py \
    --image data/processed/test/lane_in/test_set_clips_0530_1492638760826674913_0_20.jpg \
    --checkpoint outputs/best.pt \
    --device auto

# ONNX Runtime 推理
python scripts/infer_onnx.py \
    --onnx outputs/lane_binary.onnx \
    --dir data/processed/test/lane_out/ \
    --count 5
```

### 2. 测试集评估

```bash
# 评估 ResNet18 模型在测试集上的表现
python scripts/evaluate.py \
    --data data/processed \
    --checkpoint outputs/best.pt \
    --split test \
    --batch-size 64 \
    --device auto
```

预期输出 (基于当前模型):

```
Accuracy:  0.9156
Precision: 0.9453
Recall:    0.9413
F1 Score:  0.9433
Confusion Matrix:
    TN=137  FP=26
    FN= 28  TP=449
```

### 3. 重新训练 (从头开始)

```bash
# GPU 训练 (推荐)
python scripts/train.py \
    --data data/processed \
    --model resnet18 \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-3 \
    --device auto

# CPU 训练 (较慢, 建议减小 batch size)
python scripts/train.py \
    --data data/processed \
    --model resnet18 \
    --epochs 10 \
    --batch-size 32 \
    --device cpu

# 轻量模型训练
python scripts/train.py \
    --data data/processed \
    --model mobilenet_v2 \
    --epochs 30 \
    --batch-size 128 \
    --device auto
```

### 4. ONNX 导出与推理

```bash
# 导出为 ONNX 格式
python scripts/export_onnx.py \
    --checkpoint outputs/best.pt \
    --output outputs/lane_binary.onnx
```

### 5. 预测结果可视化

```bash
python scripts/visualize_predictions.py \
    --data data/processed \
    --checkpoint outputs/best.pt \
    --split test \
    --num 10
```

## 实验结果摘要

| 指标 | 数值 |
|------|------|
| 模型 | ResNet18 (10.6M 参数, ImageNet 预训练) |
| 数据集 | TuSimple 全量 6408 帧 (train=5128, val=640, test=640) |
| 训练策略 | Focal Loss (γ=2.0) + WeightedRandomSampler + AMP |
| 测试 Accuracy | **91.56%** |
| 测试 Precision | **94.53%** |
| 测试 Recall | **94.13%** |
| 测试 F1 | **94.33%** |
| lane_out 召回率 | **84.0%** (安全关键指标) |

详细实验过程与分析见 `docs/实验报告.md`。

## 数据集说明

`data/processed/` 中的 ROI 图像由 `scripts/build_dataset.py` 从 TuSimple 原始数据集 (1280×720) 构建生成。

每条 ROI 样本的标签判定规则：
- 图像底部 (y=690) 处，找到图像中心 (x=640) 左右两侧最近的车道线
- 左右边界均存在且图像中心位于两者之间 → `lane_in` 
- 否则 → `lane_out`

如需从原始 TuSimple 数据重建，请将原始数据集放置于 `data/tusimple/` 并运行：

```bash
python scripts/build_dataset.py \
    --tusimple-root data/tusimple \
    --output data/processed \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
```
