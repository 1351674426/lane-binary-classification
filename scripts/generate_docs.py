#!/usr/bin/env python3
"""
生成提交材料: PPT设计文档 (PDF) + 简短实验报告 (PDF)。

Usage:
    python scripts/generate_docs.py
"""

import json
import os
import sys
from pathlib import Path

from fpdf import FPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.lane_binary.utils import get_available_font


def _find_cjk_font_for_fpdf(pdf: FPDF):
    """跨平台查找中文字体并注册到 fpdf 实例。"""
    font_path = get_available_font()
    if font_path:
        pdf.add_font("CJK", "", font_path)
        pdf.add_font("CJK", "B", font_path)
        return True
    return False


class DesignDocPDF(FPDF):
    """PPT设计文档 (PDF格式)"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(True, margin=20)
        self.cjk_font_path = _find_cjk_font_for_fpdf(self)

    def header(self):
        if self.page_no() > 1:
            self.set_font("CJK", "", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "车道线二分类感知模块 — 设计文档", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("CJK", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    def title_page(self):
        self.add_page()
        self.ln(40)
        self.set_font("CJK", "B", 28)
        self.set_text_color(30, 60, 120)
        self.cell(0, 14, "车道线二分类感知模块", align="C")
        self.ln(12)
        self.set_font("CJK", "", 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, "基于 TuSimple 车道线标注的设计与验证", align="C")
        self.ln(20)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.5)
        self.line(50, self.get_y(), 160, self.get_y())
        self.ln(12)
        self.set_font("CJK", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "模型: ResNet18 (10.6M) | 数据: TuSimple 6408 帧", align="C")
        self.ln(7)
        self.cell(0, 8, "测试准确率: 91.56% | F1: 94.33% | lane_out召回率: 84.0%", align="C")
        self.ln(7)
        self.cell(0, 8, "框架: PyTorch 2.x + ONNX Runtime | 2026-05", align="C")

    def section_title(self, title, level=1):
        self.ln(6)
        if level == 1:
            self.set_font("CJK", "B", 18)
            self.set_text_color(30, 60, 120)
            self.cell(0, 10, title)
            self.ln(10)
            self.set_draw_color(30, 60, 120)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(6)
        elif level == 2:
            self.set_font("CJK", "B", 13)
            self.set_text_color(50, 80, 140)
            self.cell(0, 8, title)
            self.ln(10)

    def body_text(self, text):
        self.set_font("CJK", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)

    def bullet(self, text, indent=10):
        self.set_font("CJK", "", 10)
        self.set_text_color(40, 40, 40)
        start_x = self.l_margin + indent
        self.set_x(start_x)
        self.cell(5, 6, "•")
        avail_w = self.w - self.r_margin - start_x - 5
        self.multi_cell(avail_w, 6, text)

    def table_2col(self, headers, rows, col_widths=(70, 100)):
        # Header
        self.set_font("CJK", "B", 10)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        for row in rows:
            self.set_font("CJK", "", 9.5)
            self.set_text_color(40, 40, 40)
            if rows.index(row) % 2 == 0:
                self.set_fill_color(235, 240, 248)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, str(val), border=1, fill=True, align="C")
            self.ln()

    def table_3col(self, headers, rows, col_widths=(50, 50, 70)):
        self.set_font("CJK", "B", 10)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        for row in rows:
            self.set_font("CJK", "", 9.5)
            self.set_text_color(40, 40, 40)
            if rows.index(row) % 2 == 0:
                self.set_fill_color(235, 240, 248)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, str(val), border=1, fill=True, align="C")
            self.ln()


class ShortReportPDF(FPDF):
    """简短实验报告 (PDF格式, 1-2页)"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(True, margin=18)
        self.cjk_found = _find_cjk_font_for_fpdf(self)

    def header(self):
        if self.page_no() > 1:
            self.set_font("CJK", "", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "车道线二分类感知 — 简短实验报告", align="R")
            self.ln(6)

    def section(self, title):
        self.ln(4)
        self.set_font("CJK", "B", 12)
        self.set_text_color(30, 60, 120)
        self.cell(0, 7, title)
        self.ln(8)

    def body(self, text):
        self.set_font("CJK", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)

    def body_text(self, text):
        self.set_x(self.l_margin)
        self.body(text)

    def metrics_table(self, headers, rows, col_widths):
        self.set_font("CJK", "B", 9)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        total_w = sum(col_widths)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        for row in rows:
            self.set_font("CJK", "", 9)
            self.set_text_color(40, 40, 40)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, str(val), border=1, align="C")
            self.ln()


def generate_design_doc(output_dir: str):
    """生成PPT设计文档 PDF。"""
    pdf = DesignDocPDF()
    if not pdf.cjk_font_path:
        print("WARNING: No CJK font found, PDF may not render Chinese correctly.")
        return False

    # ---- 封面 ----
    pdf.title_page()

    # ---- 第2页: 项目概述 ----
    pdf.add_page()
    pdf.section_title("1. 项目概述")
    pdf.body_text(
        "本模块是自动驾驶感知系统的一部分，负责判断当前车辆是否位于自车道（ego-lane）内。"
        "输入前视摄像头 ROI 图像，输出 lane_in / lane_out 二分类结果。"
    )
    pdf.section_title("核心判定规则", level=2)
    pdf.body_text(
        "在图像底部（y=690）处，以图像中心（x=640）为界，将标注车道线分为左右两侧候选。"
        "分别取最靠近中心的一条作为自车道左/右边界。若左右边界均存在且图像中心位于两者之间 → lane_in；否则 → lane_out。"
    )
    pdf.section_title("输入输出", level=2)
    pdf.table_2col(
        ["项目", "说明"],
        [
            ["输入图像", "1280×720 前视图 → 640×360 ROI → 224×224"],
            ["输出", "1 个 logit → sigmoid → [0, 1]"],
            ["分类阈值", "0.5"],
            ["推理延迟", "~2ms (GPU, batch=1)"],
        ],
    )

    # ---- 第3页: 数据流水线 ----
    pdf.add_page()
    pdf.section_title("2. 数据流水线")
    pdf.body_text(
        "方案经过三次迭代，最终采用 '单ROI自然标注'（方案A）：每张图仅裁剪一个 640×360 的车辆前方区域，"
        "标签完全由 TuSimple 人工标注自动推导，不引入人工虚构负样本。"
    )
    pdf.section_title("数据集分布", level=2)
    pdf.table_3col(
        ["划分", "样本数", "lane_in占比"],
        [
            ["Train", "5128", "74.3%"],
            ["Val", "640", "73.9%"],
            ["Test", "640", "74.5%"],
            ["总计", "6408", "74.3%"],
        ],
    )
    pdf.section_title("方案对比", level=2)
    pdf.table_3col(
        ["方案", "样本/图", "lane_out来源"],
        [
            ["B: ROI偏移 (×)", "1", "人工构造 (不符合实际)"],
            ["C: Grid Patch (×)", "50-80", "非车道区域patch"],
            ["A: 单ROI自然标注 (✓)", "1", "真实标注缺失场景"],
        ],
    )

    # ---- 第4页: 模型架构 & 训练策略 ----
    pdf.add_page()
    pdf.section_title("3. 模型架构")
    pdf.body_text("选用 ResNet18 (10.6M参数, ImageNet预训练)，替换最后 FC 层为 Dropout(0.2) + Linear(512, 1)。")
    pdf.table_2col(
        ["组件", "配置"],
        [
            ["Backbone", "ResNet18 (torchvision)"],
            ["预训练", "ImageNet (ResNet18_Weights.DEFAULT)"],
            ["参数量", "10.6M"],
            ["输出", "单标量 logit"],
        ],
    )

    pdf.section_title("4. 训练策略 (v2 优化)")
    pdf.table_2col(
        ["策略", "配置"],
        [
            ["损失函数", "Focal Loss (α=0.25, γ=2.0)"],
            ["采样", "WeightedRandomSampler (~50:50/batch)"],
            ["优化器", "AdamW (lr=1e-3, wd=1e-4)"],
            ["调度器", "CosineAnnealingWarmRestarts (T_0=10)"],
            ["精度", "AMP 混合精度 (CUDA)"],
            ["梯度裁剪", "max_norm=1.0"],
            ["Batch Size", "128"],
            ["早停", "patience=10 (基于 val_loss)"],
        ],
    )
    pdf.section_title("Focal Loss 原理", level=2)
    pdf.body_text(
        "FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)\n"
        "当样本易分时，(1-p_t)^γ → 0, 该样本梯度贡献被削弱；"
        "当样本难分时，(1-p_t)^γ → 1, 保持原始梯度。"
        "这使模型自动聚焦于 lane_out 等难分样本，是 lane_out 召回率从 47.6% 提升至 84.0% 的核心因素。"
    )

    # ---- 第5页: 实验结果 ----
    pdf.add_page()
    pdf.section_title("5. 实验结果")

    pdf.section_title("v1 → v2 测试集对比", level=2)
    pdf.table_3col(
        ["指标", "v1 (MobileNetV2)", "v2 (ResNet18)"],
        [
            ["Accuracy", "80.00%", "91.56%"],
            ["Precision", "81.97%", "94.53%"],
            ["Recall", "92.59%", "94.13%"],
            ["F1 Score", "86.96%", "94.33%"],
            ["lane_out召回率", "47.6%", "84.0%"],
            ["数据量", "500", "6408"],
        ],
    )

    pdf.section_title("混淆矩阵 (v2)", level=2)
    pdf.table_3col(
        ["", "Pred lane_out", "Pred lane_in"],
        [
            ["True lane_out", "TN=137", "FP=26"],
            ["True lane_in", "FN=28", "TP=449"],
        ],
    )

    pdf.section_title("V1→V2 关键改进归因", level=2)
    pdf.table_3col(
        ["改进", "主要影响", "贡献"],
        [
            ["Focal Loss", "lane_out Recall", "+36.4pp (核心因素)"],
            ["WeightedSampler", "Precision + FP↓", "消除多数类梯度支配"],
            ["全量6408数据", "Generalization", "减少过拟合"],
            ["ResNet18", "Accuracy", "4.75×模型容量"],
            ["WarmRestarts", "Val稳定性", "LR重启刷新最佳"],
        ],
    )

    # ---- 第6页: 部署 & 总结 ----
    pdf.add_page()
    pdf.section_title("6. ONNX 部署")
    pdf.body_text(
        "支持 PyTorch → ONNX 导出，使用 ONNX Runtime 进行跨平台推理。\n"
        "ONNX 模型: lane_binary.onnx (43MB)\n"
        "输入: float32 [1, 3, 224, 224]\n"
        "输出: float32 [1, 1] (logit)\n"
        "精度: PyTorch vs ONNX Runtime 差异 < 1e-7 (浮点误差级)"
    )

    pdf.section_title("7. 加分项完成情况")
    pdf.table_2col(
        ["加分项", "状态"],
        [
            ["TensorBoard 记录训练过程", "✓ train.py 内置 SummaryWriter"],
            ["ONNX 导出 + Runtime推理", "✓ export_onnx.py + infer_onnx.py"],
            ["数据增强 (亮度/翻转等)", "✓ ColorJitter + RandomFlip + Rotation"],
        ],
    )

    pdf.section_title("8. 总结")
    pdf.body_text(
        "本实验验证了将车道线标注转化为二分类标签、用 ResNet18 学习 '车辆是否在自车道内' 这一感知决策的可行性。"
        "在 TuSimple 全量数据上，测试 F1 达 94.33%，lane_out 召回率达 84.0%，达到工程部署水平。"
    )

    pdf.section_title("改进方向", level=2)
    pdf.bullet("多高度判定: 在 y=300/500/690 多处判断，减少局部遮挡误判")
    pdf.bullet("时序建模: 输入连续 3-5 帧，利用帧间连续性平滑判决")
    pdf.bullet("知识蒸馏: ResNet18 → MobileNetV2，保持精度同时降低延迟")
    pdf.bullet("多数据集: 引入 CULane/BDD100K 提升复杂场景泛化能力")
    pdf.bullet("部署量化: ONNX INT8 + TensorRT, 模型压缩至 ~11MB")

    # Save
    path = os.path.join(output_dir, "设计文档.pdf")
    pdf.output(path)
    print(f"设计文档已生成: {path} ({os.path.getsize(path)/1024:.0f} KB)")
    return True


def generate_short_report(output_dir: str):
    """生成简短实验报告 PDF。"""
    pdf = ShortReportPDF()
    if not pdf.cjk_found:
        print("WARNING: No CJK font found.")
        return False

    pdf.add_page()

    # 标题
    pdf.set_font("CJK", "B", 20)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 12, "车道线二分类感知 — 简短实验报告", align="C")
    pdf.ln(14)

    # 基本信息
    pdf.set_font("CJK", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "模型: ResNet18 (10.6M) | 数据: TuSimple 6408 帧 | 框架: PyTorch 2.x + ONNX Runtime", align="C")
    pdf.ln(6)
    pdf.cell(0, 6, "训练策略: Focal Loss + WeightedSampler + AMP | 设备: GPU (CUDA)", align="C")
    pdf.ln(10)

    # ---- 1. 数据 ----
    pdf.section("1. 数据集")
    pdf.body_text("原始来源: TuSimple Lane Detection Dataset，共 6408 个标注帧 (1280×720)。")
    pdf.body_text(
        "构造方法: 每张图以图像中心 (x=640) 为基准，从底部向上裁剪 640×360 的车辆前方 ROI。"
        "标签判定: 在 y=690 处，若图像中心位于左右自车道边界之间 → lane_in (1)，否则 → lane_out (0)。"
    )
    pdf.metrics_table(
        ["划分", "样本数", "lane_in", "lane_out", "lane_in%"],
        [
            ["Train", "5128", "3812", "1316", "74.3%"],
            ["Val", "640", "473", "167", "73.9%"],
            ["Test", "640", "477", "163", "74.5%"],
        ],
        (34, 34, 34, 34, 34),
    )

    # ---- 2. 模型与训练 ----
    pdf.section("2. 模型与训练配置")
    pdf.body_text(
        "主干网络: ResNet18 (10.6M 参数, ImageNet 预训练)。\n"
        "训练策略: Focal Loss (α=0.25, γ=2.0) + WeightedRandomSampler (~50:50/batch) + "
        "AdamW (lr=1e-3, wd=1e-4) + CosineAnnealingWarmRestarts + AMP 混合精度。\n"
        "Batch Size: 128 | Epochs: 18 (早停) | 训练耗时: ~8 分钟 (GPU)。"
    )

    # ---- 3. 结果 ----
    pdf.section("3. 测试集结果")

    pdf.body_text("v1 (MobileNetV2, 500样本) vs v2 (ResNet18, 6408样本) 对比:")
    pdf.metrics_table(
        ["指标", "v1", "v2", "提升"],
        [
            ["Accuracy", "80.00%", "91.56%", "+11.56pp"],
            ["Precision", "81.97%", "94.53%", "+12.56pp"],
            ["Recall", "92.59%", "94.13%", "+1.54pp"],
            ["F1 Score", "86.96%", "94.33%", "+7.37pp"],
            ["lane_out Recall", "47.6%", "84.0%", "+36.4pp"],
        ],
        (38, 46, 46, 40),
    )

    pdf.body_text("混淆矩阵 (v2):")
    pdf.metrics_table(
        ["", "Pred lane_out", "Pred lane_in"],
        [
            ["True lane_out", "TN=137", "FP=26"],
            ["True lane_in", "FN=28", "TP=449"],
        ],
        (46, 62, 62),
    )

    # ---- 4. 加分项 ----
    pdf.section("4. 加分项完成情况")
    pdf.body_text(
        "1. TensorBoard: train.py 内置 SummaryWriter，训练曲线 (loss/accuracy/LR) "
        "保存至 outputs/runs/，可用 tensorboard --logdir outputs/runs 查看。\n"
        "2. ONNX 导出与推理: export_onnx.py 导出 → lane_binary.onnx (43MB)，"
        "infer_onnx.py 使用 ONNX Runtime 执行推理，精度误差 < 10⁻⁷。\n"
        "3. 数据增强: 训练集使用 RandomHorizontalFlip(p=0.3) + ColorJitter(brightness/contrast/saturation/hue) "
        "+ RandomRotation(±5°)，不参与增强的验证/测试集仅归一化。"
    )

    # ---- 5. 结论 ----
    pdf.section("5. 结论")
    pdf.body_text(
        "基于 ResNet18 + Focal Loss 的二分类方案在 TuSimple 全量数据集上取得了 "
        "Accuracy=91.56%、F1=94.33%、lane_out 召回率=84.0% 的性能。"
        "Focal Loss + WeightedRandomSampler 的组合将 lane_out 召回率从 47.6% 提升至 84.0%（+36.4pp），"
        "是 v2 的核心改进。模型支持 ONNX 导出与跨平台推理部署，指标达到工程应用水平。"
    )

    # Save
    path = os.path.join(output_dir, "简短实验报告.pdf")
    pdf.output(path)
    print(f"简短实验报告已生成: {path} ({os.path.getsize(path)/1024:.0f} KB)")
    return True


def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    os.makedirs(output_dir, exist_ok=True)

    ok1 = generate_design_doc(output_dir)
    ok2 = generate_short_report(output_dir)

    if ok1 and ok2:
        print("\n=== 全部文档生成完成 ===")
    else:
        print("\n=== WARNING: 部分文档生成可能需要中文字体支持 ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
