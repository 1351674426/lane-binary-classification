"""
工具函数: 自车道边界提取、ROI 裁剪、标签判定。

核心规则:
  每张图裁剪一个车辆前方 ROI，判断该 ROI 中图像底部中心
  是否位于自车道内:
    - 存在左车道线(在图像中心左侧) + 右车道线(在图像中心右侧)
      + 图像中心位于两条线之间 → lane_in (1)
    - 否则 → lane_out (0)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# TuSimple 图像固定尺寸
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def load_json_lines(filepath: str) -> list:
    """加载 TuSimple 格式的 JSON Lines 标注文件。"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_ego_lane_boundaries(
    lanes: List[List[int]],
    h_samples: List[int],
    y_threshold: int = 650,
) -> Tuple[Optional[int], Optional[int]]:
    """在指定高度 y_threshold 附近找到自车道左右边界 x 坐标。

    自车道 = 图像中心 (640) 左右两侧最近的、同时存在有效标注的两条车道线。

    Args:
        lanes: 车道线列表，每个元素是一条车道线在各 h_samples 的 x 坐标
        h_samples: 高度采样点列表
        y_threshold: 判定车道边界的参考高度

    Returns:
        (left_x, right_x): 自车道左/右边界 x 坐标，若无法确定则对应值为 None
    """
    # 找到最接近 y_threshold 的 h_samples 索引
    if y_threshold not in h_samples:
        idx = min(range(len(h_samples)), key=lambda i: abs(h_samples[i] - y_threshold))
    else:
        idx = h_samples.index(y_threshold)

    left_candidates = []
    right_candidates = []

    for lane in lanes:
        x = lane[idx] if idx < len(lane) else -2
        if x == -2:
            continue
        if x < IMAGE_WIDTH / 2:
            left_candidates.append(x)
        else:
            right_candidates.append(x)

    # 取最靠近中心的左右两条
    left_x = max(left_candidates) if left_candidates else None
    right_x = min(right_candidates) if right_candidates else None

    return left_x, right_x


def is_point_in_ego_lane(
    lanes: List[List[int]],
    h_samples: List[int],
    check_x: int,
    check_y: int,
) -> bool:
    """判断 (check_x, check_y) 点是否位于自车道内。

    根据 check_y 在 h_samples 中插值找到对应高度的左右边界，判断 check_x
    是否在两者之间。

    Args:
        lanes: 车道线列表
        h_samples: 高度采样点列表
        check_x: 检测点的 x 坐标
        check_y: 检测点的 y 坐标

    Returns:
        True 如果在车道内，False 如果在车道外或无法判断
    """
    left_x, right_x = find_ego_lane_boundaries(lanes, h_samples, y_threshold=check_y)

    if left_x is None or right_x is None:
        return False

    return left_x <= check_x <= right_x


def generate_roi_label(
    image_path: str,
    lanes: List[List[int]],
    h_samples: List[int],
    roi_width: int = 640,
    roi_height: int = 360,
    check_y: int = 690,
) -> Tuple[Image.Image, int]:
    """从单张 TuSimple 图像裁剪车辆前方 ROI 并判定标签。

    规则:
      在图像底部 (check_y) 处:
      1. 找到图像中心 (640) 左侧最近的有效车道线 → 左边界
      2. 找到图像中心 (640) 右侧最近的有效车道线 → 右边界
      3. 如果左右边界都存在且图像中心在两者之间 → lane_in (1)
      4. 否则 → lane_out (0)

    裁剪区域: 图像中下方，以图像中心为水平基准，固定 640×360 大小。

    Args:
        image_path: 原图路径
        lanes: 车道线标注
        h_samples: 高度采样点
        roi_width: ROI 宽度 (默认 640)
        roi_height: ROI 高度 (默认 360)
        check_y: 判定车道边界的参考高度 (默认 690，图像底部区域)

    Returns:
        (ROI PIL Image, label)  label: 1=lane_in, 0=lane_out
    """
    image = Image.open(image_path).convert("RGB")

    # 1. 按规则判定 label
    left_x, right_x = find_ego_lane_boundaries(lanes, h_samples, y_threshold=check_y)
    center_x = IMAGE_WIDTH // 2  # 640

    if left_x is not None and right_x is not None and left_x <= center_x <= right_x:
        label = 1  # lane_in
    else:
        label = 0  # lane_out

    # 2. 裁剪车辆前方 ROI
    roi_top = max(0, IMAGE_HEIGHT - roi_height)  # 从底部向上取
    left = center_x - roi_width // 2
    right = center_x + roi_width // 2
    left = max(0, left)
    right = min(IMAGE_WIDTH, right)

    roi = image.crop((left, roi_top, right, min(IMAGE_HEIGHT, roi_top + roi_height)))

    return roi, label


def ensure_dir(dirpath: str) -> Path:
    """确保目录存在，不存在则创建。"""
    p = Path(dirpath)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_available_font(bold: bool = False) -> str:
    """跨平台查找可用字体路径，用于 PIL 图片文字标注。

    按优先级搜索常见中文字体，返回第一个存在的字体路径。
    若全部未找到，返回空字符串（调用方应回退到 PIL 默认字体）。

    Args:
        bold: 是否优先选择粗体字体

    Returns:
        字体文件路径，找不到则返回 ""
    """
    import platform
    import os as _os

    is_windows = platform.system() == "Windows"
    is_macos = platform.system() == "Darwin"

    if is_windows:
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/msyhbd.ttc",       # 微软雅黑 粗体
            "C:/Windows/Fonts/simsun.ttc",       # 宋体
            "C:/Windows/Fonts/simhei.ttf",       # 黑体
        ]
    elif is_macos:
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    # Bold variants first if requested
    if bold:
        bold_candidates = [
            "C:/Windows/Fonts/msyhbd.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        ]
        candidates = bold_candidates + candidates

    if not bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])

    for path in candidates:
        if _os.path.exists(path):
            return path
    return ""
